// Headless convex-die optimizer for Blender.
//
// Stock apps/main_convex_dice.cpp opens Polyscope and waits for the
// "dice energy opt" button. This file is the same loop with the draw
// calls removed. visualize_current_probs_and_goals takes the solver by
// value and only registers Polyscope quantities, so skipping it does
// not change the update.
//
// Build against resting-rigid-bodies with Polyscope OFF and inverse
// design ON. Do not define RESTING_RIGID_BODIES_USE_POLYSCOPE.
//
// Copy to apps/main_convex_dice_headless.cpp and add the target in
// solver/CMakeLists.txt.snippet.

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include "args.hxx"
#include "boundary_tools.h"
#include "file_IO.h"
#include "geometry_utils.h"
#include "inverse_design/dice_energy.h"
#include "inverse_design/prob_assignment.h"
#include "mesh_factory.h"
#include "utils.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace geometrycentral;
using namespace geometrycentral::surface;

namespace fs = std::filesystem;

static constexpr const char* kHandshake = "resting-dice-headless 1";

static Eigen::Vector3d gc_vec_to_eigen(const Vector3& v) {
    return Eigen::Vector3d(v.x, v.y, v.z);
}

static void load_mesh(const std::string& path,
                      ManifoldSurfaceMesh*& out_mesh,
                      VertexPositionGeometry*& out_geometry) {
    std::unique_ptr<SurfaceMesh> nm_mesh_ptr;
    std::unique_ptr<VertexPositionGeometry> nm_geometry_ptr;
    std::tie(nm_mesh_ptr, nm_geometry_ptr) = readSurfaceMesh(path);
    SurfaceMesh* nm_mesh = nm_mesh_ptr.release();
    VertexPositionGeometry* nm_geometry = nm_geometry_ptr.release();

    nm_mesh->greedilyOrientFaces();
    nm_mesh->compress();
    std::unique_ptr<ManifoldSurfaceMesh> manifold = nm_mesh->toManifoldMesh();
    out_mesh = manifold.release();
    out_geometry = new VertexPositionGeometry(*out_mesh);
    for (Vertex v : out_mesh->vertices()) {
        out_geometry->inputVertexPositions[v.getIndex()] =
            nm_geometry->inputVertexPositions[v.getIndex()];
    }
    delete nm_mesh;
    delete nm_geometry;
}

static void write_error_report(const fs::path& report_path, const std::string& message) {
    nlohmann::json j;
    j["ok"] = false;
    j["error"] = message;
    fs::create_directories(report_path.parent_path());
    std::ofstream out(report_path);
    if (out) {
        out << j.dump(2) << "\n";
    }
    std::cerr << message << "\n";
}

static nlohmann::json stable_faces_json(Forward3DSolver& solver) {
    BoundaryBuilder builder(&solver);
    builder.build_boundary_normals();
    Eigen::VectorXd areas = builder.face_region_area.toVector();
    solver.hullGeometry->requireFaceNormals();
    Eigen::VectorX<Vector3> normals = solver.hullGeometry->faceNormals.toVector();
    solver.hullGeometry->unrequireFaceNormals();

    nlohmann::json faces = nlohmann::json::array();
    for (Face f : solver.hullMesh->faces()) {
        double area = areas[f.getIndex()];
        if (area <= 0.0) {
            continue;
        }
        Vector3 n = normals[f.getIndex()];
        faces.push_back({
            {"probability", area / (4.0 * 3.14159265358979323846)},
            {"face_normal", {n.x, n.y, n.z}},
            {"hull_face_index", static_cast<int>(f.getIndex())},
        });
    }
    return faces;
}

struct OptResult {
    int iterations_run = 0;
    bool stopped_early = false;
    double final_dice_energy = 0.0;
    int assigned_sides = 0;
    std::string policy;
};

static OptResult optimize(ManifoldSurfaceMesh* mesh,
                          VertexPositionGeometry* geometry,
                          Vector3 G,
                          ConvexDiceParams& dice_params,
                          const std::string& policy_in,
                          bool use_loaded,
                          const std::vector<std::pair<Vector3, double>>& loaded_pairs,
                          const std::string& loaded_policy,
                          const fs::path& out_obj,
                          nlohmann::json& faces_out) {
    // Same prep as initialize_state, without init_visuals.
    preprocess_mesh(mesh, geometry, /*triangulate=*/false, /*do_remesh=*/false);
    center_and_normalize(mesh, geometry);

    Forward3DSolver tmp_solver(mesh, geometry, G, /*concave_input=*/true);
    tmp_solver.set_uniform_G();
    tmp_solver.initialize_pre_computes();
    G = tmp_solver.get_G();
    Eigen::Vector3d G_vec = gc_vec_to_eigen(G);
    Eigen::MatrixX3d hull_positions =
        vertex_data_to_matrix(tmp_solver.hullGeometry->inputVertexPositions);

    std::string policy_general = policy_in;
    std::string policy_shape;
    auto space = policy_in.find(' ');
    if (space != std::string::npos) {
        policy_general = policy_in.substr(0, space);
        policy_shape = policy_in.substr(space + 1);
    }

    std::vector<std::pair<Vector3, double>> normal_prob_pairs;
    if (use_loaded) {
        normal_prob_pairs = loaded_pairs;
        if (!loaded_policy.empty()) {
            policy_general = loaded_policy;
        }
        std::cout << "using loaded assignment, policy " << policy_general << "\n";
    } else if (policy_general == "fairCluster") {
        // Upstream calls normal_prob_assignment(filename) first. For a sword
        // that returns empty, then fairCluster overwrites it. Skip the empty
        // call. The face pick is the part that matters.
        (void)policy_shape;
        normal_prob_pairs = normal_prob_assignment_fair(&tmp_solver, dice_params.fair_sides_count);
        policy_general = "manualCluster";
    } else {
        throw std::runtime_error(
            "pass --normal_probs, or --policy fairCluster. Named policies "
            "(cube, circus) are not a sword-to-die path.");
    }
    if (normal_prob_pairs.empty()) {
        throw std::runtime_error(
            "no target faces. fairCluster only sees stable faces that already "
            "exist; this hull has fewer than requested.");
    }

    auto initial_pairs = normal_prob_pairs;
    OptResult result;
    result.assigned_sides = static_cast<int>(initial_pairs.size());
    result.policy = policy_general;

    double current_sobolev_lambda = dice_params.sobolev_lambda;
    double default_ls_step = dice_params.dice_energy_step;
    const double step_tol = 1e-9;
    Eigen::MatrixX3d dfdV;
    const size_t step_count = static_cast<size_t>(std::max(dice_params.DE_step_count, 0));

    for (size_t iter = 0; iter < step_count; ++iter) {
        Eigen::Vector3d dfdG;
        dfdV = Eigen::MatrixX3d::Zero(hull_positions.rows(), 3);
        double dice_e = 0.0;
        get_dice_energy_grads(
            hull_positions, G_vec,
            dice_params.bary_reg, dice_params.coplanar_reg,
            dice_params.cluster_distance_reg, dice_params.unstable_attraction_thresh,
            dfdV, dfdG, dice_e, dice_params.frozen_G,
            policy_general, normal_prob_pairs, dice_params.fair_sides_count);
        result.final_dice_energy = dice_e;
        result.iterations_run = static_cast<int>(iter + 1);
        std::cout << "i: " << iter << "\tDE: " << dice_e << std::endl;

        if (dice_params.do_sobolev_dice_grads) {
            current_sobolev_lambda *= dice_params.sobolev_lambda_decay;
            dfdV = sobolev_diffuse_gradients(
                dfdV, *tmp_solver.hullMesh, *tmp_solver.hullGeometry,
                current_sobolev_lambda, static_cast<size_t>(dice_params.sobolev_p));
        }

        double opt_step_size = 1.0;
        if (dice_params.dice_search_decay < 1.0) {
            opt_step_size = hull_update_line_search(
                dfdV, hull_positions, G_vec,
                dice_params.bary_reg, dice_params.coplanar_reg,
                dice_params.cluster_distance_reg, dice_params.unstable_attraction_thresh,
                policy_general, normal_prob_pairs,
                static_cast<size_t>(dice_params.fair_sides_count),
                default_ls_step, dice_params.dice_search_decay,
                dice_params.frozen_G, /*max_iter=*/1000, step_tol);
        }
        std::cout << "  line search step size: " << opt_step_size << "\n";
        if (opt_step_size < step_tol) {
            result.stopped_early = true;
            std::cout << "  line search step size too small; breaking\n";
            break;
        }

        hull_positions = hull_positions - opt_step_size * dfdV;
        tmp_solver = Forward3DSolver(hull_positions, G_vec, /*concave_input=*/false);
        if (!dice_params.frozen_G) {
            tmp_solver.set_uniform_G();
            G_vec = gc_vec_to_eigen(tmp_solver.get_G());
        }
        hull_positions = vertex_data_to_matrix(tmp_solver.hullGeometry->inputVertexPositions);
        tmp_solver.initialize_pre_computes();
        if (policy_general != "fair") {
            auto clustered = manual_clustered_face_prob_assignment(&tmp_solver, normal_prob_pairs);
            normal_prob_pairs = update_normal_prob_assignment(
                &tmp_solver, clustered, dice_params.update_with_max_prob_face);
        }
    }

    // Write before tmp_solver dies. Upstream stashes raw pointers and saves
    // later from the GUI; that is a dangling use if the destructor frees them.
    fs::create_directories(out_obj.parent_path());
    writeSurfaceMesh(*tmp_solver.hullMesh, *tmp_solver.hullGeometry, out_obj.string());
    faces_out = stable_faces_json(tmp_solver);

    fs::path dir = out_obj.parent_path();
    save_vector3_double_pairs(initial_pairs, (dir / "initial_normal_probs.json").string(), result.policy);
    save_vector3_double_pairs(normal_prob_pairs, (dir / "final_normal_probs.json").string(), result.policy);
    save_convex_dice_params(dice_params, (dir / "params.json").string());
    if (dice_params.frozen_G) {
        std::ofstream gfile(dir / ("G_" + out_obj.stem().string() + ".txt"));
        gfile << G.x << " " << G.y << " " << G.z << "\n";
    }
    return result;
}

int main(int argc, char** argv) {
    args::ArgumentParser parser("Headless dice-energy optimization. No Polyscope.");
    args::HelpFlag help(parser, "help", "show help", {'h', "help"});
    args::Flag headless(parser, "headless", "required handshake; this binary has no GUI", {"headless"});
    args::ValueFlag<std::string> mesh_arg(parser, "mesh", "input mesh", {'m', "mesh"});
    args::ValueFlag<std::string> out_arg(parser, "out", "optimized OBJ", {'o', "out"});
    args::ValueFlag<std::string> params_arg(parser, "params", "params JSON", {"params"});
    args::ValueFlag<std::string> policy_arg(parser, "policy", "fairCluster or manualCluster", {'p', "policy"}, "manualCluster");
    args::ValueFlag<std::string> normals_arg(parser, "normal_probs", "target assignment JSON", {"normal_probs"});
    args::ValueFlag<std::string> report_arg(parser, "report", "report JSON", {"report"});
    args::ValueFlag<int> sides_arg(parser, "sides", "overrides fair_sides_count", {"sides"});

    try {
        parser.ParseCLI(argc, argv);
    } catch (const args::Help&) {
        std::cout << parser;
        return 0;
    } catch (const args::ParseError& e) {
        std::cerr << e.what() << "\n" << parser;
        return 1;
    } catch (const args::ValidationError& e) {
        std::cerr << e.what() << "\n" << parser;
        return 1;
    }

    if (!headless || !mesh_arg || !out_arg || !report_arg) {
        std::cerr << "required: --headless --mesh --out --report\n";
        return 2;
    }

    fs::path report_path = args::get(report_arg);
    try {
        std::cout << kHandshake << "\n";
        ConvexDiceParams dice_params;
        if (params_arg) {
            if (!load_convex_dice_params(dice_params, args::get(params_arg))) {
                throw std::runtime_error("failed to load params");
            }
        }
        if (sides_arg) {
            dice_params.fair_sides_count = args::get(sides_arg);
        }
        if (dice_params.fair_sides_count < 2) {
            throw std::runtime_error("fair_sides_count must be >= 2");
        }

        bool use_loaded = false;
        std::vector<std::pair<Vector3, double>> loaded_pairs;
        std::string loaded_policy;
        if (normals_arg) {
            loaded_pairs = load_vector3_double_pairs(args::get(normals_arg), &loaded_policy);
            if (loaded_pairs.empty()) {
                throw std::runtime_error("normal-probability file has no pairs");
            }
            use_loaded = true;
        }

        ManifoldSurfaceMesh* mesh = nullptr;
        VertexPositionGeometry* geometry = nullptr;
        load_mesh(args::get(mesh_arg), mesh, geometry);
        Vector3 G = Vector3::zero();
        nlohmann::json faces;
        OptResult result = optimize(
            mesh, geometry, G, dice_params, args::get(policy_arg),
            use_loaded, loaded_pairs, loaded_policy, args::get(out_arg), faces);

        nlohmann::json report;
        report["ok"] = true;
        report["error"] = "";
        report["policy"] = result.policy;
        report["requested_sides"] = dice_params.fair_sides_count;
        report["assigned_sides"] = result.assigned_sides;
        report["iterations_run"] = result.iterations_run;
        report["stopped_early"] = result.stopped_early;
        report["final_dice_energy"] = result.final_dice_energy;
        report["output_obj"] = fs::absolute(args::get(out_arg)).string();
        report["stable_faces"] = faces;
        if (result.assigned_sides < dice_params.fair_sides_count) {
            report["warning"] =
                "assigned fewer faces than requested. fairCluster cannot invent faces.";
        } else {
            report["warning"] = "";
        }
        fs::create_directories(report_path.parent_path());
        std::ofstream report_out(report_path);
        report_out << report.dump(2) << "\n";
        std::cout << "wrote " << report_path << "\n";
        delete mesh;
        delete geometry;
        return 0;
    } catch (const std::exception& e) {
        write_error_report(report_path, e.what());
        return 1;
    }
}
