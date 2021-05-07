#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/differentiation/ad.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include "modded_periodic_functions.h"
#include "parameters.h"
#include "postprocessing.h"

using namespace dealii;

template <int dim>
class ComposedFunction: public Function<dim> {
  public:
    ComposedFunction();
    void vector_value(const Point<dim>& p, Vector<double>& values)
            const override;
    std::shared_ptr<Function<dim>> f1;
    std::shared_ptr<Function<dim>> f2;
    double gamma {1};
};

template <int dim>
ComposedFunction<dim>::ComposedFunction() {
    f1 = std::make_shared<Functions::ZeroFunction<dim>>(3);
    f2 = std::make_shared<Functions::ZeroFunction<dim>>(3);
}

template <int dim>
void ComposedFunction<dim>::vector_value(
        const Point<dim>& p,
        Vector<double>& values) const {

    Vector<double> values_1(3);
    f1->vector_value(p, values_1);
    Vector<double> values_2(3);
    f2->vector_value(p, values_2);
    values.add(1 - gamma, values_1, gamma, values_2);
}

template <int dim>
class SolveRing {
  public:
    SolveRing(Params<dim>& prms);
    void run();

  private:
    void make_grid();
    void initiate_system();
    void setup_constraints(unsigned int stage_i);
    void update_constraints();
    void update_boundary_function_gamma();
    void update_boundary_function_stage(unsigned int stage_i);
    void setup_sparsity_pattern();
    void assemble(const bool initial_step, const bool assemble_matrix);
    void assemble_system(const bool initial_step);
    void assemble_rhs(const bool initial_step);
    void solve(const bool initial_step);
    double calc_residual_norm();
    void newton_iteration(bool first_step, const std::string checkpoint);
    void center_solution_on_mean();
    void refine_mesh(unsigned int stage_i);
    void move_mesh();
    void integrate_over_boundaries();
    void output_grid() const;
    void output_checkpoint(const std::string checkpoint) const;
    void output_results(const std::string checkpoint) const;
    void output_moved_mesh_results(const std::string checkpoint) const;
    void load_checkpoint(const std::string checkpoint);
    std::string format_gamma();

    Triangulation<dim> triangulation;
    FESystem<dim> fe;
    DoFHandler<dim> dof_handler;
    AffineConstraints<double> zero_constraints;
    AffineConstraints<double> nonzero_constraints;
    Functions::ZeroFunction<dim> zero_function;
    ComposedFunction<dim> boundary_function;

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> present_solution;
    Vector<double> newton_update;
    Vector<double> system_rhs;
    Vector<double> evaluation_point;
    double present_energy;

    std::vector<Point<dim>> dofs_to_supports;
    std::vector<GridTools::PeriodicFacePair<
            typename DoFHandler<dim, dim>::cell_iterator>>
            face_pairs;
    double gamma;

    Params<dim>& prms;
    double lambda;
    double mu;
};

template <int dim>
SolveRing<dim>::SolveRing(Params<dim>& prms):
        fe(FE_Q<dim>(1), dim),
        dof_handler(triangulation),
        zero_function {3},
        prms {prms} {
    mu = prms.E / (2 * (1 + prms.nu));
    lambda = prms.E * prms.nu / ((1 + prms.nu) * (1 - 2 * prms.nu));
}

template <int dim>
void SolveRing<dim>::run() {
    make_grid();
    output_grid();
    std::string checkpoint;
    bool first_step {true};
    if (prms.load_from_checkpoint) {
        first_step = false;
        load_checkpoint(prms.input_checkpoint);
    }
    else {
        initiate_system();
    }
    setup_constraints(prms.starting_stage);
    setup_sparsity_pattern();
    for (unsigned int stage_i {prms.starting_stage};
         stage_i != prms.num_boundary_stages;
         stage_i++) {
        update_boundary_function_stage(stage_i);
        unsigned int num_gamma_iters {prms.num_gamma_iters[stage_i]};
        for (unsigned int i {0}; i != num_gamma_iters; i++) {
            gamma = static_cast<double>((i + 1)) / num_gamma_iters;
            std::string gamma_formatted {format_gamma()};
            checkpoint = std::to_string(stage_i + 1) + "-" + gamma_formatted;
            cout << "Stage " << std::to_string(stage_i + 1) << ", gamma "
                 << gamma_formatted << std::endl;
            update_constraints();
            newton_iteration(first_step, checkpoint);
            output_checkpoint(checkpoint);
            first_step = false;
        }
    }

    checkpoint = checkpoint =
            std::to_string(prms.num_boundary_stages + 1) + "_" + "refine-0";
    output_checkpoint(checkpoint);
    output_results(checkpoint);
    integrate_over_boundaries();

    for (unsigned int i {0}; i != prms.adaptive_refinements; i++) {
        cout << "Grid refinement " << std::to_string(i + 1) << std::endl;
        checkpoint = "refine-" + std::to_string(i + 1);
        refine_mesh(prms.num_boundary_stages - 1);
        newton_iteration(first_step, checkpoint);
        output_checkpoint(checkpoint);
        integrate_over_boundaries();
    }

    checkpoint = "moved-mesh";
    // move_mesh();
    // output_moved_mesh_results(checkpoint);
}

template <int dim>
void SolveRing<dim>::make_grid() {
    const Point<dim>& origin {0, 0, 0};
    const Point<dim>& size {prms.length, prms.width, prms.width};
    const std::vector<unsigned int> subdivisions {
            prms.x_subdivisions, prms.y_subdivisions, prms.z_subdivisions};
    GridGenerator::subdivided_hyper_rectangle(
            triangulation, subdivisions, origin, size);
    for (auto& face: triangulation.active_face_iterators()) {
        if (std::fabs(face->center()(0)) < 1e-12) {
            face->set_boundary_id(1);
        }
        else if (std::fabs(face->center()(0) - prms.length) < 1e-12) {
            face->set_boundary_id(2);
        }
    }
}

template <int dim>
void SolveRing<dim>::initiate_system() {
    dof_handler.distribute_dofs(fe);
    present_solution.reinit(dof_handler.n_dofs());
    newton_update.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
}

template <int dim>
void SolveRing<dim>::setup_constraints(unsigned int stage_i) {
    const MappingQ1<dim> mapping;
    dofs_to_supports.resize(dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points<dim, dim>(
            mapping, dof_handler, dofs_to_supports);
    collect_periodic_faces(dof_handler, 1, 2, 0, face_pairs);

    update_boundary_function_stage(stage_i);
    nonzero_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, nonzero_constraints);
    make_periodicity_constraints<dim, dim, double>(
            face_pairs,
            nonzero_constraints,
            dofs_to_supports,
            boundary_function);
    nonzero_constraints.close();

    zero_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints);
    make_periodicity_constraints<dim, dim, double>(
            face_pairs, zero_constraints, dofs_to_supports, zero_function);
    zero_constraints.close();

    cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
         << std::endl;
}

template <int dim>
void SolveRing<dim>::update_constraints() {
    update_boundary_function_gamma();
    nonzero_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, nonzero_constraints);
    make_periodicity_constraints<dim, dim, double>(
            face_pairs,
            nonzero_constraints,
            dofs_to_supports,
            boundary_function);
    nonzero_constraints.close();
}

template <int dim>
void SolveRing<dim>::update_boundary_function_gamma() {
    boundary_function.gamma = gamma;
}

template <int dim>
void SolveRing<dim>::update_boundary_function_stage(unsigned int stage_i) {
    boundary_function.f1 = prms.boundary_functions[stage_i];
    boundary_function.f2 = prms.boundary_functions[stage_i + 1];
}

template <int dim>
void SolveRing<dim>::setup_sparsity_pattern() {
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, nonzero_constraints);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
}

template <int dim>
void SolveRing<dim>::assemble(
        const bool initial_step,
        const bool assemble_matrix) {
    if (assemble_matrix) {
        system_matrix = 0;
    }
    system_rhs = 0;
    present_energy = 0;

    QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values(
            fe,
            quadrature_formula,
            update_values | update_gradients | update_quadrature_points |
                    update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    const FEValuesExtractors::Vector displacements(0);

    using ADHelper = Differentiation::AD::EnergyFunctional<
            Differentiation::AD::NumberTypes::sacado_dfad_dfad,
            double>;
    using ADNumberType = typename ADHelper::ad_type;

    for (const auto& cell: dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);
        const unsigned int n_independent_variables = local_dof_indices.size();

        ADHelper ad_helper(n_independent_variables);
        ad_helper.register_dof_values(evaluation_point, local_dof_indices);
        const std::vector<ADNumberType>& dof_values_ad =
                ad_helper.get_sensitive_dof_values();
        ADNumberType energy_ad = ADNumberType(0.0);

        std::vector<Tensor<2, dim, ADNumberType>> old_solution_gradients(
                fe_values.n_quadrature_points);
        fe_values[displacements].get_function_gradients_from_local_dof_values(
                dof_values_ad, old_solution_gradients);

        for (const unsigned int q_index: fe_values.quadrature_point_indices()) {
            const Tensor<2, dim, ADNumberType> grad_u {
                    old_solution_gradients[q_index]};
            const Tensor<2, dim, ADNumberType> grad_u_T {transpose(grad_u)};
            const Tensor<2, dim, ADNumberType> green_lagrange_strain_tensor {
                    0.5 * (grad_u + grad_u_T + grad_u * grad_u_T)};
            ADNumberType t1 = lambda / 2 *
                              std::pow(trace(green_lagrange_strain_tensor), 2);
            ADNumberType t2 = mu * trace(green_lagrange_strain_tensor *
                                         green_lagrange_strain_tensor);
            ADNumberType pi {t1 + t2};
            energy_ad += pi * fe_values.JxW(q_index);
        }

        ad_helper.register_energy_functional(energy_ad);
        present_energy += ad_helper.compute_energy();
        ad_helper.compute_residual(cell_rhs);
        cell_rhs *= -1.0; // RHS = - residual
        if (assemble_matrix) {
            ad_helper.compute_linearization(cell_matrix);
        }

        const AffineConstraints<double>& constraints_used =
                initial_step ? nonzero_constraints : zero_constraints;
        if (assemble_matrix) {
            constraints_used.distribute_local_to_global(
                    cell_matrix,
                    cell_rhs,
                    local_dof_indices,
                    system_matrix,
                    system_rhs,
                    true);
        }
        else {
            constraints_used.distribute_local_to_global(
                    cell_rhs, local_dof_indices, system_rhs);
        }
    }
}

template <int dim>
void SolveRing<dim>::assemble_system(const bool initial_step) {
    assemble(initial_step, true);
}

template <int dim>
void SolveRing<dim>::assemble_rhs(const bool initial_step) {
    assemble(initial_step, false);
}

template <int dim>
void SolveRing<dim>::solve(const bool initial_step) {
    newton_update = 0;
    const AffineConstraints<double>& constraints_used =
            initial_step ? nonzero_constraints : zero_constraints;

    if (prms.linear_solver == "iterative") {
        SolverControl solver_control(prms.max_linear_iters, prms.linear_tol);
        SolverGMRES<Vector<double>> solver(solver_control);
        PreconditionSSOR<SparseMatrix<double>> preconditioner;
        preconditioner.initialize(system_matrix, prms.precon_relaxation_param);
        solver.solve(system_matrix, newton_update, system_rhs, preconditioner);
    }
    else if (prms.linear_solver == "direct") {
        SparseDirectUMFPACK A_direct;
        A_direct.initialize(system_matrix);
        A_direct.vmult(newton_update, system_rhs);
    }

    constraints_used.distribute(newton_update);
}

template <int dim>
double SolveRing<dim>::calc_residual_norm() {
    Vector<double> residual(dof_handler.n_dofs());
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i) {
        if (!nonzero_constraints.is_constrained(i)) {
            residual(i) = system_rhs(i);
        }
    }

    return residual.l2_norm();
}

template <int dim>
void SolveRing<dim>::newton_iteration(
        bool first_step,
        const std::string checkpoint) {

    bool boundary_updated {!first_step};
    unsigned int line_search_n = 0;
    double last_res = 1.0;
    double current_res = 1.0;

    while ((first_step or (current_res > prms.nonlinear_tol)) and
           line_search_n < prms.max_n_line_searches) {
        if (first_step) {
            evaluation_point = present_solution;
            assemble_system(first_step);
            solve(first_step);
            present_solution = newton_update;
            first_step = false;
            evaluation_point = present_solution;
            assemble_rhs(first_step);
            current_res = calc_residual_norm();
            std::cout << "The residual of initial guess is " << current_res
                      << std::endl;
            std::cout << "The energy of initial guess is " << present_energy
                      << std::endl;
            last_res = current_res;
        }
        else {
            nonzero_constraints.distribute(present_solution);
            evaluation_point = present_solution;
            assemble_system(first_step);
            solve(first_step);
            for (double alpha {1.0}; alpha > prms.min_alpha;
                 alpha *= prms.alpha_factor) {

                evaluation_point = present_solution;
                evaluation_point.add(alpha, newton_update);
                assemble_rhs(first_step);
                current_res = calc_residual_norm();
                std::cout << "  alpha: " << std::setw(10) << alpha
                          << std::setw(0) << "  residual: " << current_res
                          << std::endl;
                if (boundary_updated or
                    ((last_res - current_res) >=
                     (alpha * last_res * prms.alpha_check_factor))) {
                    boundary_updated = false;
                    break;
                }
            }
            present_solution = evaluation_point;
            std::cout << "  number of line searches: " << line_search_n
                      << "  residual: " << current_res
                      << "  energy: " << present_energy << std::endl;
            last_res = current_res;
            ++line_search_n;
            center_solution_on_mean();
        }
    }
    output_results(checkpoint);
}

template <int dim>
void SolveRing<dim>::center_solution_on_mean() {
    Vector<double> mean_u(3);
    for (unsigned int i {0}; i != dof_handler.n_dofs(); i++) {
        const unsigned int component_i {i % dim};
        mean_u[component_i] += present_solution[i];
    }
    mean_u /= static_cast<double>(dof_handler.n_dofs()) / 3;
    for (unsigned int i {0}; i != dof_handler.n_dofs(); i++) {
        const unsigned int component_i {i % dim};
        present_solution[i] -= mean_u[component_i];
    }
}

template <int dim>
void SolveRing<dim>::refine_mesh(unsigned int stage_i) {
    Vector<double> cells_to_refine(triangulation.n_active_cells());
    cells_to_refine = 1;
    GridRefinement::refine(triangulation, cells_to_refine, 0);
    triangulation.prepare_coarsening_and_refinement();
    SolutionTransfer<dim, Vector<double>> soltrans(dof_handler);
    soltrans.prepare_for_pure_refinement();
    triangulation.execute_coarsening_and_refinement();

    dof_handler.distribute_dofs(fe);
    Vector<double> interpolated_solution(dof_handler.n_dofs());
    soltrans.refine_interpolate(present_solution, interpolated_solution);

    present_solution.reinit(dof_handler.n_dofs());
    newton_update.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
    setup_constraints(stage_i);
    setup_sparsity_pattern();
    present_solution = interpolated_solution;
}

template <int dim>
void SolveRing<dim>::move_mesh() {
    std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
    for (auto& cell: dof_handler.active_cell_iterators()) {
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
             ++v) {
            if (vertex_touched[cell->vertex_index(v)] == false) {
                vertex_touched[cell->vertex_index(v)] = true;
                Point<dim> vertex_displacement;
                for (unsigned int d = 0; d < dim; ++d)
                    vertex_displacement[d] =
                            present_solution(cell->vertex_dof_index(v, d));
                cell->vertex(v) += vertex_displacement;
            }
        }
    }
}

template <int dim>
void SolveRing<dim>::integrate_over_boundaries() {
    QGauss<dim - 1> quadrature_formula(fe.degree + 1);
    FEFaceValues<dim> fe_face_values(
            fe,
            quadrature_formula,
            update_values | update_gradients | update_quadrature_points |
                    update_JxW_values | update_normal_vectors);

    std::vector<Tensor<1, dim, double>> material_force(2);
    std::vector<Tensor<1, dim, double>> spatial_force(2);
    std::vector<Tensor<1, dim, double>> ave_material_normal(2);
    std::vector<Tensor<1, dim, double>> ave_spatial_normal(2);
    const FEValuesExtractors::Vector displacements(0);
    for (const auto& cell: dof_handler.active_cell_iterators()) {
        for (const auto face_i: GeometryInfo<dim>::face_indices()) {
            const unsigned int boundary_id {cell->face(face_i)->boundary_id()};
            if (not(boundary_id == 1 or boundary_id == 2)) {
                continue;
            }
            fe_face_values.reinit(cell, face_i);
            std::vector<Tensor<1, dim, double>> normal_vectors {
                    fe_face_values.get_normal_vectors()};
            std::vector<Tensor<2, dim, double>> solution_gradients(
                    fe_face_values.n_quadrature_points);
            fe_face_values[displacements].get_function_gradients(
                    present_solution, solution_gradients);
            for (const auto q_i: fe_face_values.quadrature_point_indices()) {
                const Tensor<2, dim, double> grad_u {solution_gradients[q_i]};
                Tensor<2, dim, double> identity_rank2 {};
                for (unsigned int d {0}; d != dim; d++) {
                    identity_rank2[d][d] = 1;
                }
                const Tensor<1, dim, double> material_normal {
                        normal_vectors[q_i]};

                const Tensor<2, dim, double> grad_u_T {transpose(grad_u)};
                const Tensor<2, dim, double> green_lagrange_strain {
                        0.5 * (grad_u + grad_u_T + grad_u * grad_u_T)};
                const Tensor<2, dim, double> piola_kirchhoff {
                        lambda * trace(green_lagrange_strain) * identity_rank2 +
                        mu * green_lagrange_strain};

                ave_material_normal[boundary_id - 1] += material_normal;
                material_force[boundary_id - 1] += piola_kirchhoff *
                                                   material_normal *
                                                   fe_face_values.JxW(q_i);

                const Tensor<2, dim, double> deformation_grad {
                        grad_u + identity_rank2};
                const double deformation_grad_det {
                        determinant(deformation_grad)};
                const Tensor<2, dim, double> cauchy {
                        deformation_grad * piola_kirchhoff *
                        transpose(deformation_grad) / deformation_grad_det};

                Tensor<1, dim, double> spatial_normal {
                        transpose(invert(deformation_grad)) * material_normal};
                spatial_normal /= spatial_normal.norm();
                ave_spatial_normal[boundary_id - 1] += spatial_normal;
                spatial_force[boundary_id - 1] +=
                        cauchy * spatial_normal * fe_face_values.JxW(q_i);
            }
        }
    }
    ave_material_normal[0] /= ave_material_normal[0].norm();
    ave_spatial_normal[0] /= ave_spatial_normal[0].norm();
    ave_material_normal[1] /= ave_material_normal[1].norm();
    ave_spatial_normal[1] /= ave_spatial_normal[1].norm();
    const Tensor<1, dim, double> left_material_normal_force = {
            (material_force[0] * ave_material_normal[0]) *
            ave_material_normal[0]};
    const Tensor<1, dim, double> left_material_shear_force = {
            material_force[0] - left_material_normal_force};
    const Tensor<1, dim, double> left_spatial_normal_force = {
            (spatial_force[0] * ave_spatial_normal[0]) * ave_spatial_normal[0]};
    const Tensor<1, dim, double> left_spatial_shear_force = {
            spatial_force[0] - left_spatial_normal_force};
    const Tensor<1, dim, double> right_material_normal_force = {
            (material_force[1] * ave_material_normal[1]) *
            ave_material_normal[1]};
    const Tensor<1, dim, double> right_material_shear_force = {
            material_force[1] - right_material_normal_force};
    const Tensor<1, dim, double> right_spatial_normal_force = {
            (spatial_force[1] * ave_spatial_normal[1]) * ave_spatial_normal[1]};
    const Tensor<1, dim, double> right_spatial_shear_force = {
            spatial_force[1] - right_spatial_normal_force};

    cout << "Left boundary material normal force: "
         << left_material_normal_force.norm() << std::endl;
    cout << "Right boundary material normal force: "
         << right_material_normal_force.norm() << std::endl;
    cout << "Left boundary material shear force: "
         << left_material_shear_force.norm() << std::endl;
    cout << "Right boundary material shear force: "
         << right_material_shear_force.norm() << std::endl;
    cout << "Left boundary spatial normal force: "
         << left_spatial_normal_force.norm() << std::endl;
    cout << "Right boundary spatial normal force: "
         << right_spatial_normal_force.norm() << std::endl;
    cout << "Left boundary spatial shear force: "
         << left_spatial_shear_force.norm() << std::endl;
    cout << "Right boundary spatial shear force: "
         << right_spatial_shear_force.norm() << std::endl;
}

template <int dim>
void SolveRing<dim>::output_grid() const {
    GridOut grid_out;
    std::ofstream grid_output {prms.output_prefix + "_mesh.vtk"};
    grid_out.write_vtk(triangulation, grid_output);
}

template <int dim>
void SolveRing<dim>::output_checkpoint(const std::string checkpoint) const {
    std::ofstream solution_out {
            prms.output_prefix + "_displacement_" + checkpoint + ".ar"};
    boost::archive::text_oarchive solution_ar {solution_out};
    present_solution.save(solution_ar, 0);

    std::ofstream triangulation_out {
            prms.output_prefix + "_triangulation_" + checkpoint + ".ar"};
    boost::archive::text_oarchive triangulation_ar {triangulation_out};
    triangulation.save(triangulation_ar, 0);

    std::ofstream dof_handler_out {
            prms.output_prefix + "_dof_handler_" + checkpoint + ".ar"};
    boost::archive::text_oarchive dof_handler_ar {dof_handler_out};
    dof_handler.save(dof_handler_ar, 0);
}

template <int dim>
void SolveRing<dim>::output_results(const std::string checkpoint) const {
    DataOutFaces<dim> data_out_faces;
    data_out_faces.attach_dof_handler(dof_handler);
    FacesPostprocessor<dim> faces_postprocessor {lambda, mu};
    data_out_faces.add_data_vector(present_solution, faces_postprocessor);
    data_out_faces.build_patches();

    std::ofstream data_output_faces(
            prms.output_prefix + "_faces_" + checkpoint + ".vtk");
    data_out_faces.write_vtk(data_output_faces);
}

template <int dim>
void SolveRing<dim>::output_moved_mesh_results(
        const std::string checkpoint) const {

    SpatialStressVectorMovedMeshPostprocess<dim> postprocessor {lambda, mu};

    DataOutFaces<dim> data_out_faces;
    data_out_faces.attach_dof_handler(dof_handler);
    data_out_faces.add_data_vector(present_solution, postprocessor);
    data_out_faces.build_patches();

    std::ofstream data_output_faces(
            prms.output_prefix + "_faces-moved-mesh_" + checkpoint + ".vtk");
    data_out_faces.write_vtk(data_output_faces);
}

template <int dim>
void SolveRing<dim>::load_checkpoint(const std::string checkpoint) {
    std::ifstream solution_inp {
            prms.input_prefix + "_displacement_" + checkpoint + ".ar"};
    boost::archive::text_iarchive solution_ar {solution_inp};
    present_solution.load(solution_ar, 0);

    std::ifstream triangulation_inp {
            prms.input_prefix + "_triangulation_" + checkpoint + ".ar"};
    boost::archive::text_iarchive triangulation_ar {triangulation_inp};
    triangulation.load(triangulation_ar, 0);

    dof_handler.distribute_dofs(fe);
    std::ifstream dof_handler_inp {
            prms.input_prefix + "_dof_handler_" + checkpoint + ".ar"};
    boost::archive::text_iarchive dof_handler_ar {dof_handler_inp};
    dof_handler.load(dof_handler_ar, 0);

    newton_update.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
}

template <int dim>
std::string SolveRing<dim>::format_gamma() {
    std::ostringstream stream_obj;
    stream_obj << std::fixed;
    stream_obj << std::setprecision(prms.gamma_precision);
    stream_obj << gamma;
    return stream_obj.str();
}

int main() {
    deallog.depth_console(0);
    Params<3> prms {};
    SolveRing<3> ring_solver {prms};
    ring_solver.run();

    return 0;
}
