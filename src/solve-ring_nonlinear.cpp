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

namespace solve_ring {

using namespace parameters;
using namespace postprocessing;
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
ComposedFunction<dim>::ComposedFunction(): Function<dim>(3) {
    f1 = std::make_shared<Functions::ZeroFunction<dim>>(3);
    f2 = std::make_shared<Functions::ZeroFunction<dim>>(3);
}

template <int dim>
void ComposedFunction<dim>::vector_value(
        const Point<dim>& p,
        Vector<double>& values) const {
    values[0] = 0;
    values[1] = 0;
    values[2] = 0;
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
    void setup_side_to_side_periodic_constraints();
    template <typename FaceIterator>
    void set_overlap_constraints(
            FaceIterator& left_face,
            FaceIterator& right_face);
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
    void center_solution_on_vertex();
    void refine_mesh(unsigned int stage_i);
    void move_mesh();
    void integrate_over_boundaries();
    void output_grid() const;
    void output_checkpoint(const std::string checkpoint) const;
    void output_results(const std::string checkpoint) const;
    void output_moved_mesh_results(const std::string checkpoint) const;
    void load_checkpoint(const std::string checkpoint);
    std::string format_gamma();

    Params<dim>& prms;
    double lambda;
    double mu;

    Triangulation<dim> triangulation;
    FESystem<dim> fe;
    DoFHandler<dim> dof_handler;
    AffineConstraints<double> zero_constraints;
    AffineConstraints<double> nonzero_constraints;
    Functions::ZeroFunction<dim> zero_function;
    std::vector<ComposedFunction<dim>> boundary_function;
    std::pair<double, double> boundary {};

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> initial_stage_solution;
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
};

template <int dim>
SolveRing<dim>::SolveRing(Params<dim>& prms):
        prms {prms},
        fe(FE_Q<dim>(1), dim),
        dof_handler(triangulation),
        zero_function {3},
        boundary_function(prms.num_boundary_conditions) {
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
         stage_i != prms.num_boundary_stages + 1;
         stage_i++) {
        update_boundary_function_stage(stage_i);
        unsigned int num_gamma_iters {prms.num_gamma_iters[stage_i]};
        for (unsigned int i {0}; i != num_gamma_iters; i++) {
            gamma = static_cast<double>((i + 1)) / num_gamma_iters;
            std::string gamma_formatted {format_gamma()};
            checkpoint = std::to_string(stage_i) + "-" +
                         std::to_string(prms.starting_refinement) + "-" +
                         gamma_formatted;
            cout << "Stage " << std::to_string(stage_i) << ", gamma "
                 << gamma_formatted << std::endl;
            update_constraints();
            newton_iteration(first_step, checkpoint);
            output_checkpoint(checkpoint);
            first_step = false;
        }
    }
    integrate_over_boundaries();
    for (unsigned int i {prms.starting_refinement + 1};
         i != prms.starting_refinement + prms.adaptive_refinements + 1;
         i++) {
        cout << "Grid refinement " << std::to_string(i) << std::endl;
        checkpoint = std::to_string(prms.num_boundary_stages) + "-" +
                     std::to_string(i);
        refine_mesh(prms.num_boundary_stages);
        newton_iteration(first_step, checkpoint);
        output_checkpoint(checkpoint);
        integrate_over_boundaries();
    }

    // checkpoint = "moved-mesh";
    // move_mesh();
    // output_moved_mesh_results(checkpoint);
}

template <int dim>
void SolveRing<dim>::make_grid() {
    const Point<dim>& origin {0, 0, 0};
    const Point<dim>& size {prms.beam_X, prms.beam_Y, prms.beam_Z};
    const std::vector<unsigned int> subdivisions {
            prms.x_subdivisions, prms.y_subdivisions, prms.z_subdivisions};
    GridGenerator::subdivided_hyper_rectangle(
            triangulation, subdivisions, origin, size);
    for (auto& face: triangulation.active_face_iterators()) {
        if (std::fabs(face->center()(0)) < 1e-12) {
            face->set_boundary_id(1);
        }
        else if (std::fabs(face->center()(0) - prms.beam_X) < 1e-12) {
            face->set_boundary_id(2);
        }
    }
}

template <int dim>
void SolveRing<dim>::initiate_system() {
    dof_handler.distribute_dofs(fe);
    present_solution.reinit(dof_handler.n_dofs());
    initial_stage_solution.reinit(dof_handler.n_dofs());
    newton_update.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
}

template <int dim>
void SolveRing<dim>::setup_constraints(unsigned int stage_i) {
    update_boundary_function_stage(stage_i);
    nonzero_constraints.clear();
    zero_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, nonzero_constraints);
    DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints);
    if (prms.boundary_type == "periodic_left_right") {
        const MappingQ1<dim> mapping;
        dofs_to_supports.resize(dof_handler.n_dofs());
        DoFTools::map_dofs_to_support_points<dim, dim>(
                mapping, dof_handler, dofs_to_supports);
        collect_periodic_faces(dof_handler, 1, 2, 0, face_pairs);
        make_periodicity_constraints<dim, dim, double>(
                face_pairs,
                nonzero_constraints,
                dofs_to_supports,
                boundary_function[0]);
        make_periodicity_constraints<dim, dim, double>(
                face_pairs, zero_constraints, dofs_to_supports, zero_function);
    }
    else if (prms.boundary_type == "periodic_bottom_left_top_right") {
        const MappingQ1<dim> mapping;
        dofs_to_supports.resize(dof_handler.n_dofs());
        DoFTools::map_dofs_to_support_points<dim, dim>(
                mapping, dof_handler, dofs_to_supports);
        setup_side_to_side_periodic_constraints();
    }
    else if (prms.boundary_type == "dirichlet_left_right") {
        VectorTools::interpolate_boundary_values(
                dof_handler, 1, boundary_function[0], nonzero_constraints);
        VectorTools::interpolate_boundary_values(
                dof_handler, 2, boundary_function[1], nonzero_constraints);
        VectorTools::interpolate_boundary_values(
                dof_handler,
                1,
                Functions::ZeroFunction<dim>(3),
                zero_constraints);
        VectorTools::interpolate_boundary_values(
                dof_handler,
                2,
                Functions::ZeroFunction<dim>(3),
                zero_constraints);
    }
    else if (prms.boundary_type == "dirichlet_left_periodic_top_right") {

        // Setup Dirichlet conditions on the left face
        VectorTools::interpolate_boundary_values(
                dof_handler, 1, boundary_function[0], nonzero_constraints);
        VectorTools::interpolate_boundary_values(
                dof_handler,
                1,
                Functions::ZeroFunction<dim>(3),
                zero_constraints);

        // Setup up pair constraints on the right face
        const MappingQ1<dim> mapping;
        dofs_to_supports.resize(dof_handler.n_dofs());
        DoFTools::map_dofs_to_support_points<dim, dim>(
                mapping, dof_handler, dofs_to_supports);
        setup_side_to_side_periodic_constraints();
    }
    nonzero_constraints.close();
    zero_constraints.close();

    cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
         << std::endl;
}

struct VectorDoFPairs {
    types::global_dof_index left_X;
    types::global_dof_index left_Y;
    types::global_dof_index left_Z;
    types::global_dof_index right_X;
    types::global_dof_index right_Y;
    types::global_dof_index right_Z;

    bool operator<(const VectorDoFPairs& a) const {
        bool less_than {false};
        if (this->left_X < a.left_X) {
            less_than = true;
        }
        else if (this->left_X == a.left_X) {
            if (this->left_Y < a.left_Y) {
                less_than = true;
            }
            else if (this->left_Z == a.left_Z) {
                if (this->left_Z < a.left_Z) {
                    less_than = true;
                }
            }
        }

        return less_than;
    }
};

void insert_vector_dof_pairs(
        const std::vector<types::global_dof_index> local_left_dofs,
        const std::vector<types::global_dof_index> local_right_dofs,
        std::set<VectorDoFPairs>& dof_pairs) {
    const size_t num_vector_dof_pairs {local_left_dofs.size()};
    for (size_t i {0}; i != num_vector_dof_pairs; i += 3) {
        VectorDoFPairs vector_dof_pairs {
                local_left_dofs[0],
                local_left_dofs[1],
                local_left_dofs[2],
                local_right_dofs[0],
                local_right_dofs[0],
                local_right_dofs[0]};
        dof_pairs.insert(vector_dof_pairs);
    }
}

std::vector<double> average_face_offset(
        const std::vector<types::global_dof_index> local_left_dofs,
        const std::vector<types::global_dof_index> local_right_dofs,
        const Vector<double> present_solution) {

    const size_t num_vector_dof_pairs {local_left_dofs.size()};
    std::vector<double> offset(3);
    for (size_t i {0}; i != num_vector_dof_pairs; i += 3) {
        double left_X {present_solution(local_left_dofs[i])};
        double right_X {present_solution(local_right_dofs[i])};
        double X_offset {right_X - left_X};
        offset[0] += X_offset;
        double left_Y {present_solution(local_left_dofs[i + 1])};
        double right_Y {present_solution(local_right_dofs[i + 1])};
        double Y_offset {right_Y - left_Y};
        offset[1] += Y_offset;
        double left_Z {present_solution(local_left_dofs[i + 2])};
        double right_Z {present_solution(local_right_dofs[i + 2])};
        double Z_offset {right_Z - left_Z};
        offset[2] += Z_offset;
        // cout << left_X << " " << right_X << " " << left_Y << " " << right_Y
        // << " " << " " << left_Z << " " << right_Z << " " << std::endl;
    }
    for (unsigned int i {0}; i != 3; i++) {
        offset[i] /= (num_vector_dof_pairs / 3.0);
    }
    // cout << offset[0] << " " << offset[1] << " " << offset[2] << std::endl <<
    // std::endl;

    return offset;
}

template <int dim>
template <typename FaceIterator>
void SolveRing<dim>::set_overlap_constraints(
        FaceIterator& left_face,
        FaceIterator& right_face) {

    const unsigned int dofs_per_face = fe.n_dofs_per_face();
    std::vector<types::global_dof_index> left_dofs(dofs_per_face);
    std::vector<types::global_dof_index> right_dofs(dofs_per_face);
    left_face->get_dof_indices(left_dofs);
    right_face->get_dof_indices(right_dofs);
    for (unsigned int i {0}; i != dofs_per_face; ++i) {
        auto component_index = fe.face_system_to_component_index(i).first;
        auto left_dof = left_dofs[i];
        auto right_dof = right_dofs[i];
        // cout << left_dof << " " << right_dof << std::endl;
        //cout << i;
        if (nonzero_constraints.is_constrained(right_dof)) {
        //    cout << " skip" << std::endl;
            continue;
        }
        //cout << std::endl;
        nonzero_constraints.add_line(right_dof);
        nonzero_constraints.add_entry(right_dof, left_dof, 1.0);

        double left_solution {initial_stage_solution[left_dof]};
        double right_solution {initial_stage_solution[right_dof]};
        double spatial_offset {right_solution - left_solution};
        Vector<double> spatial_offset_vector(dim);
        spatial_offset_vector[component_index] = spatial_offset;

        boundary_function[0].f1 =
                std::make_shared<Functions::ConstantFunction<dim, double>>(
                        spatial_offset_vector);

        Point<dim> left_dof_support {dofs_to_supports[left_dof]};
        Point<dim> right_dof_support {dofs_to_supports[right_dof]};
        // cout << left_dof_support[0] << " " << left_dof_support[1] << " "
        //     << left_dof_support[2] << std::endl;
        // cout << right_dof_support[0] << " " << right_dof_support[1] << " "
        //     << right_dof_support[2] << std::endl;
        // cout << spatial_offset << std::endl;
        Point<dim> diff {right_dof_support - left_dof_support};
        Point<dim> ref {abs(diff[0]), left_dof_support[1], left_dof_support[2]};

        Vector<double> offset(3);
        boundary_function[0].vector_value(ref, offset);
        //cout << offset[component_index] << std::endl << std::endl;
        nonzero_constraints.set_inhomogeneity(
                right_dof, offset[component_index]);
    }
}

template <int dim>
void SolveRing<dim>::setup_side_to_side_periodic_constraints() {
    double material_offset = prms.beam_X - boundary.first - boundary.second;
    std::set<VectorDoFPairs> dof_pairs {};
    for (auto& left_cell: dof_handler.active_cell_iterators()) {
        for (auto& left_face: left_cell->face_iterators()) {
            double left_X {left_face->center()(0)};
            double left_Y {left_face->center()(1)};
            double left_Z {left_face->center()(2)};
            if (not(boundary.first <= left_X and left_X <= boundary.second and
                    left_Y < 1e-12)) {
                continue;
            }
            for (auto& right_cell: dof_handler.active_cell_iterators()) {
                for (auto& right_face: right_cell->face_iterators()) {
                    double right_X {right_face->center()(0)};
                    double right_Y {right_face->center()(1)};
                    double right_Z {right_face->center()(2)};
                    if (not(prms.beam_X - boundary.second <= right_X and
                            right_X <= prms.beam_X - boundary.first and
                            std::fabs(right_Y - prms.beam_Y) < 1e-12)) {
                        continue;
                    }
                    if (std::fabs(right_X - left_X - material_offset) <
                                1e-12 and
                        std::fabs(right_Z - left_Z) < 1e-12) {

                        set_overlap_constraints(left_face, right_face);
                        make_periodicity_constraints(
                                left_face,
                                right_face,
                                zero_constraints,
                                dofs_to_supports,
                                zero_function);
                    }
                }
            }
        }
    }
}

template <int dim>
void SolveRing<dim>::update_constraints() {
    update_boundary_function_gamma();
    nonzero_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, nonzero_constraints);
    if (prms.boundary_type == "periodic_left_right") {
        make_periodicity_constraints<dim, dim, double>(
                face_pairs,
                nonzero_constraints,
                dofs_to_supports,
                boundary_function[0]);
    }
    else if (prms.boundary_type == "periodic_bottom_left_top_right") {
        setup_side_to_side_periodic_constraints();
    }
    else if (prms.boundary_type == "dirichlet_left_right") {
        VectorTools::interpolate_boundary_values(
                dof_handler, 1, boundary_function[0], nonzero_constraints);
        VectorTools::interpolate_boundary_values(
                dof_handler, 2, boundary_function[1], nonzero_constraints);
    }
    else if (prms.boundary_type == "dirichlet_left_periodic_top_right") {
        VectorTools::interpolate_boundary_values(
                dof_handler, 1, boundary_function[0], nonzero_constraints);
        setup_side_to_side_periodic_constraints();
    }
    nonzero_constraints.close();
}

template <int dim>
void SolveRing<dim>::update_boundary_function_gamma() {
    for (auto& bf: boundary_function) {
        bf.gamma = gamma;
    }
}

template <int dim>
void SolveRing<dim>::update_boundary_function_stage(unsigned int stage_i) {
    for (unsigned int i {0}; i != prms.num_boundary_conditions; i++) {
        boundary_function[i].f1 = prms.boundary_functions[stage_i - 1][i];
        boundary_function[i].f2 = prms.boundary_functions[stage_i][i];
    }
    boundary.first = prms.left_boundaries[stage_i];
    boundary.second = prms.right_boundaries[stage_i];
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
                    0.5 * (grad_u + grad_u_T + grad_u_T * grad_u)};
            ADNumberType t1 = lambda / 2 *
                              std::pow(trace(green_lagrange_strain_tensor), 2);
            ADNumberType t2 = mu * double_contract<0, 0, 1, 1>(
                                           green_lagrange_strain_tensor,
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
            if (line_search_n == 0) {
                output_results(checkpoint + "-initial");
            }
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

            if (prms.centering == "vertex") {
                center_solution_on_vertex();
            }
            else if (prms.centering == "mean") {
                center_solution_on_mean();
            }
        }
    }
    output_results(checkpoint);
    initial_stage_solution = present_solution;
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
void SolveRing<dim>::center_solution_on_vertex() {
    Point<dim> corner {0, 0, 0};
    Vector<double> corner_solution(3);
    for (unsigned int i {0}; i != dof_handler.n_dofs(); i++) {
        if (std::fabs(dofs_to_supports[i][0] - corner[0]) < 1e-12 and
            std::fabs(dofs_to_supports[i][1] - corner[1]) < 1e-12 and
            std::fabs(dofs_to_supports[i][2] - corner[2]) < 1e-12) {
            for (unsigned int j {0}; j != 3; j++) {
                corner_solution[j] = present_solution[i + j];
            }
            break;
        }
    }
    for (unsigned int i {0}; i != dof_handler.n_dofs(); i++) {
        const unsigned int component_i {i % dim};
        present_solution[i] -= corner_solution[component_i];
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

    std::vector<Tensor<1, dim, double>> first_pseudo_material_force(2);
    std::vector<Tensor<1, dim, double>> second_pseudo_material_force(2);
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
                const Tensor<1, dim, double> material_normal {
                        normal_vectors[q_i]};

                const Tensor<2, dim, double> grad_u_T {transpose(grad_u)};
                const Tensor<2, dim, double> green_lagrange_strain {
                        0.5 * (grad_u + grad_u_T + grad_u_T * grad_u)};

                const Tensor<2, dim, double> deformation_grad {
                        grad_u + unit_symmetric_tensor<dim>()};
                const double deformation_grad_det {
                        determinant(deformation_grad)};

                const Tensor<2, dim, double> second_piola_kirchhoff {
                        lambda * trace(green_lagrange_strain) *
                                unit_symmetric_tensor<dim>() +
                        2 * mu * green_lagrange_strain};
                const Tensor<2, dim, double> first_piola_kirchhoff {
                        deformation_grad * second_piola_kirchhoff};

                ave_material_normal[boundary_id - 1] += material_normal;
                second_pseudo_material_force[boundary_id - 1] +=
                        second_piola_kirchhoff * material_normal *
                        fe_face_values.JxW(q_i);
                first_pseudo_material_force[boundary_id - 1] +=
                        first_piola_kirchhoff * material_normal *
                        fe_face_values.JxW(q_i);

                const Tensor<2, dim, double> cauchy {
                        deformation_grad * second_piola_kirchhoff *
                        transpose(deformation_grad) / deformation_grad_det};

                Tensor<1, dim, double> spatial_normal {
                        transpose(invert(deformation_grad)) * material_normal};
                spatial_normal /= spatial_normal.norm();
                double da_dA {
                        deformation_grad_det * spatial_normal *
                        transpose(invert(deformation_grad)) * material_normal};
                spatial_force[boundary_id - 1] += cauchy * spatial_normal *
                                                  fe_face_values.JxW(q_i) *
                                                  da_dA;
                ave_spatial_normal[boundary_id - 1] += spatial_normal;
            }
        }
    }
    ave_material_normal[0] /= ave_material_normal[0].norm();
    ave_spatial_normal[0] /= ave_spatial_normal[0].norm();
    ave_material_normal[1] /= ave_material_normal[1].norm();
    ave_spatial_normal[1] /= ave_spatial_normal[1].norm();

    const Tensor<1, dim, double> left_second_pseudo_material_normal_force = {
            (second_pseudo_material_force[0] * ave_material_normal[0]) *
            ave_material_normal[0]};
    const Tensor<1, dim, double> left_second_pseudo_material_shear_force = {
            second_pseudo_material_force[0] -
            left_second_pseudo_material_normal_force};

    const Tensor<1, dim, double> left_first_pseudo_material_normal_force = {
            (first_pseudo_material_force[0] * ave_material_normal[0]) *
            ave_material_normal[0]};
    const Tensor<1, dim, double> left_first_pseudo_material_shear_force = {
            first_pseudo_material_force[0] -
            left_first_pseudo_material_normal_force};

    const Tensor<1, dim, double> left_spatial_normal_force = {
            (spatial_force[0] * ave_spatial_normal[0]) * ave_spatial_normal[0]};
    const Tensor<1, dim, double> left_spatial_shear_force = {
            spatial_force[0] - left_spatial_normal_force};

    const Tensor<1, dim, double> right_second_pseudo_material_normal_force = {
            (second_pseudo_material_force[1] * ave_material_normal[1]) *
            ave_material_normal[1]};
    const Tensor<1, dim, double> right_second_pseudo_material_shear_force = {
            second_pseudo_material_force[1] -
            right_second_pseudo_material_normal_force};

    const Tensor<1, dim, double> right_first_pseudo_material_normal_force = {
            (first_pseudo_material_force[1] * ave_material_normal[1]) *
            ave_material_normal[1]};
    const Tensor<1, dim, double> right_first_pseudo_material_shear_force = {
            first_pseudo_material_force[1] -
            right_first_pseudo_material_normal_force};

    const Tensor<1, dim, double> right_spatial_normal_force = {
            (spatial_force[1] * ave_spatial_normal[1]) * ave_spatial_normal[1]};
    const Tensor<1, dim, double> right_spatial_shear_force = {
            spatial_force[1] - right_spatial_normal_force};

    cout << "Left boundary second pseudo material force: "
         << second_pseudo_material_force[0].norm() << std::endl;
    cout << "Right boundary second pseudo material force: "
         << second_pseudo_material_force[1].norm() << std::endl;
    cout << "Left boundary second pseudo material normal force: "
         << left_second_pseudo_material_normal_force.norm() << std::endl;
    cout << "Right boundary second pseudo material normal force: "
         << right_second_pseudo_material_normal_force.norm() << std::endl;
    cout << "Left boundary second pseudo material shear force: "
         << left_second_pseudo_material_shear_force.norm() << std::endl;
    cout << "Right boundary second pseudo material shear force: "
         << right_second_pseudo_material_shear_force.norm() << std::endl;
    cout << std::endl;

    cout << "Left boundary first pseudo material force: "
         << first_pseudo_material_force[0].norm() << std::endl;
    cout << "Right boundary first pseudo material force: "
         << first_pseudo_material_force[1].norm() << std::endl;
    cout << "Left boundary first pseudo material normal force: "
         << left_first_pseudo_material_normal_force.norm() << std::endl;
    cout << "Right boundary first pseudo material normal force: "
         << right_first_pseudo_material_normal_force.norm() << std::endl;
    cout << "Left boundary first pseudo material shear force: "
         << left_first_pseudo_material_shear_force.norm() << std::endl;
    cout << "Right boundary first pseudo material shear force: "
         << right_first_pseudo_material_shear_force.norm() << std::endl;
    cout << std::endl;

    cout << "Left boundary spatial force: " << spatial_force[0].norm()
         << std::endl;
    cout << "Right boundary spatial force: " << spatial_force[1].norm()
         << std::endl;
    cout << "Left boundary spatial normal force: "
         << left_spatial_normal_force.norm() << std::endl;
    cout << "Right boundary spatial normal force: "
         << right_spatial_normal_force.norm() << std::endl;
    cout << "Left boundary spatial shear force: "
         << left_spatial_shear_force.norm() << std::endl;
    cout << "Right boundary spatial shear force: "
         << right_spatial_shear_force.norm() << std::endl;
    cout << std::endl;
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
    initial_stage_solution = present_solution;

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
} // namespace solve_ring

int main() {
    using namespace dealii;
    using namespace parameters;
    using namespace postprocessing;
    using namespace solve_ring;
    deallog.depth_console(0);
    Params<3> prms {};
    SolveRing<3> ring_solver {prms};
    ring_solver.run();

    return 0;
}
