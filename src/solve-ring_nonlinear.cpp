#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
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
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include "modded_periodic_functions.h"

using namespace dealii;

template <int dim>
class SolveRing {
  public:
    SolveRing(double length, double width, double E, double nu);
    void run();

  private:
    void make_grid();
    void initiate_system();
    void setup_constraints();
    void update_constraints();
    void setup_sparsity_pattern();
    void assemble(const bool initial_step, const bool assemble_matrix);
    void assemble_system(const bool initial_step);
    void assemble_rhs(const bool initial_step);
    void solve(const bool initial_step);
    double calc_residual_norm();
    void newton_iteration(
            const double tolerance,
            const unsigned int max_n_line_searches,
            bool first_step,
            const std::string checkpoint);
    void output_grid() const;
    void output_checkpoint(const std::string checkpoint) const;
    void output_results(const std::string checkpoint) const;
    void load_checkpoint(const std::string checkpoint);

    Triangulation<dim> triangulation;
    FESystem<dim> fe;
    DoFHandler<dim> dof_handler;
    AffineConstraints<double> zero_constraints;
    AffineConstraints<double> nonzero_constraints;

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> present_solution;
    Vector<double> newton_update;
    Vector<double> system_rhs;
    Vector<double> evaluation_point;

    std::vector<Point<dim>> dofs_to_supports;
    std::vector<GridTools::PeriodicFacePair<
            typename DoFHandler<dim, dim>::cell_iterator>>
            face_pairs;
    std::vector<double> gamma;

    double length;
    double width;
    double E;
    double nu;
    double lambda;
    double mu;
};

template <int dim>
class BoundaryValues: public Function<dim> {
  public:
    BoundaryValues(double gamma, unsigned int comps);
    void vector_value(const Point<dim>& p, Vector<double>& values)
            const override;

  private:
    double gamma;
};

template <int dim>
BoundaryValues<dim>::BoundaryValues(double gamma, unsigned int comps):
        gamma {gamma}, Function<dim> {comps} {}

template <int dim>
void BoundaryValues<dim>::vector_value(
        const Point<dim>& p,
        Vector<double>& values) const {

    values[0] = gamma * (p[0] * (2 / numbers::PI - 1) + p[1]);
    values[1] = gamma * (-2 * p[0] / numbers::PI - p[1]);
    values[2] = 0;
}

template <int dim>
SolveRing<dim>::SolveRing(double length, double width, double E, double nu):
        fe(FE_Q<dim>(1), dim),
        dof_handler(triangulation),
        length {length},
        width {width},
        E {E},
        nu {nu} {
    mu = E / (2 * (1 + nu));
    lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
}

template <int dim>
void SolveRing<dim>::make_grid() {
    const Point<dim>& origin {0, 0, 0};
    const Point<dim>& size {length, width, width};
    GridGenerator::hyper_rectangle(triangulation, origin, size);
    for (auto& face: triangulation.active_face_iterators()) {
        if (std::fabs(face->center()(1) - width / 2) < 1e-12) {
            if (std::fabs(face->center()(0)) < 1e-12) {
                face->set_boundary_id(1);
            }
            else if (std::fabs(face->center()(0) - length) < 1e-12) {
                face->set_boundary_id(2);
            }
        }
    }
    triangulation.refine_global(2);
}

template <int dim>
void SolveRing<dim>::initiate_system() {
    dof_handler.distribute_dofs(fe);
    present_solution.reinit(dof_handler.n_dofs());
    newton_update.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
}

template <int dim>
void SolveRing<dim>::setup_constraints() {
    const MappingQ1<dim> mapping;
    dofs_to_supports.resize(dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points<dim, dim>(
            mapping, dof_handler, dofs_to_supports);
    collect_periodic_faces(dof_handler, 1, 2, 0, face_pairs);

    zero_constraints.clear();
    make_periodicity_constraints<dim, dim, double>(
            face_pairs, zero_constraints, dofs_to_supports, {0.0, 0.0, 0.0});
    // VectorTools::interpolate_boundary_values(
    //        dof_handler, 1, Functions::ZeroFunction<dim>(3),
    //        zero_constraints);
    // VectorTools::interpolate_boundary_values(
    //        dof_handler, 2, Functions::ZeroFunction<dim>(3),
    //        zero_constraints);
    zero_constraints.close();

    cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
         << std::endl;
}

template <int dim>
void SolveRing<dim>::update_constraints() {
    nonzero_constraints.clear();
    make_periodicity_constraints<dim, dim, double>(
            face_pairs, nonzero_constraints, dofs_to_supports, gamma);
    /*VectorTools::interpolate_boundary_values(
            dof_handler,
            1,
            Functions::ZeroFunction<dim>(3),
            nonzero_constraints);
    VectorTools::interpolate_boundary_values(
            dof_handler,
            2,
            BoundaryValues<dim>(gamma[0], 3),
            nonzero_constraints);*/
    nonzero_constraints.close();
    // nonzero_constraints.distribute(present_solution);
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
    const AffineConstraints<double>& constraints_used =
            initial_step ? nonzero_constraints : zero_constraints;
    SolverControl solver_control(10000, 1e-12);
    SolverGMRES<Vector<double>> solver(solver_control);
    PreconditionJacobi<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.0);
    // SolverCG<Vector<double>> solver(solver_control);
    // PreconditionSSOR<SparseMatrix<double>> preconditioner;
    // preconditioner.initialize(system_matrix, 1.2);
    solver.solve(system_matrix, newton_update, system_rhs, preconditioner);
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
        const double tolerance,
        const unsigned int max_n_line_searches,
        bool first_step,
        const std::string checkpoint) {

    unsigned int line_search_n = 0;
    double last_res = 1.0;
    double current_res = 1.0;

    first_step = false;
    while ((first_step or (current_res > tolerance)) and
           line_search_n < max_n_line_searches) {
        if (first_step) {
            newton_update = present_solution;
            evaluation_point = present_solution;
            assemble_system(false);
            solve(first_step);
            present_solution = newton_update;
            nonzero_constraints.distribute(present_solution);
            first_step = false;
            evaluation_point = present_solution;
            assemble_rhs(first_step);
            current_res = calc_residual_norm();
            std::cout << "The residual of initial guess is " << current_res
                      << std::endl;
            last_res = current_res;
            output_results("test1");
        }
        else {
            evaluation_point = present_solution;
            assemble_system(first_step);
            solve(first_step);
            for (double alpha {1.0}; alpha > 1e-5; alpha *= 0.5) {
                evaluation_point = present_solution;
                evaluation_point.add(alpha, newton_update);
                nonzero_constraints.distribute(evaluation_point);
                assemble_rhs(first_step);
                current_res = calc_residual_norm();
                std::cout << "  alpha: " << std::setw(10) << alpha
                          << std::setw(0) << "  residual: " << current_res
                          << std::endl;
                // if ((last_res - current_res) >= (alpha * last_res / 2)) {
                if (current_res < last_res) {
                    break;
                }
            }
            present_solution = evaluation_point;
            output_results("test2");
            std::cout << "  number of line searches: " << line_search_n
                      << "  residual: " << current_res << std::endl;
            last_res = current_res;
            ++line_search_n;
        }
    }
    output_results(checkpoint);
}

template <int dim>
void SolveRing<dim>::output_grid() const {
    GridOut grid_out;
    std::ofstream grid_output {"outs/mesh.vtk"};
    grid_out.write_vtk(triangulation, grid_output);
}

template <int dim>
void SolveRing<dim>::output_checkpoint(const std::string checkpoint) const {
    std::ofstream solution_out {"outs/solution_" + checkpoint + ".txt"};
    boost::archive::text_oarchive solution_ar {solution_out};
    present_solution.save(solution_ar, 0);

    std::ofstream triangulation_out {
            "outs/triangulation_" + checkpoint + ".txt"};
    boost::archive::text_oarchive triangulation_ar {triangulation_out};
    triangulation.save(triangulation_ar, 0);

    std::ofstream dof_handler_out {"outs/dof_handler_" + checkpoint + ".txt"};
    boost::archive::text_oarchive dof_handler_ar {dof_handler_out};
    dof_handler.save(dof_handler_ar, 0);
}

template <int dim>
void SolveRing<dim>::output_results(const std::string checkpoint) const {
    DataOut<dim> data_out;
    std::vector<std::string> solution_names(dim, "displacement");
    data_out.attach_dof_handler(dof_handler);
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation(
                    dim,
                    DataComponentInterpretation::component_is_part_of_vector);
    data_out.add_data_vector(
            present_solution,
            solution_names,
            DataOut<dim>::type_dof_data,
            data_component_interpretation);
    data_out.build_patches();
    std::ofstream data_output("outs/solution_" + checkpoint + ".vtk");
    data_out.write_vtk(data_output);
}

template <int dim>
void SolveRing<dim>::load_checkpoint(const std::string checkpoint) {
    std::ifstream solution_inp {"outs/solution_" + checkpoint + ".txt"};
    boost::archive::text_iarchive solution_ar {solution_inp};
    present_solution.load(solution_ar, 0);

    std::ifstream triangulation_inp {
            "outs/triangulation_" + checkpoint + ".txt"};
    boost::archive::text_iarchive triangulation_ar {triangulation_inp};
    triangulation.load(triangulation_ar, 0);

    dof_handler.distribute_dofs(fe);
    std::ifstream dof_handler_inp {"outs/dof_handler_" + checkpoint + ".txt"};
    boost::archive::text_iarchive dof_handler_ar {dof_handler_inp};
    dof_handler.load(dof_handler_ar, 0);

    newton_update.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
}

template <int dim>
void SolveRing<dim>::run() {
    make_grid();
    output_grid();
    int num_gamma_iters {1};
    std::string checkpoint;
    //initiate_system();
    load_checkpoint("final");
    setup_constraints();
    bool first_step {true};
    for (auto i {0}; i != num_gamma_iters; i++) {
        double gamma_i = static_cast<double>((i + 1)) / num_gamma_iters;
        gamma_i = 1.1;
        std::ostringstream stream_obj;
        stream_obj << std::fixed;
        stream_obj << std::setprecision(2);
        stream_obj << gamma_i;
        checkpoint = stream_obj.str();
        cout << "Gamma " << stream_obj.str() << std::endl;
        gamma = {gamma_i, gamma_i, gamma_i};
        update_constraints();
        if (first_step) {
            setup_sparsity_pattern();
        }
        newton_iteration(1e-12, 10, first_step, checkpoint);
        output_checkpoint(checkpoint);
        first_step = false;
    }
    checkpoint = "final";
    //output_checkpoint(checkpoint);
    //output_results(checkpoint);
}

int main() {
    deallog.depth_console(0);
    SolveRing<3> ring_solver {20, 2, 2.2e7, 0.3};
    ring_solver.run();

    return 0;
}
