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
struct Params {
    unsigned int refinements;
    double length;
    double width;
    unsigned int num_boundary_stages;
    unsigned int starting_stage;
    unsigned int num_gamma_iters;

    // There is probably a better way to do this
    static const unsigned int max_stages {5};
    std::vector<std::shared_ptr<Functions::ParsedFunction<dim>>>
            boundary_functions;
    double E;
    double nu;
    unsigned int max_n_line_searches;
    double alpha_factor;
    double min_alpha;
    double alpha_check_factor;
    double nonlinear_tol;
    unsigned int max_linear_iters;
    double linear_tol;
    bool load_from_checkpoint;
    std::string input_prefix;
    std::string input_checkpoint;
    std::string output_prefix;

    Params();
    static void declare_parameters(ParameterHandler& prm);
    void parse_parameters(ParameterHandler& prm);
};

template <int dim>
Params<dim>::Params() {
    for (unsigned int i {0}; i != max_stages; i++) {
        boundary_functions.push_back(
                std::make_shared<Functions::ParsedFunction<dim>>(dim));
    }
    ParameterHandler prm;
    declare_parameters(prm);
    prm.parse_input(std::cin);
    parse_parameters(prm);
}

template <int dim>
void Params<dim>::declare_parameters(ParameterHandler& prm) {
    prm.enter_subsection("Mesh and geometry");
    {
        prm.declare_entry(
                "Number of refinements",
                "2",
                Patterns::Integer(0),
                "Number of global mesh refinement steps "
                "applied to initial coarse grid");
        prm.declare_entry(
                "length", "20", Patterns::Double(0), "Length of beam");
        prm.declare_entry("width", "2", Patterns::Double(0), "Width of beam");
    }
    prm.leave_subsection();

    prm.enter_subsection("Boundary conditions");
    {
        prm.declare_entry(
                "Number of boundary stages",
                "1",
                Patterns::Integer(1),
                "Number of boundary stages");
        prm.declare_entry(
                "Starting stage",
                "0",
                Patterns::Integer(0),
                "Boundary stage to start calculations on");
        prm.declare_entry(
                "Number of boundary increments",
                "1",
                Patterns::Integer(1),
                "Number of boundary increments");
    }
    prm.leave_subsection();

    for (unsigned int i {0}; i != max_stages + 1; i++) {
        prm.enter_subsection("Boundary function stage " + std::to_string(i));
        { Functions::ParsedFunction<dim>::declare_parameters(prm, dim); }
        prm.leave_subsection();
    }

    prm.enter_subsection("Physical constants");
    {
        prm.declare_entry(
                "Young's modulus",
                "2.2e7",
                Patterns::Double(0),
                "Young's modulus");
        prm.declare_entry(
                "Poisson's ratio",
                "0.3",
                Patterns::Double(0),
                "Poisson's ratio");
    }
    prm.leave_subsection();

    prm.enter_subsection("Solver");
    {
        prm.declare_entry(
                "Maximum number of line searches",
                "10",
                Patterns::Integer(0),
                "Maximum number of line searches");
        prm.declare_entry(
                "Minimum alpha",
                "1e-5",
                Patterns::Double(0),
                "Smallest alpha to try before moving on");
        prm.declare_entry(
                "Alpha factor",
                "0.5",
                Patterns::Double(0),
                "Factor to multiply alpha by when trying smaller step");
        prm.declare_entry(
                "Alpha check factor",
                "0.5",
                Patterns::Double(0),
                "Larger values require bigger residual decrease for acceptable "
                "alpha");
        prm.declare_entry(
                "Nonlinear tolerance",
                "1e-12",
                Patterns::Double(0),
                "Tolerance for nonlinear solver");
        prm.declare_entry(
                "Maximum number of linear solver iterations",
                "10000",
                Patterns::Integer(1),
                "Maximum number of linear solver iterations");
        prm.declare_entry(
                "Linear tolerance",
                "1e-12",
                Patterns::Double(0),
                "Tolerance for linear solver");
    }
    prm.leave_subsection();

    prm.enter_subsection("Input and output");
    {
        prm.declare_entry(
                "Load from checkpoint",
                "false",
                Patterns::Bool(),
                "Begin calculation from a checkpoint");
        prm.declare_entry(
                "Input filename prefix",
                "",
                Patterns::Anything(),
                "Prefix of the output filename");
        prm.declare_entry(
                "Input checkpoint",
                "",
                Patterns::Anything(),
                "Checkpoint of the input file");
        prm.declare_entry(
                "Output filename prefix",
                "",
                Patterns::Anything(),
                "Prefix of the output filename");
    }
    prm.leave_subsection();
}

template <int dim>
void Params<dim>::parse_parameters(ParameterHandler& prm) {
    prm.enter_subsection("Mesh and geometry");
    {
        refinements = prm.get_integer("Number of refinements");
        length = prm.get_double("length");
        width = prm.get_double("width");
    }
    prm.leave_subsection();

    prm.enter_subsection("Boundary conditions");
    {
        num_boundary_stages = prm.get_integer("Number of boundary stages");
        starting_stage = prm.get_integer("Starting stage");
        num_gamma_iters = prm.get_integer("Number of boundary increments");
    }
    prm.leave_subsection();

    for (unsigned int i {0}; i != num_boundary_stages + 1; i++) {
        prm.enter_subsection("Boundary function stage " + std::to_string(i));
        boundary_functions[i]->parse_parameters(prm);
        prm.leave_subsection();
    }

    prm.enter_subsection("Physical constants");
    {
        E = prm.get_double("Young's modulus");
        nu = prm.get_double("Poisson's ratio");
    }
    prm.leave_subsection();

    prm.enter_subsection("Solver");
    {
        max_n_line_searches =
                prm.get_integer("Maximum number of line searches");
        alpha_factor = prm.get_double("Alpha factor");
        alpha_check_factor = prm.get_double("Alpha check factor");
        min_alpha = prm.get_double("Minimum alpha");
        nonlinear_tol = prm.get_double("Nonlinear tolerance");
        max_linear_iters =
                prm.get_integer("Maximum number of linear solver iterations");
        linear_tol = prm.get_double("Linear tolerance");
    }
    prm.leave_subsection();

    prm.enter_subsection("Input and output");
    {
        load_from_checkpoint = prm.get_bool("Load from checkpoint");
        input_prefix = prm.get("Input filename prefix");
        input_checkpoint = prm.get("Input checkpoint");
        output_prefix = prm.get("Output filename prefix");
    }
    prm.leave_subsection();
}

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
class SolveRing {
  public:
    SolveRing(Params<dim>& prms);
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
    std::string format_gamma(const double gamma);
    void update_boundary_function_gamma();
    void update_boundary_function_stage(unsigned int stage_i);

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
void SolveRing<dim>::make_grid() {
    const Point<dim>& origin {0, 0, 0};
    const Point<dim>& size {prms.length, prms.width, prms.width};
    GridGenerator::hyper_rectangle(triangulation, origin, size);
    for (auto& face: triangulation.active_face_iterators()) {
        if (std::fabs(face->center()(1) - prms.width / 2) < 1e-12) {
            if (std::fabs(face->center()(0)) < 1e-12) {
                face->set_boundary_id(1);
            }
            else if (std::fabs(face->center()(0) - prms.length) < 1e-12) {
                face->set_boundary_id(2);
            }
        }
    }
    triangulation.refine_global(prms.refinements);
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

    update_boundary_function_stage(prms.starting_stage);
    nonzero_constraints.clear();
    make_periodicity_constraints<dim, dim, double>(
            face_pairs,
            nonzero_constraints,
            dofs_to_supports,
            boundary_function);
    nonzero_constraints.close();

    zero_constraints.clear();
    make_periodicity_constraints<dim, dim, double>(
            face_pairs, zero_constraints, dofs_to_supports, zero_function);
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
    update_boundary_function_gamma();
    nonzero_constraints.clear();
    make_periodicity_constraints<dim, dim, double>(
            face_pairs,
            nonzero_constraints,
            dofs_to_supports,
            boundary_function);
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
    SolverControl solver_control(prms.max_linear_iters, prms.linear_tol);
    SolverGMRES<Vector<double>> solver(solver_control);
    PreconditionJacobi<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);
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

    bool boundary_updated {!first_step};
    unsigned int line_search_n = 0;
    double last_res = 1.0;
    double current_res = 1.0;

    while ((first_step or (current_res > tolerance)) and
           line_search_n < max_n_line_searches) {
        if (first_step) {
            evaluation_point = present_solution;
            assemble_system(first_step);
            newton_update = present_solution;
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
        }
        else {
            evaluation_point = present_solution;
            assemble_system(first_step);
            solve(first_step);
            for (double alpha {1.0}; alpha > prms.min_alpha;
                 alpha *= prms.alpha_factor) {
                evaluation_point = present_solution;
                evaluation_point.add(alpha, newton_update);
                nonzero_constraints.distribute(evaluation_point);
                assemble_rhs(first_step);
                current_res = calc_residual_norm();
                std::cout << "  alpha: " << std::setw(10) << alpha
                          << std::setw(0) << "  residual: " << current_res
                          << std::endl;
                if (boundary_updated or
                    ((last_res - current_res) >=
                     (alpha * last_res * prms.alpha_check_factor))) {
                    // if (boundary_updated or (current_res < last_res)) {
                    boundary_updated = false;
                    break;
                }
            }
            present_solution = evaluation_point;
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
    std::ofstream data_output(
            prms.output_prefix + "_displacement_" + checkpoint + ".vtk");
    data_out.write_vtk(data_output);
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
std::string SolveRing<dim>::format_gamma(const double gamma) {

    // Subtract a half to keep integer values of log10 in precision below
    int precision {
            static_cast<int>(std::log10(prms.num_gamma_iters - 0.5)) + 1};
    std::ostringstream stream_obj;
    stream_obj << std::fixed;
    stream_obj << std::setprecision(precision);
    stream_obj << gamma;
    return stream_obj.str();
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
    setup_constraints();
    setup_sparsity_pattern();
    for (unsigned int stage_i {prms.starting_stage};
         stage_i != prms.num_boundary_stages;
         stage_i++) {
        update_boundary_function_stage(stage_i);
        for (unsigned int i {0}; i != prms.num_gamma_iters; i++) {
            gamma = static_cast<double>((i + 1)) / prms.num_gamma_iters;
            std::string gamma_formatted {format_gamma(gamma)};
            checkpoint = std::to_string(stage_i) + "-" + gamma_formatted;
            cout << "Stage " << std::to_string(stage_i) << ", gamma "
                 << gamma_formatted << std::endl;
            update_constraints();
            newton_iteration(
                    prms.nonlinear_tol,
                    prms.max_n_line_searches,
                    first_step,
                    checkpoint);
            output_checkpoint(checkpoint);
            first_step = false;
        }
        checkpoint = "final";
        output_checkpoint(checkpoint);
        output_results(checkpoint);
    }
}

int main() {
    deallog.depth_console(0);
    Params<3> prms {};
    SolveRing<3> ring_solver {prms};
    ring_solver.run();

    return 0;
}
