#ifndef parameters_h
#define parameteres_h

#include <vector>

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/utilities.h>

using namespace dealii;

template <int dim>
struct Params {
    unsigned int global_refinements;
    unsigned int adaptive_refinements;
    double length;
    double width;
    unsigned int x_subdivisions;
    unsigned int y_subdivisions;
    unsigned int z_subdivisions;
    unsigned int num_boundary_stages;
    unsigned int starting_stage;
    std::vector<unsigned int> num_gamma_iters;

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
    std::string linear_solver;
    unsigned int max_linear_iters;
    double linear_tol;
    double precon_relaxation_param;
    bool load_from_checkpoint;
    std::string input_prefix;
    std::string input_checkpoint;
    std::string output_prefix;
    unsigned int gamma_precision;

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
                "Number of initial refinements",
                "2",
                Patterns::Integer(0),
                "Number of initial mesh refinements");
        prm.declare_entry(
                "Number of final refinements",
                "0",
                Patterns::Integer(0),
                "Number of final mesh refinements");
        prm.declare_entry(
                "Beam length", "20", Patterns::Double(0), "Length of beam");
        prm.declare_entry(
                "Beam width", "2", Patterns::Double(0), "Width of beam");
        prm.declare_entry(
                "x subdivisions",
                "1",
                Patterns::Integer(1),
                "Number of subdivisions along x");
        prm.declare_entry(
                "y subdivisions",
                "1",
                Patterns::Integer(1),
                "Number of subdivisions along y");
        prm.declare_entry(
                "z subdivisions",
                "1",
                Patterns::Integer(1),
                "Number of subdivisions along z");
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
    }
    prm.leave_subsection();

    for (unsigned int i {0}; i != max_stages + 1; i++) {
        prm.enter_subsection("Boundary function stage " + std::to_string(i));
        {
            prm.declare_entry(
                    "Number of boundary increments",
                    "1",
                    Patterns::Integer(1),
                    "Number of boundary increments");
            Functions::ParsedFunction<dim>::declare_parameters(prm, dim);
        }
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
                "Linear solver",
                "direct",
                Patterns::Anything(),
                "Type of linear solver (direct or iterative)");
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
        prm.declare_entry(
                "Preconditioner relaxation parameter",
                "1.0",
                Patterns::Double(0),
                "Preconditioner relaxation parameter");
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
        prm.declare_entry(
                "Gamma precision",
                "1",
                Patterns::Integer(0),
                "Number of digits to include in gamma string");
    }
    prm.leave_subsection();
}

template <int dim>
void Params<dim>::parse_parameters(ParameterHandler& prm) {
    prm.enter_subsection("Mesh and geometry");
    {
        global_refinements = prm.get_integer("Number of initial refinements");
        adaptive_refinements = prm.get_integer("Number of final refinements");
        length = prm.get_double("Beam length");
        width = prm.get_double("Beam width");
        x_subdivisions = prm.get_integer("x subdivisions");
        y_subdivisions = prm.get_integer("y subdivisions");
        z_subdivisions = prm.get_integer("z subdivisions");
    }
    prm.leave_subsection();

    prm.enter_subsection("Boundary conditions");
    {
        num_boundary_stages = prm.get_integer("Number of boundary stages");
        starting_stage = prm.get_integer("Starting stage");
    }
    prm.leave_subsection();

    for (unsigned int i {0}; i != num_boundary_stages + 1; i++) {
        prm.enter_subsection("Boundary function stage " + std::to_string(i));
        {
            num_gamma_iters.push_back(
                    prm.get_integer("Number of boundary increments"));
            boundary_functions[i]->parse_parameters(prm);
        }
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
        linear_solver = prm.get("Linear solver");
        max_linear_iters =
                prm.get_integer("Maximum number of linear solver iterations");
        linear_tol = prm.get_double("Linear tolerance");
        precon_relaxation_param =
                prm.get_double("Preconditioner relaxation parameter");
    }
    prm.leave_subsection();

    prm.enter_subsection("Input and output");
    {
        load_from_checkpoint = prm.get_bool("Load from checkpoint");
        input_prefix = prm.get("Input filename prefix");
        input_checkpoint = prm.get("Input checkpoint");
        output_prefix = prm.get("Output filename prefix");
        gamma_precision = prm.get_integer("Gamma precision");
    }
    prm.leave_subsection();
}

#endif
