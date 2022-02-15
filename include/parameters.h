#ifndef parameters_h
#define parameters_h

#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/utilities.h>

namespace parameters {

using namespace dealii;

template <int dim>
struct Params {

    // Mesh and geometry
    unsigned int initial_refinements;
    unsigned int final_refinements;
    unsigned int starting_refinement;
    std::string refine_direction;
    std::string mesh_type;
    double beam_X;
    double beam_Y;
    double beam_Z;
    double cylinder_radius;
    double cylinder_length;
    unsigned int x_subdivisions;
    unsigned int y_subdivisions;
    unsigned int z_subdivisions;
    unsigned int fe_degree;
    std::string centering;
    bool set_ring_config;

    // Boundary conditions
    unsigned int num_boundary_domains;
    unsigned int num_boundary_stages;
    unsigned int num_boundary_conditions;
    unsigned int starting_stage;
    static const unsigned int max_stages {10};
    static const unsigned int max_conditions {10};
    static const unsigned int max_domains {10};

    // Bounday domain definition
    std::vector<double> min_X {};
    std::vector<double> max_X {};
    std::vector<double> min_Y {};
    std::vector<double> max_Y {};
    std::vector<double> min_Z {};
    std::vector<double> max_Z {};

    // Bounday condition definitions
    std::vector<std::string> boundary_type {};
    std::unordered_map<unsigned int, unsigned int> associated_domain {};
    std::unordered_map<unsigned int, unsigned int> constrained_domain {};
    std::unordered_map<unsigned int, unsigned int> anchor_domain {};

    // Bounday function definitions
    std::vector<unsigned int> num_gamma_iters {};
    std::vector<std::vector<bool>> use_current_config {};
    std::vector<std::vector<std::shared_ptr<Functions::ParsedFunction<dim>>>>
            boundary_functions;

    // Physical constants
    double E;
    double nu;

    // Solver
    unsigned int max_n_line_searches;
    double alpha_factor;
    double min_alpha;
    double alpha_check_factor;
    double nonlinear_tol;
    std::string linear_solver;
    unsigned int max_linear_iters;
    double linear_tol;
    double precon_relaxation_param;

    // Input and output
    bool load_from_checkpoint;
    std::string input_prefix;
    std::string input_checkpoint;
    std::string output_prefix;
    unsigned int gamma_precision;

    Params();
    Params(const std::string config_filename);
    static void declare_parameters(ParameterHandler& prm);
    void parse_parameters(ParameterHandler& prm);
    ParameterHandler prm;
};

template <int dim>
Params<dim>::Params() {
    declare_parameters(prm);
}

template <int dim>
Params<dim>::Params(const std::string config_filename) {
    for (unsigned int i {0}; i != max_stages; i++) {
        boundary_functions.push_back({});
        for (unsigned int j {0}; j != max_conditions; j++) {
            boundary_functions[i].push_back(
                    std::make_shared<Functions::ParsedFunction<dim>>(dim));
        }
    }
    declare_parameters(prm);
    std::ifstream config_file {config_filename};
    prm.parse_input(config_file);
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
                "Number of mesh before solving");
        prm.declare_entry(
                "Number of final refinements",
                "0",
                Patterns::Integer(0),
                "Number of refinements after solving");
        prm.declare_entry(
                "Starting refinement level",
                "0",
                Patterns::Integer(0),
                "For use if continuing from a previous calculations where some "
                "refinements were made");
        prm.declare_entry(
                "Refinement direction",
                "",
                Patterns::Anything(),
                "Axis or axes to refine along (x, yz, or nothing for "
                "isotropic)");
        prm.declare_entry(
                "Mesh type",
                "beam",
                Patterns::Anything(),
                "Currently only beam has been tested (but cylinder can be "
                "specified)");
        prm.declare_entry(
                "Beam X", "100", Patterns::Double(0), "Length of beam / m");
        prm.declare_entry(
                "Beam Y", "1", Patterns::Double(0), "Height of beam / m");
        prm.declare_entry(
                "Beam Z", "1", Patterns::Double(0), "Width of beam / m");
        prm.declare_entry(
                "Cylinder radius",
                "100",
                Patterns::Double(0),
                "Radius of cylinder / m");
        prm.declare_entry(
                "Cylinder length",
                "100",
                Patterns::Double(0),
                "Length of cylinder / m");
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
        prm.declare_entry(
                "FE degree", "1", Patterns::Integer(1), "FE degree (1 or 2)");
        prm.declare_entry(
                "Centering",
                "None",
                Patterns::Anything(),
                "Method to center the solution, required when using only pair "
                "constraints");
        prm.declare_entry(
                "Set ring configuration",
                "false",
                Patterns::Bool(),
                "Set initial configuration to be a ring");
    }
    prm.leave_subsection();

    prm.enter_subsection("Boundary conditions");
    {
        prm.declare_entry(
                "Number of boundary domains",
                "1",
                Patterns::Integer(1),
                "Number of boundary domains");
        prm.declare_entry(
                "Number of boundary conditions",
                "1",
                Patterns::Integer(1),
                "Number of boundary conditions");
        prm.declare_entry(
                "Number of boundary stages",
                "1",
                Patterns::Integer(1),
                "Number of boundary stages");
        prm.declare_entry(
                "Starting stage",
                "1",
                Patterns::Integer(0),
                "Boundary stage to start calculations on");
    }
    prm.leave_subsection();

    for (unsigned int i {0}; i != max_domains + 1; i++) {
        prm.enter_subsection("Boundary domain " + std::to_string(i));
        {
            prm.declare_entry(
                    "X min",
                    "0",
                    Patterns::Double(),
                    "Minimum X in domain / m");
            prm.declare_entry(
                    "X max",
                    "0",
                    Patterns::Double(),
                    "Maximum X in domain / m");
            prm.declare_entry(
                    "Y min",
                    "0",
                    Patterns::Double(),
                    "Minimum Y in domain / m");
            prm.declare_entry(
                    "Y max",
                    "0",
                    Patterns::Double(),
                    "Maximum Y in domain / m");
            prm.declare_entry(
                    "Z min",
                    "0",
                    Patterns::Double(),
                    "Minimum Z in domain / m");
            prm.declare_entry(
                    "Z max",
                    "0",
                    Patterns::Double(),
                    "Maximum Z in domain / m");
        }
        prm.leave_subsection();
    }

    for (unsigned int i {0}; i != max_conditions + 1; i++) {
        prm.enter_subsection("Boundary condition " + std::to_string(i));
        {
            prm.declare_entry(
                    "Boundary type",
                    "dirichlet",
                    Patterns::Anything(),
                    "Type of boundary condition (dirichlet or pair "
                    "constraint)");
            prm.declare_entry(
                    "Associated domain",
                    "0",
                    Patterns::Integer(0),
                    "For use with dirichlet boundary condition");
            prm.declare_entry(
                    "Constrained domain",
                    "0",
                    Patterns::Integer(0),
                    "The domain that pair constraints will apply to");
            prm.declare_entry(
                    "Anchor domain",
                    "0",
                    Patterns::Integer(0),
                    "The domain that the pair constraints use as a reference");
        }
        prm.leave_subsection();
    }

    for (unsigned int i {0}; i != max_stages + 1; i++) {
        prm.enter_subsection("Boundary function stage " + std::to_string(i));
        {
            prm.declare_entry(
                    "Number of boundary increments",
                    "1",
                    Patterns::Integer(1),
                    "The number of steps to increment the boundary");
            for (unsigned int j {0}; j != max_conditions + 1; j++) {
                prm.enter_subsection("Boundary condition " + std::to_string(j));
                {
                    prm.declare_entry(
                            "Use current configuration",
                            "false",
                            Patterns::Bool(),
                            "Set the initial boundary condition to the current "
                            "configuration instead of using previous stage");
                    Functions::ParsedFunction<dim>::declare_parameters(
                            prm, dim);
                }
                prm.leave_subsection();
            }
        }
        prm.leave_subsection();
    }

    prm.enter_subsection("Physical constants");
    {
        prm.declare_entry(
                "Young's modulus",
                "2.2e7",
                Patterns::Double(0),
                "Young's modulus / N m^-2");
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
                "Maximum number of line searches for Newton's method");
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
                "Larger values require bigger residual decrease for "
                "acceptable alpha");
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
                "Prefix of the input filename");
        prm.declare_entry(
                "Input checkpoint",
                "",
                Patterns::Anything(),
                "Checkpoint of the input file, which is the part of the "
                "checkpoint files written during the calculation that has "
                "format [stage index]-[refinement index]-[gamma]");
        prm.declare_entry(
                "Output filename prefix",
                "",
                Patterns::Anything(),
                "Prefix of the output filename");
        prm.declare_entry(
                "Gamma precision",
                "1",
                Patterns::Integer(0),
                "Number of digits to include in gamma string used in writing "
                "the checkpoint");
    }
    prm.leave_subsection();
}

template <int dim>
void Params<dim>::parse_parameters(ParameterHandler& prm) {
    prm.enter_subsection("Mesh and geometry");
    {
        initial_refinements = prm.get_integer("Number of initial refinements");
        final_refinements = prm.get_integer("Number of final refinements");
        starting_refinement = prm.get_integer("Starting refinement level");
        refine_direction = prm.get("Refinement direction");
        mesh_type = prm.get("Mesh type");
        beam_X = prm.get_double("Beam X");
        beam_Y = prm.get_double("Beam Y");
        beam_Z = prm.get_double("Beam Z");
        cylinder_radius = prm.get_double("Cylinder radius");
        cylinder_length = prm.get_double("Cylinder length");
        x_subdivisions = prm.get_integer("x subdivisions");
        y_subdivisions = prm.get_integer("y subdivisions");
        z_subdivisions = prm.get_integer("z subdivisions");
        fe_degree = prm.get_integer("FE degree");
        centering = prm.get("Centering");
        set_ring_config = prm.get_bool("Set ring configuration");
    }
    prm.leave_subsection();

    prm.enter_subsection("Boundary conditions");
    {
        num_boundary_domains = prm.get_integer("Number of boundary domains");
        num_boundary_stages = prm.get_integer("Number of boundary stages");
        num_boundary_conditions =
                prm.get_integer("Number of boundary conditions");
        starting_stage = prm.get_integer("Starting stage");
    }
    prm.leave_subsection();

    for (unsigned int i {0}; i != num_boundary_conditions + 1; i++) {
        prm.enter_subsection("Boundary domain " + std::to_string(i));
        {
            min_X.push_back(prm.get_double("X min"));
            max_X.push_back(prm.get_double("X max"));
            min_Y.push_back(prm.get_double("Y min"));
            max_Y.push_back(prm.get_double("Y max"));
            min_Z.push_back(prm.get_double("Z min"));
            max_Z.push_back(prm.get_double("Z max"));
        }
        prm.leave_subsection();
    }

    for (unsigned int i {0}; i != num_boundary_conditions + 1; i++) {
        prm.enter_subsection("Boundary condition " + std::to_string(i));
        {
            boundary_type.push_back(prm.get("Boundary type"));
            associated_domain[i] = prm.get_integer("Associated domain");
            constrained_domain[i] = prm.get_integer("Constrained domain");
            anchor_domain[i] = prm.get_integer("Anchor domain");
        }
        prm.leave_subsection();
    }

    for (unsigned int i {0}; i != num_boundary_stages + 1; i++) {
        prm.enter_subsection("Boundary function stage " + std::to_string(i));
        {
            num_gamma_iters.push_back(
                    prm.get_integer("Number of boundary increments"));
            use_current_config.push_back({});
            for (unsigned int j {0}; j != num_boundary_conditions; j++) {
                prm.enter_subsection("Boundary condition " + std::to_string(j));
                {
                    use_current_config[i].push_back(
                            prm.get_bool("Use current configuration"));
                    boundary_functions[i][j]->parse_parameters(prm);
                }
                prm.leave_subsection();
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
            max_linear_iters = prm.get_integer(
                    "Maximum number of linear solver iterations");
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
}
} // namespace parameters

#endif
