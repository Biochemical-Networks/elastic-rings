#ifndef postprocessing_h
#define postprocessing_h

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>
#include <deal.II/numerics/data_postprocessor.h>

using namespace dealii;

template <int dim>
class FacesPostprocessor: public DataPostprocessor<dim> {
  public:
    FacesPostprocessor(const double lambda, const double mu);
    virtual void evaluate_vector_field(
            const DataPostprocessorInputs::Vector<dim>& input_data,
            std::vector<Vector<double>>& computed_quantities) const;
    virtual std::vector<std::string> get_names() const;
    virtual std::vector<
            DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const;
    virtual UpdateFlags get_needed_update_flags() const;

  private:
    const double lambda;
    const double mu;
    const std::vector<std::string> scalar_names {"grad_norm"};
    const std::vector<std::string> vector_names {
            "displacement",
            "material_normal",
            "material_stress",
            "spatial_normal",
            "spatial_stress"};
};

template <int dim>
FacesPostprocessor<dim>::FacesPostprocessor(
        const double lambda,
        const double mu):
        lambda {lambda}, mu {mu} {}

template <int dim>
unsigned int add_scalar_to_computed(
        unsigned int q_i,
        unsigned int cq_i,
        double q,
        std::vector<Vector<double>>& computed_quantities) {
    computed_quantities[q_i][cq_i] = q;
    cq_i++;

    return cq_i;
}

template <int dim>
unsigned int add_vector_to_computed(
        unsigned int q_i,
        unsigned int cq_i,
        Tensor<1, dim, double> q,
        std::vector<Vector<double>>& computed_quantities) {

    for (unsigned int d {0}; d != dim; d++) {
        computed_quantities[q_i][cq_i] = q[d];
        cq_i++;
    }

    return cq_i;
}

template <int dim>
void FacesPostprocessor<dim>::evaluate_vector_field(
        const DataPostprocessorInputs::Vector<dim>& input_data,
        std::vector<Vector<double>>& computed_quantities) const {

    for (unsigned int q_i {0}; q_i != input_data.normals.size(); q_i++) {
        Tensor<1, dim, double> displacement;
        Tensor<2, dim, double> grad_u {};
        Tensor<2, dim, double> identity_rank2 {};
        for (unsigned int d {0}; d != dim; d++) {
            displacement[d] = input_data.solution_values[q_i][d];
            grad_u[d] = input_data.solution_gradients[q_i][d];
            identity_rank2[d][d] = 1;

        }
        const Tensor<1, dim, double> material_normal {input_data.normals[q_i]};

        const Tensor<2, dim, double> grad_u_T {transpose(grad_u)};
        const Tensor<2, dim, double> green_lagrange_strain {
                0.5 * (grad_u + grad_u_T + grad_u * grad_u_T)};
        const Tensor<2, dim, double> piola_kirchhoff {
                lambda * trace(green_lagrange_strain) * identity_rank2 +
                mu * green_lagrange_strain};

        const Tensor<1, dim, double> material_stress =
                piola_kirchhoff * material_normal;

        const Tensor<2, dim, double> deformation_grad {grad_u + identity_rank2};
        const double deformation_grad_det {determinant(deformation_grad)};
        const Tensor<2, dim, double> cauchy {
                deformation_grad * piola_kirchhoff *
                transpose(deformation_grad) / deformation_grad_det};

        Tensor<1, dim, double> spatial_normal {
                transpose(invert(deformation_grad)) * material_normal};
        spatial_normal /= spatial_normal.norm();
        const Tensor<1, dim, double> spatial_stress {cauchy * spatial_normal};
        double grad_norm {sqrt(double_contract<0, 0, 1, 1>(grad_u, grad_u))};

        // For now these need to be done in the right order
        unsigned int cq_i {0};
        cq_i = add_scalar_to_computed<dim>(
                q_i, cq_i, grad_norm, computed_quantities);
        cq_i = add_vector_to_computed<dim>(
                q_i, cq_i, displacement, computed_quantities);
        cq_i = add_vector_to_computed<dim>(
                q_i, cq_i, material_normal, computed_quantities);
        cq_i = add_vector_to_computed<dim>(
                q_i, cq_i, material_stress, computed_quantities);
        cq_i = add_vector_to_computed<dim>(
                q_i, cq_i, spatial_normal, computed_quantities);
        cq_i = add_vector_to_computed<dim>(
                q_i, cq_i, spatial_stress, computed_quantities);
    }
}

template <int dim>
void insert_vector_names(
        std::vector<std::string>& solution_names,
        std::string name) {
    std::vector<std::string> names(dim, name);
    solution_names.insert(solution_names.end(), names.begin(), names.end());
}

template <int dim>
std::vector<std::string> FacesPostprocessor<dim>::get_names() const {
    std::vector<std::string> solution_names {};
    for (auto name: scalar_names) {
        solution_names.push_back(name);
    }
    for (auto name: vector_names) {
        insert_vector_names<dim>(solution_names, name);
    }

    return solution_names;
}

template <int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
FacesPostprocessor<dim>::get_data_component_interpretation() const {
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
            interpretation {};
    for (unsigned int i {0}; i != scalar_names.size(); i++) {
        interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
    }
    for (unsigned int d {0}; d != dim; d++) {
        for (unsigned int i {0}; i != vector_names.size(); i++) {
            interpretation.push_back(
                    DataComponentInterpretation::component_is_part_of_vector);
        }
    }

    return interpretation;
}

template <int dim>
UpdateFlags FacesPostprocessor<dim>::get_needed_update_flags() const {
    return update_values | update_gradients | update_quadrature_points |
           update_normal_vectors;
}

template <int dim>
class SpatialStressVectorMovedMeshPostprocess:
        public DataPostprocessorVector<dim> {
  public:
    SpatialStressVectorMovedMeshPostprocess(
            const double lambda,
            const double mu);
    virtual void evaluate_vector_field(
            const DataPostprocessorInputs::Vector<dim>& input_data,
            std::vector<Vector<double>>& computed_quantities) const;

  private:
    const double lambda;
    const double mu;
};

template <int dim>
SpatialStressVectorMovedMeshPostprocess<dim>::
        SpatialStressVectorMovedMeshPostprocess(
                const double lambda,
                const double mu):
        DataPostprocessorVector<dim>(
                "spatial_stress",
                update_gradients | update_normal_vectors),
        lambda {lambda},
        mu {mu} {}

template <int dim>
void SpatialStressVectorMovedMeshPostprocess<dim>::evaluate_vector_field(
        const DataPostprocessorInputs::Vector<dim>& input_data,
        std::vector<Vector<double>>& computed_quantities) const {

    for (unsigned int i {0}; i != input_data.normals.size(); i++) {
        Tensor<2, dim, double> grad_u {};
        Tensor<2, dim, double> identity_tensor {};
        for (unsigned int d {0}; d != dim; d++) {
            grad_u[d] = input_data.solution_gradients[i][d];
            identity_tensor[d][d] = 1;
        }
        const Tensor<2, dim, double> grad_u_T {transpose(grad_u)};
        const Tensor<2, dim, double> green_lagrange_strain_tensor {
                0.5 * (grad_u + grad_u_T + grad_u * grad_u_T)};
        const Tensor<2, dim, double> piola_kirchhoff_tensor {
                lambda * trace(green_lagrange_strain_tensor) * identity_tensor +
                mu * green_lagrange_strain_tensor};
        const Tensor<2, dim, double> deformation_grad {
                grad_u + identity_tensor};
        const double deformation_grad_det {determinant(deformation_grad)};
        const Tensor<2, dim, double> cauchy_tensor {
                deformation_grad * piola_kirchhoff_tensor *
                transpose(deformation_grad) / deformation_grad_det};
        const Tensor<1, dim, double> spatial_normal_vector {
                input_data.normals[i]};
        const Tensor<1, dim, double> spatial_stress_vector_t =
                cauchy_tensor * spatial_normal_vector;
        Vector<double> spatial_stress_vector(dim);
        for (unsigned int d {0}; d != dim; d++) {
            spatial_stress_vector[d] = spatial_stress_vector_t[d];
        }
        computed_quantities[i] = spatial_stress_vector;
    }
}

#endif
