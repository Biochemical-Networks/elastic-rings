#ifndef modded_periodic_functions_h
#define modded_periodic_functions_h

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_q1.h>

using namespace dealii;

template <typename FaceIterator, typename number, int dim>
void set_periodicity_constraints(
        const FaceIterator& face_1,
        const typename identity<FaceIterator>::type& face_2,
        const FullMatrix<double>& transformation,
        AffineConstraints<number>& affine_constraints,
        const std::vector<Point<dim>> dofs_to_supports,
        std::vector<double> gamma,
        const ComponentMask& component_mask,
        const bool face_orientation,
        const bool face_flip,
        const bool face_rotation,
        const number periodicity_factor) {
    static const int spacedim = FaceIterator::AccessorType::space_dimension;

    // we should be in the case where face_1 is active, i.e. has no children:
    Assert(!face_1->has_children(), ExcInternalError());

    Assert(face_1->n_active_fe_indices() == 1, ExcInternalError());

    // TODO: the implementation makes the assumption that all faces have the
    // same number of dofs
    //    AssertDimension(
    //            face_1->get_fe(face_1->nth_active_fe_index(0)).n_unique_faces(),
    //            1);
    //    AssertDimension(
    //            face_2->get_fe(face_2->nth_active_fe_index(0)).n_unique_faces(),
    //            1);
    //    const unsigned int face_no = 0;

    // If face_2 does have children, then we need to iterate over these
    // children and set periodic constraints in the inverse direction:
    if (face_2->has_children()) {
        Assert(face_2->n_children() == GeometryInfo<dim>::max_children_per_face,
               ExcNotImplemented());

        const unsigned int dofs_per_face =
                face_1->get_fe(face_1->nth_active_fe_index(0))
                        .n_dofs_per_face();
        FullMatrix<double> child_transformation(dofs_per_face, dofs_per_face);
        FullMatrix<double> subface_interpolation(dofs_per_face, dofs_per_face);

        for (unsigned int c = 0; c < face_2->n_children(); ++c) {
            // get the interpolation matrix recursively from the one that
            // interpolated from face_1 to face_2 by multiplying from the left
            // with the one that interpolates from face_2 to its child
            const auto& fe = face_1->get_fe(face_1->nth_active_fe_index(0));
            fe.get_subface_interpolation_matrix(fe, c, subface_interpolation);
            subface_interpolation.mmult(child_transformation, transformation);

            set_periodicity_constraints(
                    face_1,
                    face_2->child(c),
                    child_transformation,
                    affine_constraints,
                    dofs_to_supports,
                    gamma,
                    component_mask,
                    face_orientation,
                    face_flip,
                    face_rotation,
                    periodicity_factor);
        }
        return;
    }

    // If we reached this point then both faces are active. Now all
    // that is left is to match the corresponding DoFs of both faces.
    const unsigned int face_1_index = face_1->nth_active_fe_index(0);
    const unsigned int face_2_index = face_2->nth_active_fe_index(0);
    Assert(face_1->get_fe(face_1_index) == face_2->get_fe(face_2_index),
           ExcMessage("Matching periodic cells need to use the same finite "
                      "element"));

    const FiniteElement<dim, spacedim>& fe = face_1->get_fe(face_1_index);

    Assert(component_mask.represents_n_components(fe.n_components()),
           ExcMessage("The number of components in the mask has to be either "
                      "zero or equal to the number of components in the finite "
                      "element."));

    const unsigned int dofs_per_face = fe.n_dofs_per_face();

    std::vector<types::global_dof_index> dofs_1(dofs_per_face);
    std::vector<types::global_dof_index> dofs_2(dofs_per_face);

    face_1->get_dof_indices(dofs_1, face_1_index);
    face_2->get_dof_indices(dofs_2, face_2_index);

    // If either of the two faces has an invalid dof index, stop. This is
    // so that there is no attempt to match artificial cells of parallel
    // distributed triangulations.
    //
    // While it seems like we ought to be able to avoid even calling
    // set_periodicity_constraints for artificial faces, this situation
    // can arise when a face that is being made periodic is only
    // partially touched by the local subdomain.
    // make_periodicity_constraints will be called recursively even for
    // the section of the face that is not touched by the local
    // subdomain.
    //
    // Until there is a better way to determine if the cells that
    // neighbor a face are artificial, we simply test to see if the face
    // does not have a valid dof initialization.
    for (unsigned int i = 0; i < dofs_per_face; i++)
        if (dofs_1[i] == numbers::invalid_dof_index ||
            dofs_2[i] == numbers::invalid_dof_index) {
            return;
        }

    // Well, this is a hack:
    //
    // There is no
    //   face_to_face_index(face_index,
    //                      face_orientation,
    //                      face_flip,
    //                      face_rotation)
    // function in FiniteElementData, so we have to use
    //   face_to_cell_index(face_index, face
    //                      face_orientation,
    //                      face_flip,
    //                      face_rotation)
    // But this will give us an index on a cell - something we cannot work
    // with directly. But luckily we can match them back :-]
    std::map<unsigned int, unsigned int> cell_to_rotated_face_index;

    // Build up a cell to face index for face_2:
    for (unsigned int i = 0; i < dofs_per_face; ++i) {
        const unsigned int cell_index = fe.face_to_cell_index(
                i,
                0, /* It doesn't really matter, just
                    * assume we're on the first face...
                    */
                true,
                false,
                false // default orientation
        );
        cell_to_rotated_face_index[cell_index] = i;
    }

    // Loop over all dofs on face 2 and constrain them against all
    // matching dofs on face 1:
    for (unsigned int i = 0; i < dofs_per_face; ++i) {
        auto component_index {fe.face_system_to_component_index(i).first};
        // Obey the component mask
        if ((component_mask.n_selected_components(fe.n_components()) !=
             fe.n_components()) &&
            !component_mask[component_index])
            continue;

        // We have to be careful to treat so called "identity
        // constraints" special. These are constraints of the form
        // x1 == constraint_factor * x_2. In this case, if the constraint
        // x2 == 1./constraint_factor * x1 already exists we are in trouble.
        //
        // Consequently, we have to check that we have indeed such an
        // "identity constraint". We do this by looping over all entries
        // of the row of the transformation matrix and check whether we
        // find exactly one nonzero entry. If this is the case, set
        // "is_identity_constrained" to true and record the corresponding
        // index and constraint_factor.
        bool is_identity_constrained = false;
        unsigned int target = numbers::invalid_unsigned_int;
        number constraint_factor = periodicity_factor;

        constexpr double eps = 1.e-13;
        for (unsigned int jj = 0; jj < dofs_per_face; ++jj) {
            const auto entry = transformation(i, jj);
            if (std::abs(entry) > eps) {
                if (is_identity_constrained) {
                    // We did encounter more than one nonzero entry, so
                    // the dof is not identity constrained. Set the
                    // boolean to false and break out of the for loop.
                    is_identity_constrained = false;
                    break;
                }
                is_identity_constrained = true;
                target = jj;
                constraint_factor = entry * periodicity_factor;
            }
        }

        // Next, we work on all constraints that are not identity
        // constraints, i.e., constraints that involve an interpolation
        // step that constrains the current dof (on face 2) to more than
        // one dof on face 1.
        if (!is_identity_constrained) {

            // The current dof is already constrained. There is nothing
            // left to do.
            if (affine_constraints.is_constrained(dofs_2[i]))
                continue;

            // Enter the constraint piece by piece:
            affine_constraints.add_line(dofs_2[i]);

            for (unsigned int jj = 0; jj < dofs_per_face; ++jj) {

                // Get the correct dof index on face_1 respecting the
                // given orientation:
                const unsigned int j =
                        cell_to_rotated_face_index[fe.face_to_cell_index(
                                jj,
                                0,
                                face_orientation,
                                face_flip,
                                face_rotation)];

                if (std::abs(transformation(i, jj)) > eps)
                    affine_constraints.add_entry(
                            dofs_2[i], dofs_1[j], transformation(i, jj));
            }

            // Continue with next dof.
            continue;
        }

        // We are left with an "identity constraint".

        // Get the correct dof index on face_1 respecting the given
        // orientation:
        const unsigned int j = cell_to_rotated_face_index[fe.face_to_cell_index(
                target, 0, face_orientation, face_flip, face_rotation)];

        auto dof_left = dofs_1[j];
        auto dof_right = dofs_2[i];

        // If dof_left is already constrained, or dof_left < dof_right we
        // flip the order to ensure that dofs are constrained in a stable
        // manner on different MPI processes.
        if (affine_constraints.is_constrained(dof_left) ||
            (dof_left < dof_right &&
             !affine_constraints.is_constrained(dof_right))) {
            std::swap(dof_left, dof_right);
            constraint_factor = 1. / constraint_factor;
        }

        // Next, we try to enter the constraint
        //   dof_left = constraint_factor * dof_right;

        // If both degrees of freedom are constrained, there is nothing we
        // can do. Simply continue with the next dof.
        if (affine_constraints.is_constrained(dof_left) &&
            affine_constraints.is_constrained(dof_right))
            continue;

        // We have to be careful that adding the current identity
        // constraint does not create a constraint cycle. Thus, check for
        // a dependency cycle:
        bool constraints_are_cyclic = true;
        number cycle_constraint_factor = constraint_factor;

        for (auto test_dof = dof_right; test_dof != dof_left;) {
            if (!affine_constraints.is_constrained(test_dof)) {
                constraints_are_cyclic = false;
                break;
            }

            const auto& constraint_entries =
                    *affine_constraints.get_constraint_entries(test_dof);
            if (constraint_entries.size() == 1) {
                test_dof = constraint_entries[0].first;
                cycle_constraint_factor *= constraint_entries[0].second;
            }
            else {
                constraints_are_cyclic = false;
                break;
            }
        }

        // In case of a dependency cycle we, either
        //  - do nothing if cycle_constraint_factor == 1. In this case all
        //    degrees
        //    of freedom are already periodically constrained,
        //  - otherwise, force all dofs to zero (by setting dof_left to
        //    zero). The reasoning behind this is the fact that
        //    cycle_constraint_factor != 1 occurs in situations such as
        //    x1 == x2 and x2 == -1. * x1. This system is only solved by
        //    x_1 = x_2 = 0.
        if (constraints_are_cyclic) {
            if (std::abs(cycle_constraint_factor - 1.) > eps)
                affine_constraints.add_line(dof_left);
        }
        else {
            affine_constraints.add_line(dof_left);
            affine_constraints.add_entry(
                    dof_left, dof_right, constraint_factor);

            // My addition for inhomogeneity
            Point<dim> dof_left_support {dofs_to_supports[dof_left]};
            Point<dim> dof_right_support {dofs_to_supports[dof_right]};
            Point<dim> diff {dof_right_support - dof_left_support};
            // Point<dim> offset {gamma[0]*1, 0, 0};
            Point<dim> offset {0, 0, 0};
            Point<dim> ref {
                    abs(diff[0]), dof_left_support[1], dof_left_support[2]};
            offset[0] = gamma[0] * (ref[0] * (2 / numbers::PI - 1) + ref[1]);
            offset[1] = gamma[1] * (-2 * ref[0] / numbers::PI - ref[1]);
            offset[2] = 0;
            //offset[0] = (1.0 - gamma[0]) *
            //                    (ref[0] * (2 / numbers::PI - 1) + ref[1]) +
            //            gamma[0] * -ref[0];
            //offset[1] =
            //        (1.0 - gamma[1]) * (-2 * ref[0] / numbers::PI - ref[1]) +
            //        gamma[1] * (-ref[0] / numbers::PI - 2 * ref[1]);
            //offset[2] = 0;
            affine_constraints.set_inhomogeneity(
                    dof_left, offset[component_index]);

            // The number 1e10 in the assert below is arbitrary. If the
            // absolute value of constraint_factor is too large, then probably
            // the absolute value of periodicity_factor is too large or too
            // small. This would be equivalent to an evanescent wave that has
            // a very small wavelength. A quick calculation shows that if
            // |periodicity_factor| > 1e10 -> |np.exp(ikd)|> 1e10, therefore k
            // is imaginary (evanescent wave) and the evanescent wavelength is
            // 0.27 times smaller than the dimension of the structure,
            // lambda=((2*pi)/log(1e10))*d. Imaginary wavenumbers can be
            // interesting in some cases
            // (https://doi.org/10.1103/PhysRevA.94.033813).In order to
            // implement the case of in which the wavevector can be imaginary
            // it would be necessary to rewrite this function and the dof
            // ordering method should be modified.
            // Let's take the following constraint a*x1 + b*x2 = 0. You could
            // just always pick x1 = b/a*x2, but in practice this is not so
            // stable if a could be a small number -- intended to be zero, but
            // just very small due to roundoff. Of course, constraining x2 in
            // terms of x1 has the same problem. So one chooses x1 = b/a*x2 if
            // |b|<|a|, and x2 = a/b*x1 if |a|<|b|.
            Assert(std::abs(constraint_factor) < 1e10,
                   ExcMessage("The periodicity constraint is too large. The "
                              "parameter periodicity_factor might be too large "
                              "or too small."));
        }
    } /* for dofs_per_face */
}

template <int dim, int spacedim>
FullMatrix<double> compute_transformation(
        const FiniteElement<dim, spacedim>& fe,
        const FullMatrix<double>& matrix,
        const std::vector<unsigned int>& first_vector_components) {
    // TODO: the implementation makes the assumption that all faces have the
    // same number of dofs
    //    AssertDimension(fe.n_unique_faces(), 1);
    //    const unsigned int face_no = 0;

    Assert(matrix.m() == matrix.n(), ExcInternalError());

    const unsigned int n_dofs_per_face = fe.n_dofs_per_face();

    if (matrix.m() == n_dofs_per_face) {
        // In case of m == n == n_dofs_per_face the supplied matrix is already
        // an interpolation matrix, so we use it directly:
        return matrix;
    }

    if (first_vector_components.empty() && matrix.m() == 0) {
        // Just the identity matrix in case no rotation is specified:
        return IdentityMatrix(n_dofs_per_face);
    }

    // The matrix describes a rotation and we have to build a transformation
    // matrix, we assume that for a 0* rotation we would have to build the
    // identity matrix

    Assert(matrix.m() == spacedim, ExcInternalError())

            Quadrature<dim - 1>
                    quadrature(fe.get_unit_face_support_points());

    // have an array that stores the location of each vector-dof tuple we want
    // to rotate.
    using DoFTuple = std::array<unsigned int, spacedim>;

    // start with a pristine interpolation matrix...
    FullMatrix<double> transformation = IdentityMatrix(n_dofs_per_face);

    for (unsigned int i = 0; i < n_dofs_per_face; ++i) {
        std::vector<unsigned int>::const_iterator comp_it = std::find(
                first_vector_components.begin(),
                first_vector_components.end(),
                fe.face_system_to_component_index(i).first);
        if (comp_it != first_vector_components.end()) {
            const unsigned int first_vector_component = *comp_it;

            // find corresponding other components of vector
            DoFTuple vector_dofs;
            vector_dofs[0] = i;
            unsigned int n_found = 1;

            Assert(*comp_it + spacedim <= fe.n_components(),
                   ExcMessage("Error: the finite element does not have enough "
                              "components "
                              "to define rotated periodic boundaries."));

            for (unsigned int k = 0; k < n_dofs_per_face; ++k)
                if ((k != i) && (quadrature.point(k) == quadrature.point(i)) &&
                    (fe.face_system_to_component_index(k).first >=
                     first_vector_component) &&
                    (fe.face_system_to_component_index(k).first <
                     first_vector_component + spacedim)) {
                    vector_dofs
                            [fe.face_system_to_component_index(k).first -
                             first_vector_component] = k;
                    n_found++;
                    if (n_found == dim)
                        break;
                }

            // ... and rotate all dofs belonging to vector valued components
            // that are selected by first_vector_components:
            for (int i = 0; i < spacedim; ++i) {
                transformation[vector_dofs[i]][vector_dofs[i]] = 0.;
                for (int j = 0; j < spacedim; ++j)
                    transformation[vector_dofs[i]][vector_dofs[j]] =
                            matrix[i][j];
            }
        }
    }
    return transformation;
}

template <typename FaceIterator, typename number, int dim>
void make_periodicity_constraints(
        const FaceIterator& face_1,
        const typename identity<FaceIterator>::type& face_2,
        AffineConstraints<number>& constraints,
        std::vector<Point<dim>> dofs_to_supports,
        std::vector<double> gamma,
        const ComponentMask& component_mask = ComponentMask(),
        const bool face_orientation = true,
        const bool face_flip = false,
        const bool face_rotation = false,
        const FullMatrix<double>& matrix = FullMatrix<double>(),
        const std::vector<unsigned int>& first_vector_components =
                std::vector<unsigned int>(),
        const number periodicity_factor = 1.);

template <typename FaceIterator, typename number, int dim>
void make_periodicity_constraints(
        const FaceIterator& face_1,
        const typename identity<FaceIterator>::type& face_2,
        AffineConstraints<number>& affine_constraints,
        std::vector<Point<dim>> dofs_to_supports,
        std::vector<double> gamma,
        const ComponentMask& component_mask,
        const bool face_orientation,
        const bool face_flip,
        const bool face_rotation,
        const FullMatrix<double>& matrix,
        const std::vector<unsigned int>& first_vector_components,
        const number periodicity_factor) {
    // TODO: the implementation makes the assumption that all faces have the
    // same number of dofs
    //    AssertDimension(
    //            face_1->get_fe(face_1->nth_active_fe_index(0)).n_unique_faces(),
    //            1);
    //    AssertDimension(
    //            face_2->get_fe(face_2->nth_active_fe_index(0)).n_unique_faces(),
    //            1);
    //    const unsigned int face_no = 0;

    static const int spacedim = FaceIterator::AccessorType::space_dimension;

    Assert((dim != 1) || (face_orientation == true && face_flip == false &&
                          face_rotation == false),
           ExcMessage("The supplied orientation "
                      "(face_orientation, face_flip, face_rotation) "
                      "is invalid for 1D"));

    Assert((dim != 2) || (face_orientation == true && face_rotation == false),
           ExcMessage("The supplied orientation "
                      "(face_orientation, face_flip, face_rotation) "
                      "is invalid for 2D"));

    Assert(face_1 != face_2,
           ExcMessage("face_1 and face_2 are equal! Cannot constrain DoFs "
                      "on the very same face"));

    Assert(face_1->at_boundary() && face_2->at_boundary(),
           ExcMessage("Faces for periodicity constraints must be on the "
                      "boundary"));

    Assert(matrix.m() == matrix.n(),
           ExcMessage("The supplied (rotation or interpolation) matrix must "
                      "be a square matrix"));

    Assert(first_vector_components.empty() || matrix.m() == spacedim,
           ExcMessage("first_vector_components is nonempty, so matrix must "
                      "be a rotation matrix exactly of size spacedim"));

#ifdef DEBUG
    if (!face_1->has_children()) {
        Assert(face_1->n_active_fe_indices() == 1, ExcInternalError());
        const unsigned int n_dofs_per_face =
                face_1->get_fe(face_1->nth_active_fe_index(0))
                        .n_dofs_per_face();

        Assert(matrix.m() == 0 ||
                       (first_vector_components.empty() &&
                        matrix.m() == n_dofs_per_face) ||
                       (!first_vector_components.empty() &&
                        matrix.m() == spacedim),
               ExcMessage(
                       "The matrix must have either size 0 or spacedim "
                       "(if first_vector_components is nonempty) "
                       "or the size must be equal to the # of DoFs on the face "
                       "(if first_vector_components is empty)."));
    }

    if (!face_2->has_children()) {
        Assert(face_2->n_active_fe_indices() == 1, ExcInternalError());
        const unsigned int n_dofs_per_face =
                face_2->get_fe(face_2->nth_active_fe_index(0))
                        .n_dofs_per_face();

        Assert(matrix.m() == 0 ||
                       (first_vector_components.empty() &&
                        matrix.m() == n_dofs_per_face) ||
                       (!first_vector_components.empty() &&
                        matrix.m() == spacedim),
               ExcMessage(
                       "The matrix must have either size 0 or spacedim "
                       "(if first_vector_components is nonempty) "
                       "or the size must be equal to the # of DoFs on the face "
                       "(if first_vector_components is empty)."));
    }
#endif

    // A lookup table on how to go through the child faces depending on the
    // orientation:

    static const int lookup_table_2d[2][2] = {
            //          flip:
            {0, 1}, //  false
            {1, 0}, //  true
    };

    static const int lookup_table_3d[2][2][2][4] = {
            //                    orientation flip  rotation
            {
                    {
                            {0, 2, 1, 3}, //  false       false false
                            {2, 3, 0, 1}, //  false       false true
                    },
                    {
                            {3, 1, 2, 0}, //  false       true  false
                            {1, 0, 3, 2}, //  false       true  true
                    },
            },
            {
                    {
                            {0, 1, 2, 3}, //  true        false false
                            {1, 3, 0, 2}, //  true        false true
                    },
                    {
                            {3, 2, 1, 0}, //  true        true  false
                            {2, 0, 3, 1}, //  true        true  true
                    },
            },
    };

    if (face_1->has_children() && face_2->has_children()) {
        // In the case that both faces have children, we loop over all children
        // and apply make_periodicty_constrains recursively:

        Assert(face_1->n_children() ==
                               GeometryInfo<dim>::max_children_per_face &&
                       face_2->n_children() ==
                               GeometryInfo<dim>::max_children_per_face,
               ExcNotImplemented());

        for (unsigned int i = 0; i < GeometryInfo<dim>::max_children_per_face;
             ++i) {
            // Lookup the index for the second face
            unsigned int j;
            switch (dim) {
            case 2:
                j = lookup_table_2d[face_flip][i];
                break;
            case 3:
                j = lookup_table_3d[face_orientation][face_flip][face_rotation]
                                   [i];
                break;
            default:
                AssertThrow(false, ExcNotImplemented());
            }

            make_periodicity_constraints(
                    face_1->child(i),
                    face_2->child(j),
                    affine_constraints,
                    dofs_to_supports,
                    gamma,
                    component_mask,
                    face_orientation,
                    face_flip,
                    face_rotation,
                    matrix,
                    first_vector_components,
                    periodicity_factor);
        }
    }
    else {
        // Otherwise at least one of the two faces is active and we need to do
        // some work and enter the constraints!

        // The finite element that matters is the one on the active face:
        const FiniteElement<dim, spacedim>& fe =
                face_1->has_children() ?
                        face_2->get_fe(face_2->nth_active_fe_index(0)) :
                        face_1->get_fe(face_1->nth_active_fe_index(0));

        const unsigned int n_dofs_per_face = fe.n_dofs_per_face();

        // Sometimes we just have nothing to do (for all finite elements, or
        // systems which accidentally don't have any dofs on the boundary).
        if (n_dofs_per_face == 0)
            return;

        const FullMatrix<double> transformation =
                compute_transformation(fe, matrix, first_vector_components);

        if (!face_2->has_children()) {
            // Performance hack: We do not need to compute an inverse if the
            // matrix is the identity matrix.
            if (first_vector_components.empty() && matrix.m() == 0) {
                set_periodicity_constraints(
                        face_2,
                        face_1,
                        transformation,
                        affine_constraints,
                        dofs_to_supports,
                        gamma,
                        component_mask,
                        face_orientation,
                        face_flip,
                        face_rotation,
                        periodicity_factor);
            }
            else {
                FullMatrix<double> inverse(transformation.m());
                inverse.invert(transformation);

                set_periodicity_constraints(
                        face_2,
                        face_1,
                        inverse,
                        affine_constraints,
                        dofs_to_supports,
                        gamma,
                        component_mask,
                        face_orientation,
                        face_flip,
                        face_rotation,
                        periodicity_factor);
            }
        }
        else {
            Assert(!face_1->has_children(), ExcInternalError());

            // Important note:
            // In 3D we have to take care of the fact that face_rotation gives
            // the relative rotation of face_1 to face_2, i.e. we have to invert
            // the rotation when constraining face_2 to face_1. Therefore
            // face_flip has to be toggled if face_rotation is true: In case of
            // inverted orientation, nothing has to be done.
            set_periodicity_constraints(
                    face_1,
                    face_2,
                    transformation,
                    affine_constraints,
                    dofs_to_supports,
                    gamma,
                    component_mask,
                    face_orientation,
                    face_orientation ? face_rotation ^ face_flip : face_flip,
                    face_rotation,
                    periodicity_factor);
        }
    }
}

template <int dim, int spacedim, typename number>
void make_periodicity_constraints(
        const std::vector<GridTools::PeriodicFacePair<
                typename DoFHandler<dim, spacedim>::cell_iterator>>&
                periodic_faces,
        AffineConstraints<number>& constraints,
        std::vector<Point<dim>> dofs_to_supports,
        std::vector<double> gamma,
        const ComponentMask& component_mask = ComponentMask(),
        const std::vector<unsigned int>& first_vector_components =
                std::vector<unsigned int>(),
        const number periodicity_factor = 1.);

template <int dim, int spacedim, typename number>
void make_periodicity_constraints(
        const std::vector<GridTools::PeriodicFacePair<
                typename DoFHandler<dim, spacedim>::cell_iterator>>&
                periodic_faces,
        AffineConstraints<number>& constraints,
        std::vector<Point<dim>> dofs_to_supports,
        std::vector<double> gamma,
        const ComponentMask& component_mask,
        const std::vector<unsigned int>& first_vector_components,
        const number periodicity_factor) {
    // Loop over all periodic faces...
    for (auto& pair: periodic_faces) {
        using FaceIterator = typename DoFHandler<dim, spacedim>::face_iterator;
        const FaceIterator face_1 = pair.cell[0]->face(pair.face_idx[0]);
        const FaceIterator face_2 = pair.cell[1]->face(pair.face_idx[1]);

        Assert(face_1->at_boundary() && face_2->at_boundary(),
               ExcInternalError());

        Assert(face_1 != face_2, ExcInternalError());

        // ... and apply the low level make_periodicity_constraints function to
        // every matching pair:
        make_periodicity_constraints(
                face_1,
                face_2,
                constraints,
                dofs_to_supports,
                gamma,
                component_mask,
                pair.orientation[0],
                pair.orientation[1],
                pair.orientation[2],
                pair.matrix,
                first_vector_components,
                periodicity_factor);
    }
}

#endif
