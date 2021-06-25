#ifndef subdivided_cylinder_h
#define subdivided_cylinder_h

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/table.h>

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

namespace SubdividedCylinder {

using namespace dealii;

template <int dim>
void subdivided_cylinder(
        Triangulation<dim>& tria,
        const unsigned int x_subdivisions,
        const double radius = 1.,
        const double half_length = 1.);

// Implementation for 3D only
template <>
void subdivided_cylinder(
        Triangulation<3>& tria,
        const unsigned int x_subdivisions,
        const double radius,
        const double half_length) {
    // Copy the base from hyper_ball<3>
    // and transform it to yz
    const double d = radius / std::sqrt(2.0);
    const double a = d / (1 + std::sqrt(2.0));

    std::vector<Point<3>> vertices;
    const double initial_height = -half_length;
    const double height_increment = 2. * half_length / x_subdivisions;

    for (unsigned int rep = 0; rep < (x_subdivisions + 1); ++rep) {
        const double height = initial_height + height_increment * rep;

        vertices.emplace_back(Point<3>(-d, height, -d));
        vertices.emplace_back(Point<3>(d, height, -d));
        vertices.emplace_back(Point<3>(-a, height, -a));
        vertices.emplace_back(Point<3>(a, height, -a));
        vertices.emplace_back(Point<3>(-a, height, a));
        vertices.emplace_back(Point<3>(a, height, a));
        vertices.emplace_back(Point<3>(-d, height, d));
        vertices.emplace_back(Point<3>(d, height, d));
    }

    // Turn cylinder such that y->x
    for (auto& vertex: vertices) {
        const double h = vertex(1);
        vertex(1) = -vertex(0);
        vertex(0) = h;
    }

    std::vector<std::vector<int>> cell_vertices;
    cell_vertices.push_back({0, 1, 8, 9, 2, 3, 10, 11});
    cell_vertices.push_back({0, 2, 8, 10, 6, 4, 14, 12});
    cell_vertices.push_back({2, 3, 10, 11, 4, 5, 12, 13});
    cell_vertices.push_back({1, 7, 9, 15, 3, 5, 11, 13});
    cell_vertices.push_back({6, 4, 14, 12, 7, 5, 15, 13});

    for (unsigned int rep = 1; rep < x_subdivisions; ++rep) {
        for (unsigned int i = 0; i < 5; ++i) {
            std::vector<int> new_cell_vertices(8);
            for (unsigned int j = 0; j < 8; ++j)
                new_cell_vertices[j] = cell_vertices[i][j] + 8 * rep;
            cell_vertices.push_back(new_cell_vertices);
        }
    }

    unsigned int n_cells = x_subdivisions * 5;

    std::vector<CellData<3>> cells(n_cells, CellData<3>());

    for (unsigned int i = 0; i < n_cells; ++i) {
        for (unsigned int j = 0; j < 8; ++j)
            cells[i].vertices[j] = cell_vertices[i][j];
        cells[i].material_id = 0;
    }

    tria.create_triangulation(
            std::vector<Point<3>>(std::begin(vertices), std::end(vertices)),
            cells,
            SubCellData()); // no boundary information

    // set boundary indicators for the
    // faces at the ends to 1 and 2,
    // respectively. note that we also
    // have to deal with those lines
    // that are purely in the interior
    // of the ends. we determine whether
    // an edge is purely in the
    // interior if one of its vertices
    // is at coordinates '+-a' as set
    // above
    tria.set_all_manifold_ids_on_boundary(0);

    double esp {half_length * 1e-5};
    for (const auto& cell: tria.cell_iterators()) {
        for (unsigned int i: GeometryInfo<3>::face_indices()) {
            if (cell->at_boundary(i)) {
                if (cell->face(i)->center()(0) > half_length - esp) {
                    cell->face(i)->set_boundary_id(2);
                    cell->face(i)->set_manifold_id(numbers::flat_manifold_id);

                    for (unsigned int e = 0;
                         e < GeometryInfo<3>::lines_per_face;
                         ++e) {
                        if ((std::fabs(cell->face(i)->line(e)->vertex(0)[1]) ==
                             a) ||
                            (std::fabs(cell->face(i)->line(e)->vertex(0)[2]) ==
                             a) ||
                            (std::fabs(cell->face(i)->line(e)->vertex(1)[1]) ==
                             a) ||
                            (std::fabs(cell->face(i)->line(e)->vertex(1)[2]) ==
                             a)) {
                            cell->face(i)->line(e)->set_boundary_id(2);
                            cell->face(i)->line(e)->set_manifold_id(
                                    numbers::flat_manifold_id);
                        }
                    }
                }
                else if (cell->face(i)->center()(0) < -half_length + esp) {
                    cell->face(i)->set_boundary_id(1);
                    cell->face(i)->set_manifold_id(numbers::flat_manifold_id);

                    for (unsigned int e = 0;
                         e < GeometryInfo<3>::lines_per_face;
                         ++e) {
                        if ((std::fabs(cell->face(i)->line(e)->vertex(0)[1]) ==
                             a) ||
                            (std::fabs(cell->face(i)->line(e)->vertex(0)[2]) ==
                             a) ||
                            (std::fabs(cell->face(i)->line(e)->vertex(1)[1]) ==
                             a) ||
                            (std::fabs(cell->face(i)->line(e)->vertex(1)[2]) ==
                             a)) {
                            cell->face(i)->line(e)->set_boundary_id(1);
                            cell->face(i)->line(e)->set_manifold_id(
                                    numbers::flat_manifold_id);
                        }
                    }
                }
            }
        }
    }
    tria.set_manifold(0, CylindricalManifold<3>());
}
} // namespace SubdividedCylinder

#endif
