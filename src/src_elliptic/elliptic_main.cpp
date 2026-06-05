#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace dealii;

namespace
{
  constexpr unsigned int dim = 3;

  const std::vector<double> study_p_values = {1.0, 3.0, 5.0};

  const std::vector<unsigned int> study_refinements = {3, 4, 5};

  const std::vector<std::string> study_preconditioners = {
    "none", "jacobi", "ssor", "ilu", "amg"};

  struct Ball
  {
    Point<dim> center;
    double     radius;
  };

  std::vector<Ball>
  make_balls()
  {
    return {{{0.18, 0.20, 0.23}, 0.090},
            {{0.34, 0.28, 0.72}, 0.075},
            {{0.52, 0.18, 0.46}, 0.080},
            {{0.76, 0.24, 0.32}, 0.070},
            {{0.24, 0.52, 0.55}, 0.085},
            {{0.48, 0.48, 0.22}, 0.070},
            {{0.68, 0.56, 0.78}, 0.095},
            {{0.86, 0.62, 0.50}, 0.060},
            {{0.30, 0.78, 0.30}, 0.075},
            {{0.55, 0.82, 0.60}, 0.090},
            {{0.78, 0.84, 0.20}, 0.070},
            {{0.14, 0.86, 0.82}, 0.065}};
  }

  double
  diffusion_value(const Point<dim> &point,
                  const double      p,
                  const std::vector<Ball> &balls)
  {
    for (const auto &ball : balls)
      if (point.distance(ball.center) <= ball.radius)
        return std::pow(10.0, p);

    return 1.0;
  }

  class DiffusionCoefficient : public Function<dim>
  {
  public:
    DiffusionCoefficient(const double p_, const std::vector<Ball> &balls_)
      : p(p_)
      , balls(balls_)
    {}

    double
    value(const Point<dim> &point,
          const unsigned int /*component*/ = 0) const override
    {
      return diffusion_value(point, p, balls);
    }

  private:
    const double             p;
    const std::vector<Ball> &balls;
  };

  class RightHandSide : public Function<dim>
  {
  public:
    double
    value(const Point<dim> & /*point*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 1.0;
    }
  };

  void
  setup_problem(const unsigned int                      refinements,
                parallel::distributed::Triangulation<dim> &mesh,
                FE_Q<dim>                             &fe,
                DoFHandler<dim>                       &dof_handler,
                AffineConstraints<double>             &constraints,
                TrilinosWrappers::SparsityPattern     &sparsity_pattern,
                TrilinosWrappers::SparseMatrix        &system_matrix,
                TrilinosWrappers::MPI::Vector         &solution,
                TrilinosWrappers::MPI::Vector         &system_rhs,
                IndexSet                              &locally_owned_dofs,
                IndexSet                              &locally_relevant_dofs,
                const MPI_Comm                         mpi_communicator)
  {
    GridGenerator::hyper_cube(mesh, 0.0, 1.0);
    mesh.refine_global(refinements);

    dof_handler.distribute_dofs(fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(),
                                             constraints);
    constraints.close();

    sparsity_pattern.reinit(locally_owned_dofs,
                            locally_owned_dofs,
                            locally_relevant_dofs,
                            mpi_communicator);
    DoFTools::make_sparsity_pattern(dof_handler,
                                    sparsity_pattern,
                                    constraints,
                                    false,
                                    Utilities::MPI::this_mpi_process(
                                      mpi_communicator));
    sparsity_pattern.compress();
    system_matrix.reinit(sparsity_pattern);

    solution.reinit(locally_owned_dofs, mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        std::cout << "cells " << mesh.n_global_active_cells() << std::endl;
        std::cout << "dofs " << dof_handler.n_dofs() << std::endl;
      }
  }

  void
  assemble_system(const DoFHandler<dim>               &dof_handler,
                  const FE_Q<dim>                    &fe,
                  const AffineConstraints<double>    &constraints,
                  const double                        p,
                  const std::vector<Ball>            &balls,
                  TrilinosWrappers::SparseMatrix     &system_matrix,
                  TrilinosWrappers::MPI::Vector      &system_rhs)
  {
    const QGauss<dim> quadrature(fe.degree + 1);
    FEValues<dim>     fe_values(fe,
                            quadrature,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q           = quadrature.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    DiffusionCoefficient diffusion(p, balls);
    RightHandSide        rhs;

    system_matrix = 0.0;
    system_rhs    = 0.0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        fe_values.reinit(cell);
        cell_matrix = 0.0;
        cell_rhs    = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
          {
            const double mu = diffusion.value(fe_values.quadrature_point(q));

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  cell_matrix(i, j) += mu * fe_values.shape_grad(i, q) *
                                       fe_values.shape_grad(j, q) *
                                       fe_values.JxW(q);

                cell_rhs(i) += rhs.value(fe_values.quadrature_point(q)) *
                               fe_values.shape_value(i, q) * fe_values.JxW(q);
              }
          }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix,
                                               cell_rhs,
                                               local_dof_indices,
                                               system_matrix,
                                               system_rhs);
      }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }

  template <class Preconditioner>
  unsigned int
  solve_cg(const TrilinosWrappers::SparseMatrix   &system_matrix,
           const TrilinosWrappers::MPI::Vector    &system_rhs,
           TrilinosWrappers::MPI::Vector          &solution,
           const Preconditioner                   &preconditioner,
           const std::string                      &name,
           const double                            setup_time,
           const MPI_Comm                          mpi_communicator)
  {
    SolverControl solver_control(20000, 1e-10 * system_rhs.l2_norm());
    SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

    Timer timer;
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    timer.stop();

    const double solve_time =
      Utilities::MPI::max(timer.wall_time(), mpi_communicator);
    const double setup_time_global =
      Utilities::MPI::max(setup_time, mpi_communicator);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      std::cout << "result " << name << " " << solver_control.last_step()
                << " " << setup_time_global << " " << solve_time << " "
                << setup_time_global + solve_time << std::endl;

    return solver_control.last_step();
  }

  void
  solve_with_preconditioner(const std::string                  &name,
                            const TrilinosWrappers::SparseMatrix &system_matrix,
                            const TrilinosWrappers::MPI::Vector  &system_rhs,
                            TrilinosWrappers::MPI::Vector        &solution,
                            const MPI_Comm                        mpi_communicator)
  {
    solution = 0.0;

    try
      {
        if (name == "none")
          {
            solve_cg(system_matrix,
                     system_rhs,
                     solution,
                     PreconditionIdentity(),
                     name,
                     0.0,
                     mpi_communicator);
            return;
          }

        Timer timer;

        if (name == "jacobi")
          {
            TrilinosWrappers::PreconditionJacobi preconditioner;
            preconditioner.initialize(system_matrix);
            timer.stop();
            solve_cg(system_matrix,
                     system_rhs,
                     solution,
                     preconditioner,
                     name,
                     timer.wall_time(),
                     mpi_communicator);
            return;
          }

        if (name == "ssor")
          {
            TrilinosWrappers::PreconditionSSOR preconditioner;
            preconditioner.initialize(system_matrix);
            timer.stop();
            solve_cg(system_matrix,
                     system_rhs,
                     solution,
                     preconditioner,
                     name,
                     timer.wall_time(),
                     mpi_communicator);
            return;
          }

        if (name == "ilu")
          {
            TrilinosWrappers::PreconditionILU preconditioner;
            preconditioner.initialize(system_matrix);
            timer.stop();
            solve_cg(system_matrix,
                     system_rhs,
                     solution,
                     preconditioner,
                     name,
                     timer.wall_time(),
                     mpi_communicator);
            return;
          }

        TrilinosWrappers::PreconditionAMG preconditioner;
        TrilinosWrappers::PreconditionAMG::AdditionalData data;
        data.elliptic              = true;
        data.higher_order_elements = true;
        data.n_cycles              = 1;
        preconditioner.initialize(system_matrix, data);
        timer.stop();
        solve_cg(system_matrix,
                 system_rhs,
                 solution,
                 preconditioner,
                 "amg",
                 timer.wall_time(),
                 mpi_communicator);
      }
    catch (const std::exception &exception)
      {
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
          std::cout << "failed " << name << " " << exception.what()
                    << std::endl;
      }
  }

  void
  write_vtk(const DoFHandler<dim>                  &dof_handler,
            const double                           p,
            const std::vector<Ball>               &balls,
            const TrilinosWrappers::MPI::Vector   &solution)
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");

    Vector<double> mu(dof_handler.get_triangulation().n_active_cells());
    unsigned int   index = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      mu[index++] = diffusion_value(cell->center(), p, balls);

    data_out.add_data_vector(mu, "mu");
    data_out.build_patches();

    const std::string name = "heterogeneous-p" + std::to_string(int(p)) + ".vtk";
    std::ofstream     out(name);
    data_out.write_vtk(out);

    std::cout << "vtk " << name << std::endl;
  }

  void
  run_case(const double                    p,
           const unsigned int              refinements,
           const std::vector<std::string> &preconditioners,
           const MPI_Comm                  mpi_communicator)
  {
    const unsigned int my_rank =
      Utilities::MPI::this_mpi_process(mpi_communicator);

    if (my_rank == 0)
      {
        std::cout << "mpi_procs "
                  << Utilities::MPI::n_mpi_processes(mpi_communicator)
                  << std::endl;
        std::cout << "p " << p << std::endl;
        std::cout << "refinements " << refinements << std::endl;
        std::cout << "balls " << make_balls().size() << std::endl;
      }

    const std::vector<Ball> balls = make_balls();

    parallel::distributed::Triangulation<dim> mesh(mpi_communicator);
    FE_Q<dim>                         fe(1);
    DoFHandler<dim>                   dof_handler(mesh);
    AffineConstraints<double>         constraints;
    TrilinosWrappers::SparsityPattern sparsity_pattern;
    TrilinosWrappers::SparseMatrix    system_matrix;
    TrilinosWrappers::MPI::Vector     solution;
    TrilinosWrappers::MPI::Vector     system_rhs;
    IndexSet                          locally_owned_dofs;
    IndexSet                          locally_relevant_dofs;

    Timer timer;
    setup_problem(refinements,
                  mesh,
                  fe,
                  dof_handler,
                  constraints,
                  sparsity_pattern,
                  system_matrix,
                  solution,
                  system_rhs,
                  locally_owned_dofs,
                  locally_relevant_dofs,
                  mpi_communicator);
    timer.stop();
    const double setup_time =
      Utilities::MPI::max(timer.wall_time(), mpi_communicator);

    timer.restart();
    assemble_system(dof_handler,
                    fe,
                    constraints,
                    p,
                    balls,
                    system_matrix,
                    system_rhs);
    timer.stop();
    const double assembly_time =
      Utilities::MPI::max(timer.wall_time(), mpi_communicator);

    if (my_rank == 0)
      std::cout << "case_setup " << setup_time << std::endl
                << "assembly " << assembly_time << std::endl;

    for (const auto &preconditioner : preconditioners)
      solve_with_preconditioner(preconditioner,
                                system_matrix,
                                system_rhs,
                                solution,
                                mpi_communicator);

    if (Utilities::MPI::n_mpi_processes(mpi_communicator) == 1)
      write_vtk(dof_handler, p, balls, solution);
  }

  void
  run_project(int argc, char **argv)
  {
    const MPI_Comm mpi_communicator = MPI_COMM_WORLD;

    const unsigned int refinements =
      argc > 2 ? static_cast<unsigned int>(std::stoi(argv[2])) : 4;

    if (argc > 1)
      {
        const double p = std::stod(argv[1]);
        const std::vector<std::string> preconditioners =
          argc > 3 ? std::vector<std::string>{argv[3]} :
                     std::vector<std::string>{"none", "jacobi", "ssor", "ilu", "amg"};
        run_case(p, refinements, preconditioners, mpi_communicator);
        return;
      }

    (void)refinements;

    for (const unsigned int study_refinement : study_refinements)
      for (const double p : study_p_values)
        run_case(p, study_refinement, study_preconditioners, mpi_communicator);
  }
} // namespace

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  run_project(argc, argv);

  return 0;
}
