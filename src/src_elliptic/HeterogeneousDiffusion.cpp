#include "HeterogeneousDiffusion.hpp"

#include <iostream>
#include <filesystem>

std::vector<HeterogeneousDiffusion::Ball>
HeterogeneousDiffusion::make_balls()
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

HeterogeneousDiffusion::HeterogeneousDiffusion(
  const unsigned int refinements_,
  const double       p_,
  const MPI_Comm     mpi_communicator_,
  const bool         verify_mode_)
  : refinements(refinements_)
  , p(p_)
  , mpi_communicator(mpi_communicator_)
  , mpi_size(Utilities::MPI::n_mpi_processes(mpi_communicator_))
  , mpi_rank(Utilities::MPI::this_mpi_process(mpi_communicator_))
  , verify_mode(verify_mode_)
  , pcout(std::cout, mpi_rank == 0)
  , balls(make_balls())
  , mesh(mpi_communicator_)
  , fe(1)
  , dof_handler(mesh)
{}

void
HeterogeneousDiffusion::setup_and_assemble()
{
  pcout << "mpi_procs "   << mpi_size     << std::endl;
  pcout << "p "           << p            << std::endl;
  pcout << "refinements " << refinements  << std::endl;
  pcout << "balls "       << balls.size() << std::endl;

  Timer timer;

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
  DoFTools::make_sparsity_pattern(
    dof_handler, sparsity_pattern, constraints, false, mpi_rank);
  sparsity_pattern.compress();
  system_matrix.reinit(sparsity_pattern);

  system_rhs.reinit(locally_owned_dofs, mpi_communicator);
  solution.reinit(locally_owned_dofs, mpi_communicator);

  pcout << "cells " << mesh.n_global_active_cells() << std::endl;
  pcout << "dofs "  << dof_handler.n_dofs()         << std::endl;

  timer.stop();
  const double setup_time =
    Utilities::MPI::max(timer.wall_time(), mpi_communicator);

  timer.restart();

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

  DiffusionCoefficient      heterogeneous_diffusion(p, balls);
  RightHandSide             heterogeneous_rhs;
  ManufacturedRightHandSide manufactured_rhs;

  const Function<dim> *rhs_ptr = verify_mode
    ? static_cast<const Function<dim> *>(&manufactured_rhs)
    : static_cast<const Function<dim> *>(&heterogeneous_rhs);

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
          const double mu = verify_mode
            ? 1.0
            : heterogeneous_diffusion.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                cell_matrix(i, j) += mu * fe_values.shape_grad(i, q) *
                                     fe_values.shape_grad(j, q) *
                                     fe_values.JxW(q);

              cell_rhs(i) += rhs_ptr->value(fe_values.quadrature_point(q)) *
                             fe_values.shape_value(i, q) * fe_values.JxW(q);
            }
        }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  timer.stop();
  const double assembly_time =
    Utilities::MPI::max(timer.wall_time(), mpi_communicator);

  pcout << "case_setup " << setup_time    << std::endl;
  pcout << "assembly "   << assembly_time << std::endl;
}

template <class Preconditioner>
unsigned int
HeterogeneousDiffusion::solve_with(const Preconditioner &preconditioner,
                                   const std::string    &name,
                                   const double          setup_time)
{
  SolverControl solver_control(20000, 1e-10 * system_rhs.l2_norm());
  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

  double condition_number = -1.0;
  solver.connect_condition_number_slot(
    [&condition_number](const double cn) { condition_number = cn; });

  Timer timer;
  solver.solve(system_matrix, solution, system_rhs, preconditioner);
  timer.stop();

  const double solve_time =
    Utilities::MPI::max(timer.wall_time(), mpi_communicator);
  const double setup_time_global =
    Utilities::MPI::max(setup_time, mpi_communicator);

  pcout << "result " << name << " " << solver_control.last_step() << " "
        << setup_time_global << " " << solve_time << " "
        << setup_time_global + solve_time << " " << condition_number
        << std::endl;

  return solver_control.last_step();
}

void
HeterogeneousDiffusion::solve(const std::string &name)
{
  solution = 0.0;

  try
    {
      if (name == "none")
        {
          solve_with(PreconditionIdentity(), name, 0.0);
          return;
        }

      Timer timer;

      if (name == "jacobi")
        {
          TrilinosWrappers::PreconditionJacobi preconditioner;
          preconditioner.initialize(system_matrix);
          timer.stop();
          solve_with(preconditioner, name, timer.wall_time());
          return;
        }

      if (name == "ssor")
        {
          TrilinosWrappers::PreconditionSSOR preconditioner;
          preconditioner.initialize(system_matrix);
          timer.stop();
          solve_with(preconditioner, name, timer.wall_time());
          return;
        }

      if (name == "ilu")
        {
          TrilinosWrappers::PreconditionILU preconditioner;
          preconditioner.initialize(system_matrix);
          timer.stop();
          solve_with(preconditioner, name, timer.wall_time());
          return;
        }

      if (name == "amg")
        {
          TrilinosWrappers::PreconditionAMG                 preconditioner;
          TrilinosWrappers::PreconditionAMG::AdditionalData data;
          data.elliptic              = true;
          data.higher_order_elements = true;
          data.n_cycles              = 1;
          preconditioner.initialize(system_matrix, data);
          timer.stop();
          solve_with(preconditioner, name, timer.wall_time());
          return;
        }

      pcout << "failed " << name << " unknown preconditioner" << std::endl;
    }
  catch (const std::exception &exception)
    {
      pcout << "failed " << name << " " << exception.what() << std::endl;
    }
}

double
HeterogeneousDiffusion::compute_error(
  const VectorTools::NormType &norm_type) const
{
  TrilinosWrappers::MPI::Vector solution_ghost(locally_owned_dofs,
                                               locally_relevant_dofs,
                                               mpi_communicator);
  solution_ghost = solution;

  const QGauss<dim>    quadrature_error(fe.degree + 2);
  const MappingQ1<dim> mapping;

  ManufacturedSolution manufactured_solution;

  Vector<double> error_per_cell(mesh.n_active_cells());
  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution_ghost,
                                    manufactured_solution,
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  return VectorTools::compute_global_error(mesh, error_per_cell, norm_type);
}

void
HeterogeneousDiffusion::output(const std::string &tag) const
{
  TrilinosWrappers::MPI::Vector solution_ghost(locally_owned_dofs,
                                               locally_relevant_dofs,
                                               mpi_communicator);

  const std::string folder = "output/";
  if (mpi_rank == 0)
    std::filesystem::create_directories(folder);
  MPI_Barrier(mpi_communicator);
  
  
  solution_ghost = solution;
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution_ghost, "solution");

  
  Vector<double>       mu_field(mesh.n_active_cells());
  DiffusionCoefficient diffusion(p, balls);
  for (const auto &cell : mesh.active_cell_iterators())
    if (cell->is_locally_owned())
      mu_field[cell->active_cell_index()] = diffusion.value(cell->center());
  data_out.add_data_vector(mu_field, "mu");

  data_out.build_patches();

  const std::string basename = "heterogeneous-" + tag + "-p" +
                               std::to_string(int(p)) + "-ref" +
                               std::to_string(refinements);
  data_out.write_vtu_with_pvtu_record(folder, basename, 0, mpi_communicator);
  pcout << "vtk " << folder << basename << ".pvtu" << std::endl;
}