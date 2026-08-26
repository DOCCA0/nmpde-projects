#include "HeterogeneousDiffusion.hpp"

#include <string>
#include <vector>

int
main(int argc, char **argv)
{
  using namespace dealii;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const MPI_Comm mpi_communicator = MPI_COMM_WORLD;

  const std::vector<double>       study_p_values      = {1.0, 2.0, 3.0, 4.0, 5.0};
  const std::vector<unsigned int> study_refinements   = {3, 4, 5};
  const std::vector<std::string>  study_preconditioners = {
    "none", "jacobi", "ssor", "ilu", "amg"};

  if (argc > 1)
    {
      const double       p           = std::stod(argv[1]);
      const unsigned int refinements =
        argc > 2 ? static_cast<unsigned int>(std::stoi(argv[2])) : 4;
      const std::vector<std::string> preconditioners =
        argc > 3 ? std::vector<std::string>{argv[3]} : study_preconditioners;

      HeterogeneousDiffusion problem(refinements, p, mpi_communicator);
      problem.setup_and_assemble();
      for (const auto &prec : preconditioners)
        problem.solve(prec);
      problem.output(preconditioners.back());
      return 0;
    }

  for (const unsigned int refinements : study_refinements)
    for (const double p : study_p_values)
      {
        HeterogeneousDiffusion problem(refinements, p, mpi_communicator);
        problem.setup_and_assemble();
        for (const auto &prec : study_preconditioners)
          problem.solve(prec);
      }

  return 0;
}