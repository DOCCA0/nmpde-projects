# Heterogeneous Diffusion Preconditioning Benchmark

This project solves a 3D heterogeneous diffusion problem with deal.II and
Trilinos, compares several `TrilinosWrappers` preconditioners, and generates
CSV data plus SVG plots for the report/slides.

## 1. Load The Environment

Run all commands from the repository root:

```bash
cd /home/wu/code/nmpde-projects
source setup_modules.sh
```

`setup_modules.sh` loads the deal.II module and adds the MPI launcher path used
by this machine.

## 2. Configure And Build

```bash
mkdir -p build
cd build
cmake ..
cmake --build .
cd ..
```

The executable is:

```bash
build/elliptic
```

## 3. Run One Manual Solver Case

The executable accepts:

```bash
build/elliptic <p> <refinement> <preconditioner>
```

Example:

```bash
source setup_modules.sh
build/elliptic 5 5 amg
```

Parallel example:

```bash
source setup_modules.sh
mpirun -np 4 build/elliptic 5 5 amg
```

If no arguments are given, the C++ code runs the full default study defined in
`src/src_elliptic/elliptic_main.cpp`.

## 4. Run The Full Benchmark And Generate CSV/Plots

The Python script builds the code, runs the default C++ study with
`np = 1, 2, 4`, parses the logs, and writes CSV plus SVG plots.

```bash
cd /home/wu/code/nmpde-projects
python3 src/script/benchmark_preconditioners.py
```

Outputs:

```text
result/benchmark_results.csv
result/build.log
result/run_np1.log
result/run_np2.log
result/run_np4.log
result/plots/iterations_vs_heterogeneity_ref5_np1.svg
result/plots/total_time_p5_ref5_np1.svg
result/plots/strong_scaling_p5_ref5.svg
result/plots/setup_solve_p5_ref5_np1.svg
result/heterogeneous-p1.vtk
result/heterogeneous-p3.vtk
result/heterogeneous-p5.vtk
```

The script intentionally does not write `result/report.md`. The report is
written manually from the generated CSV/logs and plots.

## 5. Recommended Full Reproduction Command Sequence

```bash
cd /home/wu/code/nmpde-projects
source setup_modules.sh
mkdir -p build
cd build
cmake ..
cmake --build .
cd ..
python3 src/script/benchmark_preconditioners.py
```

## 6. Notes

- The C++ benchmark parameters are in `src/src_elliptic/elliptic_main.cpp`.
- The compared preconditioners are `none`, `jacobi`, `ssor`, `ilu`, and `amg`.
- The default heterogeneity exponents are `p = 1, 3, 5`.
- The default refinement levels are `3, 4, 5`.
- The default MPI process counts in the Python script are `1, 2, 4`.
