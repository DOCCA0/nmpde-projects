## Environment Setup

This project requires the `deal.II` library and a specific `gcc` toolchain, which are managed by an environment module system. Before building the project, you must load these modules into your current shell session.

The provided `setup_modules.sh` script handles this for you. It must be `source`d for the changes to take effect in your current shell.

## Full Reproduction Command Sequence

```bash
# 1. Load required modules into the current shell
source setup_modules.sh

# 2. Configure and build the C++ executable
mkdir -p build
cd build
cmake ..
cmake --build .
cd ..

# 3. Run the Python benchmarking script
python3 src/script/benchmark_preconditioners.py
```
