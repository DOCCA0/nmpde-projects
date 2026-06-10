## Full Reproduction Command Sequence

```bash
source setup_modules.sh
mkdir -p build
cd build
cmake ..
cmake --build .
cd ..
python3 src/script/benchmark_preconditioners.py
```

