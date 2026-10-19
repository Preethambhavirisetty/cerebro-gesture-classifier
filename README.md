# 🧠 CEREBRO: Embedded Gesture Classifier

![C++17](https://img.shields.io/badge/C++17-00599C?logo=cplusplus&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-064F8C?logo=cmake&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-005CED?logo=onnx&logoColor=white)


A C++17 static library for real-time gesture classification on sensor data, written under embedded constraints. A Python pipeline trains the model and exports it to ONNX.

## Features
- `cerebro` static library: circular buffer, feature extractor, inference engine, gesture classifier
- Embedded-friendly build: `-fno-exceptions -fno-rtti`, `-Wall -Wextra -Wpedantic`, section GC in Release
- Optional ARM Cortex-M4 cross-compile (`-DTARGET_CORTEX_M4=ON`, needs `arm-none-eabi-g++`)
- ASan/UBSan in Debug builds
- CTest suite: circular buffer, feature extractor, classifier, sensor dropout
- Python trainer (synthetic data generation, feature engineering, PyTorch model, ONNX export) plus accuracy and benchmark validation

## Screenshot
<!-- Add test/benchmark output or an architecture diagram: docs/architecture.png -->

## Tech stack
C++17 · CMake ≥ 3.16 · CTest · Python · PyTorch · ONNX / onnxruntime · scikit-learn · NumPy

## Project structure
```
.
├── include/cerebro/   # circular_buffer, event, feature_extractor, gesture_classifier,
│                      # inference_engine, model_weights (.hpp)
├── src/               # feature_extractor.cpp, gesture_classifier.cpp, inference_engine.cpp
├── tests/             # CTest executables + test_runner.hpp
├── python/
│   ├── trainer/       # data_generator, feature_engineer, model, train, export_onnx
│   ├── validation/    # accuracy_validator, benchmark
│   └── requirements.txt
├── models/            # exported models (.gitkeep)
└── CMakeLists.txt
```

## Getting started
```bash
git clone https://github.com/Preethambhavirisetty/cerebro-gesture-classifier.git
cd cerebro-gesture-classifier

# Build + test the C++ library
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure

# Train the model
cd python
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m trainer.train --epochs 80 --batch-size 128
```
Cross-compile: `cmake -S . -B build-m4 -DTARGET_CORTEX_M4=ON -DCMAKE_CXX_COMPILER=arm-none-eabi-g++`

## Environment variables
None.

## Repo hygiene to-do
- `git rm -r --cached build .DS_Store` and add a `.gitignore` with `build/`, `.DS_Store`, `__pycache__/`, `models/*.onnx`

## Contributing
1. Fork the repo and create a branch: `git checkout -b feature/your-change`
2. Commit using [Conventional Commits](https://www.conventionalcommits.org/) (`feat: …`, `fix: …`)
3. Push and open a pull request against `main`

## License
No license file yet. Add one (MIT is a common choice) via **Add file → Create new file → `LICENSE` → Choose a license template**, then update this section.
