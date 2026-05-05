# ============================================================
#  Makefile — 2D Thermal Stress FDM Solver (Windows-compatible)
# ============================================================

CXX      = g++
STD      = -std=c++17
WARN     = -Wall -Wextra
OPT      = -O3 -march=native -ffast-math
SRC      = casting_solver.cpp

TARGET_SERIAL = casting_serial.exe
TARGET_OMP    = casting_omp.exe

.PHONY: all serial openmp clean run_all run_verify run_scaling plots

all: serial openmp

serial:
	$(CXX) $(STD) $(WARN) $(OPT) -o $(TARGET_SERIAL) $(SRC)
	@echo [OK] Built $(TARGET_SERIAL)

openmp:
	$(CXX) $(STD) $(WARN) $(OPT) -fopenmp -o $(TARGET_OMP) $(SRC)
	@echo [OK] Built $(TARGET_OMP)

run_verify: serial
	$(TARGET_SERIAL) steady_state results
	$(TARGET_SERIAL) cooling_curve results

run_geometry: serial
	$(TARGET_SERIAL) geometry results

run_scaling: openmp
	set OMP_NUM_THREADS=4 && $(TARGET_OMP) scaling results

run_all: serial openmp
	$(TARGET_SERIAL) steady_state results
	$(TARGET_SERIAL) cooling_curve results
	$(TARGET_SERIAL) geometry results
	set OMP_NUM_THREADS=4 && $(TARGET_OMP) scaling results

plots:
	python plot_results.py results

clean:
	if exist $(TARGET_SERIAL) del /f $(TARGET_SERIAL)
	if exist $(TARGET_OMP)    del /f $(TARGET_OMP)
	if exist results          rmdir /s /q results
	if exist figures          rmdir /s /q figures