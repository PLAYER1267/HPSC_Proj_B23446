/*
 * casting_solver.cpp
 * ============================================================
 * 2D Finite Difference Heat Equation Solver for Metal Casting
 * Thermal Stress Analysis — HPSC Assignment
 * ============================================================
 *
 * Physics:
 *   dT/dt = alpha * (d²T/dx² + d²T/dy²)      (heat equation)
 *   epsilon = CTE * (T - T_initial)            (thermal strain)
 *
 * Cell types on the grid:
 *   NORMAL    — standard interior FDM node
 *   BOUNDARY  — fixed temperature (mold wall / ambient)
 *   SOURCE    — fixed high temperature (heat feeder / hot spot)
 *   INSULATOR — zero-flux (thermally blocked) cell
 *
 * ============================================================
 * CHANGES FROM ORIGINAL (4 bugs fixed):
 *
 * FIX 1 — omp_set_num_threads(1) when serial  [THE KEY FIX]
 *   Without this, "serial" runs still use all CPU cores because
 *   OpenMP defaults to the system thread count. Both serial and
 *   parallel ran identically fast → same results.
 *   Location: run_simulation()
 *
 * FIX 2 — Lambda replaced with inline ternaries in update_interior()
 *   The lambda T_nbr() captured [&] and called g.type() + g.T[]
 *   inside the hot parallel loop. This blocked auto-vectorisation
 *   and added indirect-call overhead. Direct ternary expressions
 *   on raw pointers let the compiler emit SIMD instructions.
 *   Location: update_interior()
 *
 * FIX 3 — Grid size increased to 600x600 for scaling study
 *   At 100x100 = 10,000 cells, OpenMP thread-launch overhead
 *   exceeds the work → speedup < 1. Need ~300,000+ cells for
 *   speedup to be visible on real hardware.
 *   Location: run_scaling()
 *
 * FIX 4 — Wall time printed clearly after each thread count
 *   Without this you couldn't see the difference even if it existed.
 *   Location: run_scaling()
 *
 * Build:
 *   OpenMP : g++ -std=c++17 -O3 -fopenmp -o casting casting_solver.cpp
 *   Serial : g++ -std=c++17 -O3           -o casting_serial casting_solver.cpp
 *
 * Usage (Windows PowerShell):
 *   g++ -std=c++17 -O3 -fopenmp -o casting.exe casting_solver.cpp
 *   .\casting.exe scaling results
 *   .\casting.exe all results
 *
 * Usage (Linux/macOS):
 *   make
 *   ./casting scaling results
 *   ./casting all results
 * ============================================================
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;

// ============================================================
//  Cell type enum
// ============================================================
enum class CellType : uint8_t {
    NORMAL    = 0,
    BOUNDARY  = 1,
    SOURCE    = 2,
    INSULATOR = 3
};

// ============================================================
//  Simulation parameters
// ============================================================
struct Params {
    int    Nx          = 100;
    int    Ny          = 100;
    double Lx          = 0.10;
    double Ly          = 0.10;
    double alpha       = 4.0e-6;
    double CTE         = 12.0e-6;
    double T_initial   = 1500.0;
    double T_ambient   = 300.0;
    double T_source    = 1600.0;
    double dt          = 0.0;
    double total_time  = 60.0;
    int    output_every   = 100;
    int    progress_every = 0;
    bool   use_openmp  = false;
    int    n_threads   = 4;
    std::string run_name = "run";
};

// ============================================================
//  RAII timer
// ============================================================
class ScopedTimer {
    using Clock = std::chrono::steady_clock;
    double&           bucket_;
    Clock::time_point start_;
public:
    explicit ScopedTimer(double& b) : bucket_(b), start_(Clock::now()) {}
    ~ScopedTimer() {
        bucket_ += std::chrono::duration<double>(
            Clock::now() - start_).count();
    }
};

// ============================================================
//  Profile data
// ============================================================
struct ProfileData {
    double update_interior = 0.0;
    double apply_bc        = 0.0;
    double compute_strain  = 0.0;
    double io              = 0.0;

    double total() const {
        return update_interior + apply_bc + compute_strain + io;
    }
};

// ============================================================
//  Per-step diagnostics
// ============================================================
struct StepDiag {
    double time         = 0.0;
    double T_mean       = 0.0;
    double T_max        = -1e30;
    double T_min        =  1e30;
    double total_energy = 0.0;
    double max_strain   = 0.0;
    double strain_rms   = 0.0;
};

// ============================================================
//  Run summary
// ============================================================
struct RunSummary {
    double      wall_seconds   = 0.0;
    double      simulated_time = 0.0;
    int         total_steps    = 0;
    int         threads_used   = 1;      // NEW: track actual thread count
    StepDiag    last;
    ProfileData profile;
    double      dt_used        = 0.0;
};

// ============================================================
//  Grid class
// ============================================================
class Grid {
public:
    int Nx, Ny;
    std::vector<double>   T;
    std::vector<double>   T_next;
    std::vector<double>   strain;
    std::vector<CellType> cell_type;

    Grid(int nx, int ny, double T_init)
        : Nx(nx), Ny(ny),
          T(nx*ny, T_init),
          T_next(nx*ny, T_init),
          strain(nx*ny, 0.0),
          cell_type(nx*ny, CellType::NORMAL)
    {}

    int      idx(int i, int j) const { return i * Ny + j; }
    double&  t(int i, int j)         { return T[idx(i,j)]; }
    double   t(int i, int j) const   { return T[idx(i,j)]; }
    CellType& type(int i, int j)     { return cell_type[idx(i,j)]; }
    CellType  type(int i, int j) const { return cell_type[idx(i,j)]; }

    void fill_rect(int i0, int j0, int i1, int j1,
                   CellType ct, double fixed_T = 0.0)
    {
        for (int i = i0; i <= i1; ++i)
            for (int j = j0; j <= j1; ++j) {
                type(i,j) = ct;
                if (ct == CellType::BOUNDARY || ct == CellType::SOURCE)
                    t(i,j) = fixed_T;
            }
    }

    void set_outer_boundary(double T_amb) {
        for (int i = 0; i < Nx; ++i) {
            type(i, 0)    = CellType::BOUNDARY; t(i,0)    = T_amb;
            type(i, Ny-1) = CellType::BOUNDARY; t(i,Ny-1) = T_amb;
        }
        for (int j = 0; j < Ny; ++j) {
            type(0,    j) = CellType::BOUNDARY; t(0,j)    = T_amb;
            type(Nx-1, j) = CellType::BOUNDARY; t(Nx-1,j) = T_amb;
        }
    }

    void swap_buffers() { std::swap(T, T_next); }
};

// ============================================================
//  Utilities
// ============================================================
static std::string dstr(double v, int prec = 10) {
    std::ostringstream o; o << std::setprecision(prec) << v; return o.str();
}
static void ensure_dir(const fs::path& p) { fs::create_directories(p); }

// ============================================================
//  I/O
// ============================================================
void write_field(const fs::path& path, const Grid& g,
                 const std::string& field = "T")
{
    std::ofstream f(path);
    f << "i\\j";
    for (int j = 0; j < g.Ny; ++j) f << ',' << j;
    f << '\n';
    for (int i = 0; i < g.Nx; ++i) {
        f << i;
        for (int j = 0; j < g.Ny; ++j) {
            f << ',' << std::setprecision(6)
              << (field == "T" ? g.t(i,j) : g.strain[g.idx(i,j)]);
        }
        f << '\n';
    }
}

void write_diag_header(std::ofstream& f) {
    f << "step,time,T_mean,T_max,T_min,total_energy,max_strain,strain_rms\n";
}

void append_diag(std::ofstream& f, int step, const StepDiag& d) {
    f << step << ',' << std::setprecision(8)
      << d.time        << ',' << d.T_mean      << ',' << d.T_max   << ','
      << d.T_min       << ',' << d.total_energy << ',' << d.max_strain << ','
      << d.strain_rms  << '\n';
}

void write_profile(const fs::path& path, const ProfileData& pr,
                   int threads)
{
    std::ofstream f(path);
    const double tot = pr.total();
    auto pct = [&](double v){ return tot > 0 ? 100.0*v/tot : 0.0; };
    f << "threads,"        << threads           << '\n';
    f << "stage,time_s,percent\n";
    f << "update_interior," << pr.update_interior << ',' << pct(pr.update_interior) << '\n';
    f << "apply_bc,"        << pr.apply_bc        << ',' << pct(pr.apply_bc)        << '\n';
    f << "compute_strain,"  << pr.compute_strain  << ',' << pct(pr.compute_strain)  << '\n';
    f << "io,"              << pr.io              << ',' << pct(pr.io)              << '\n';
}

void write_summary(const fs::path& path, const RunSummary& s,
                   const std::string& label,
                   const std::vector<std::pair<std::string,std::string>>& extras = {})
{
    std::ofstream f(path);
    auto w = [&](const std::string& k, const std::string& v){
        f << std::left << std::setw(28) << k << ": " << v << '\n';
    };
    w("label",           label);
    w("threads_used",    std::to_string(s.threads_used));
    w("wall_seconds",    dstr(s.wall_seconds));
    w("simulated_time",  dstr(s.simulated_time));
    w("total_steps",     std::to_string(s.total_steps));
    w("dt_used",         dstr(s.dt_used));
    w("final_T_mean",    dstr(s.last.T_mean));
    w("final_T_max",     dstr(s.last.T_max));
    w("final_max_strain",dstr(s.last.max_strain));
    w("final_strain_rms",dstr(s.last.strain_rms));
    w("interior_%",
      dstr(100.0 * s.profile.update_interior
           / std::max(1e-12, s.profile.total())));
    for (const auto& [k,v] : extras) w(k,v);
}

void print_progress(int step, int total, const StepDiag& d,
                    int threads)
{
    std::cout << std::fixed << std::setprecision(3)
              << "  [t" << threads << "]"
              << " step " << std::setw(7) << step << "/" << total
              << "  t="     << d.time
              << "  Tmean=" << std::setprecision(1) << d.T_mean
              << "  Tmax="  << d.T_max
              << "  e_rms=" << std::scientific << std::setprecision(3)
              << d.strain_rms << '\n';
}

// ============================================================
//  Diagnostics
// ============================================================
StepDiag compute_diag(const Grid& g, double time,
                      double T_init, double CTE)
{
    StepDiag d;
    d.time = time;
    double sum_T = 0.0, sum_e2 = 0.0;
    double lmax = -1e30, lmin = 1e30;
    int    cnt  = 0;

#ifdef _OPENMP
    #pragma omp parallel for collapse(2) \
        reduction(+:sum_T,sum_e2,cnt)   \
        reduction(max:lmax)              \
        reduction(min:lmin)              \
        schedule(static)
#endif
    for (int i = 0; i < g.Nx; ++i) {
        for (int j = 0; j < g.Ny; ++j) {
            if (g.type(i,j) == CellType::INSULATOR) continue;
            const double Ti = g.t(i,j);
            const double ei = CTE * (Ti - T_init);
            sum_T += Ti;
            sum_e2 += ei * ei;
            lmax = std::max(lmax, Ti);
            lmin = std::min(lmin, Ti);
            ++cnt;
        }
    }
    d.T_max = lmax;
    d.T_min = lmin;
    if (cnt > 0) {
        d.T_mean       = sum_T / cnt;
        d.total_energy = sum_T;
        d.strain_rms   = std::sqrt(sum_e2 / cnt);
    }
    for (int k = 0; k < g.Nx * g.Ny; ++k)
        d.max_strain = std::max(d.max_strain, std::abs(g.strain[k]));
    return d;
}

// ============================================================
//  FIX 2: update_interior — raw-pointer + inline ternaries
//
//  WHY: the original used a lambda T_nbr(ni,nj) captured by [&].
//  Even though the compiler may inline it, capturing a reference
//  to the enclosing scope prevents the auto-vectoriser from
//  treating the j-loop as a simple SIMD loop.
//
//  With raw pointers and direct ternary expressions, the compiler
//  can prove there are no aliasing hazards and emit vectorised
//  (SSE/AVX) code for the inner loop.
// ============================================================
void update_interior(Grid& g, double coeff)
{
    const int Nx = g.Nx, Ny = g.Ny;

    // Raw pointers — no virtual dispatch, no aliasing ambiguity
    const CellType* __restrict__ CT = g.cell_type.data();
    const double*   __restrict__ T  = g.T.data();
          double*   __restrict__ TN = g.T_next.data();

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 1; i < Nx - 1; ++i) {
        // Inner j-loop: no OpenMP here — let the compiler vectorise it.
        // collapse(2) on the outer omp for already distributes rows
        // to threads; the j-loop inside each row is left for SIMD.
        for (int j = 1; j < Ny - 1; ++j) {
            const int k = i * Ny + j;

            // Non-normal cells: copy through unchanged
            if (CT[k] != CellType::NORMAL) {
                TN[k] = T[k];
                continue;
            }

            const double Tc = T[k];

            // FIX 2: inline ternaries instead of lambda.
            // Mirror rule for INSULATOR neighbours: use Tc (centre).
            // Flat-index arithmetic avoids idx() call overhead.
            const double Tip = (CT[k + Ny] == CellType::INSULATOR) ? Tc : T[k + Ny]; // i+1,j
            const double Tim = (CT[k - Ny] == CellType::INSULATOR) ? Tc : T[k - Ny]; // i-1,j
            const double Tjp = (CT[k +  1] == CellType::INSULATOR) ? Tc : T[k +  1]; // i,j+1
            const double Tjm = (CT[k -  1] == CellType::INSULATOR) ? Tc : T[k -  1]; // i,j-1

            TN[k] = Tc + coeff * (Tip + Tim + Tjp + Tjm - 4.0 * Tc);
        }
    }
}

// ============================================================
//  apply_bc — flat single loop (cleaner, same parallel behaviour)
// ============================================================
void apply_bc(Grid& g)
{
    const int N = g.Nx * g.Ny;
    const CellType* CT = g.cell_type.data();
    const double*   T  = g.T.data();
          double*   TN = g.T_next.data();

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int k = 0; k < N; ++k) {
        if (CT[k] == CellType::BOUNDARY || CT[k] == CellType::SOURCE)
            TN[k] = T[k];
    }
}

// ============================================================
//  compute_strain
// ============================================================
void compute_strain(Grid& g, double CTE, double T_init)
{
    const int N = g.Nx * g.Ny;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int k = 0; k < N; ++k) {
        g.strain[k] = (g.cell_type[k] == CellType::INSULATOR)
                    ? 0.0
                    : CTE * (g.T[k] - T_init);
    }
}

// ============================================================
//  MAIN SIMULATION DRIVER
// ============================================================
RunSummary run_simulation(Grid& g,
                           const Params& par,
                           const fs::path& out_dir,
                           bool write_snapshots = true)
{
    ensure_dir(out_dir);

    // ---- FIX 1: explicitly control thread count ----
    // ORIGINAL code only set threads when use_openmp==true.
    // When use_openmp==false, OpenMP still used ALL system cores
    // (e.g. 8 threads) for the "serial" baseline — so serial and
    // parallel timings were identical.
    //
    // The fix: always call omp_set_num_threads().
    //   use_openmp=false  →  omp_set_num_threads(1)   (true serial)
    //   use_openmp=true   →  omp_set_num_threads(N)   (parallel)
    int threads_used = 1;
#ifdef _OPENMP
    if (par.use_openmp && par.n_threads > 0) {
        omp_set_num_threads(par.n_threads);
        threads_used = par.n_threads;
    } else {
        omp_set_num_threads(1);   // <<< THE KEY LINE
        threads_used = 1;
    }
#endif

    const double dx = par.Lx / (par.Nx - 1);
    const double dy = par.Ly / (par.Ny - 1);
    const double dt = (par.dt > 0.0) ? par.dt
                    : 0.20 * std::min(dx*dx, dy*dy) / par.alpha;
    const double coeff = par.alpha * dt / (dx * dx);

    const int steps       = static_cast<int>(std::ceil(par.total_time / dt));
    const int prog_stride = (par.progress_every > 0)
                          ? par.progress_every
                          : std::max(1, steps / 10);

    if (coeff >= 0.25)
        std::cerr << "  WARNING: stability violated! coeff=" << coeff << '\n';

    std::cout << "  [" << par.run_name << "]"
              << "  threads=" << threads_used
              << "  grid=" << par.Nx << "x" << par.Ny
              << "  dt=" << std::scientific << std::setprecision(3) << dt
              << "  steps=" << steps
              << "  coeff=" << std::fixed << std::setprecision(4) << coeff
              << '\n';

    // Config
    {
        std::ofstream cfg(out_dir / "run_config.txt");
        cfg << "run_name="   << par.run_name   << '\n'
            << "threads="    << threads_used   << '\n'
            << "Nx="         << par.Nx         << '\n'
            << "Ny="         << par.Ny         << '\n'
            << "alpha="      << par.alpha      << '\n'
            << "CTE="        << par.CTE        << '\n'
            << "T_initial="  << par.T_initial  << '\n'
            << "T_ambient="  << par.T_ambient  << '\n'
            << "total_time=" << par.total_time << '\n'
            << "dt="         << dt             << '\n'
            << "coeff="      << coeff          << '\n';
    }

    std::ofstream diag_f(out_dir / "diagnostics.csv");
    write_diag_header(diag_f);

    ProfileData profile;
    auto wall_start = std::chrono::steady_clock::now();

    // Initial state
    {
        ScopedTimer T(profile.compute_strain);
        compute_strain(g, par.CTE, par.T_initial);
    }
    StepDiag last = compute_diag(g, 0.0, par.T_initial, par.CTE);
    append_diag(diag_f, 0, last);
    print_progress(0, steps, last, threads_used);

    if (write_snapshots) {
        ScopedTimer T(profile.io);
        write_field(out_dir / "T_step000000.csv",      g, "T");
        write_field(out_dir / "strain_step000000.csv", g, "strain");
    }

    for (int step = 1; step <= steps; ++step) {
        const double t = step * dt;

        { ScopedTimer T(profile.update_interior);
          update_interior(g, coeff); }

        { ScopedTimer T(profile.apply_bc);
          apply_bc(g); }

        g.swap_buffers();

        if (step % par.output_every == 0 || step == steps) {
            { ScopedTimer T(profile.compute_strain);
              compute_strain(g, par.CTE, par.T_initial); }
            last = compute_diag(g, t, par.T_initial, par.CTE);
            append_diag(diag_f, step, last);
        }

        if (step == steps || step % prog_stride == 0)
            print_progress(step, steps, last, threads_used);

        if (write_snapshots &&
            (step % par.output_every == 0 || step == steps))
        {
            ScopedTimer T(profile.io);
            std::ostringstream nm, nm2;
            nm  << "T_step"      << std::setw(6) << std::setfill('0') << step << ".csv";
            nm2 << "strain_step" << std::setw(6) << std::setfill('0') << step << ".csv";
            write_field(out_dir / nm.str(),  g, "T");
            write_field(out_dir / nm2.str(), g, "strain");
        }
    }

    auto wall_end = std::chrono::steady_clock::now();
    write_profile(out_dir / "profile.csv", profile, threads_used);

    RunSummary s;
    s.wall_seconds   = std::chrono::duration<double>(wall_end - wall_start).count();
    s.simulated_time = steps * dt;
    s.total_steps    = steps;
    s.threads_used   = threads_used;
    s.last           = last;
    s.profile        = profile;
    s.dt_used        = dt;
    return s;
}

// ============================================================
//  Grid builders (unchanged from original)
// ============================================================
Grid build_steady_state(const Params& par) {
    Grid g(par.Nx, par.Ny, par.T_initial);
    for (int j = 0; j < par.Ny; ++j) {
        g.type(0, j) = CellType::SOURCE;
        g.t(0, j)    = par.T_source;
        g.type(par.Nx-1, j) = CellType::BOUNDARY;
        g.t(par.Nx-1, j)    = par.T_ambient;
    }
    for (int i = 0; i < par.Nx; ++i) {
        g.type(i, 0)        = CellType::INSULATOR;
        g.type(i, par.Ny-1) = CellType::INSULATOR;
    }
    double T_mean = 0.5*(par.T_source + par.T_ambient);
    for (int i = 1; i < par.Nx-1; ++i)
        for (int j = 1; j < par.Ny-1; ++j)
            g.t(i,j) = T_mean;
    return g;
}

Grid build_cooling_block(const Params& par) {
    Grid g(par.Nx, par.Ny, par.T_ambient);
    g.set_outer_boundary(par.T_ambient);
    for (int i = par.Nx/4; i <= 3*par.Nx/4; ++i)
        for (int j = par.Ny/4; j <= 3*par.Ny/4; ++j)
            g.t(i,j) = par.T_initial;
    return g;
}

Grid build_geometry(const Params& par, int insulator_thickness,
                    bool use_feeder)
{
    Grid g(par.Nx, par.Ny, par.T_initial);
    g.set_outer_boundary(par.T_ambient);
    const int t = std::max(1, insulator_thickness);
    int bj0 = par.Ny * 3 / 10;
    for (int i = par.Nx/6;  i <= 3*par.Nx/6; ++i)
        for (int j = bj0; j < bj0 + t; ++j)
            g.type(i,j) = CellType::INSULATOR;
    int tj0 = par.Ny * 7 / 10;
    for (int i = par.Nx/2; i <= 5*par.Nx/6; ++i)
        for (int j = tj0; j < tj0 + t; ++j)
            g.type(i,j) = CellType::INSULATOR;
    if (use_feeder) {
        int fi = par.Nx / 2;
        for (int j = par.Ny - 4; j < par.Ny; ++j) {
            g.type(fi, j) = CellType::SOURCE;
            g.t(fi, j)    = par.T_source;
        }
    }
    return g;
}

Grid build_scaling_grid(const Params& par) {
    Grid g(par.Nx, par.Ny, par.T_initial);
    g.set_outer_boundary(par.T_ambient);
    return g;
}

// ============================================================
//  Experiments
// ============================================================
void run_steady_state(const fs::path& root) {
    std::cout << "\n>>> steady_state test\n";
    Params par;
    par.Nx = 80; par.Ny = 80;
    par.alpha = 4e-6; par.CTE = 12e-6;
    par.T_initial = 900.0; par.T_ambient = 300.0; par.T_source = 1200.0;
    par.total_time = 30.0; par.output_every = 200;
    par.run_name = "steady_state";

    const fs::path od = root / "steady_state";
    Grid g = build_steady_state(par);
    const RunSummary s = run_simulation(g, par, od);

    {
        std::ofstream f(od / "centre_row_profile.csv");
        f << "i,x,T,T_linear\n";
        const int jmid = par.Ny / 2;
        const double dx = par.Lx / (par.Nx - 1);
        for (int i = 0; i < par.Nx; ++i) {
            const double x = i * dx;
            const double T_lin = par.T_source
                + (par.T_ambient - par.T_source) * (x / par.Lx);
            f << i << ',' << x << ',' << g.t(i, jmid) << ',' << T_lin << '\n';
        }
    }
    write_summary(od / "summary.txt", s, "steady_state",
        {{"T_source",  dstr(par.T_source)},
         {"T_ambient", dstr(par.T_ambient)}});
    std::cout << "[steady_state]  wall=" << std::fixed << std::setprecision(4)
              << s.wall_seconds << "s  Tmean=" << s.last.T_mean << '\n';
}

void run_cooling_curve(const fs::path& root) {
    std::cout << "\n>>> cooling_curve test\n";
    Params par;
    par.Nx = 80; par.Ny = 80;
    par.alpha = 4e-6; par.CTE = 12e-6;
    par.T_initial = 1500.0; par.T_ambient = 300.0;
    par.total_time = 60.0; par.output_every = 100;
    par.run_name = "cooling_curve";

    const fs::path od = root / "cooling_curve";
    Grid g = build_cooling_block(par);
    const RunSummary s = run_simulation(g, par, od);
    write_summary(od / "summary.txt", s, "cooling_curve");
    std::cout << "[cooling_curve]  wall=" << std::fixed << std::setprecision(4)
              << s.wall_seconds << "s  final_Tmean=" << s.last.T_mean << '\n';
}

void run_geometry_experiment(const fs::path& root) {
    std::cout << "\n>>> geometry_experiment\n";
    ensure_dir(root / "geometry");
    std::ofstream tbl(root / "geometry" / "geometry_results.csv");
    tbl << "insulator_thickness,use_feeder,wall_s,"
           "final_Tmean,final_Tmax,max_strain,strain_rms\n";

    Params par;
    par.Nx = 80; par.Ny = 80;
    par.alpha = 4e-6; par.CTE = 12e-6;
    par.T_initial = 1500.0; par.T_ambient = 300.0; par.T_source = 1600.0;
    par.total_time = 30.0; par.output_every = 150;

    for (int thick : {1, 3, 6}) {
        for (bool feeder : {false, true}) {
            std::ostringstream nm;
            nm << "geom_t" << thick << (feeder ? "_feeder" : "_nofeed");
            par.run_name = nm.str();
            std::cout << "  Running " << nm.str() << '\n';
            Grid g = build_geometry(par, thick, feeder);
            const RunSummary s = run_simulation(g, par,
                                     root / "geometry" / nm.str());
            tbl << thick << ',' << (feeder?1:0) << ','
                << s.wall_seconds << ','
                << s.last.T_mean  << ',' << s.last.T_max   << ','
                << s.last.max_strain << ',' << s.last.strain_rms << '\n';
            write_summary(root/"geometry"/nm.str()/"summary.txt",
                          s, nm.str());
        }
    }
    std::cout << "[geometry_experiment] done\n";
}

// ============================================================
//  Scaling study — FIX 3 + FIX 4
// ============================================================
void run_scaling(const fs::path& root) {
    std::cout << "\n>>> scaling study\n";
    ensure_dir(root / "scaling");

    // ---- Strong scaling ----
    {
        std::ofstream tbl(root / "scaling" / "strong_scaling.csv");
        tbl << "threads,Nx,Ny,steps,wall_s,speedup,efficiency,interior_pct\n";
        double baseline = 0.0;

        for (int thr : {1, 2, 4, 8}) {
            Params par;
            // FIX 3: 600x600 instead of 200x200
            // At 200x200=40k cells, thread overhead ≈ work → no speedup visible.
            // At 600x600=360k cells, speedup is clearly visible on real hardware.
            par.Nx = 600; par.Ny = 600;
            par.alpha = 4e-6; par.CTE = 12e-6;
            par.T_initial = 1500.0; par.T_ambient = 300.0;
            par.total_time = 2.0;
            par.output_every = 999999;   // no snapshots during timing
            par.use_openmp = (thr > 1);  // FIX 1: serial uses 1 thread
            par.n_threads  = thr;
            par.run_name   = "strong_t" + std::to_string(thr);

            // FIX 4: print what we're about to do
            std::cout << "  strong scaling threads=" << thr
                      << "  grid=" << par.Nx << "x" << par.Ny << '\n';

            Grid g = build_scaling_grid(par);
            const RunSummary s = run_simulation(g, par,
                root / "scaling" / par.run_name, false);

            if (thr == 1) baseline = s.wall_seconds;
            const double speedup    = baseline / s.wall_seconds;
            const double efficiency = speedup / thr;
            const double int_pct    = 100.0 * s.profile.update_interior
                / std::max(1e-12, s.profile.total());

            // FIX 4: print wall time clearly so differences are visible
            std::cout << "    --> wall=" << std::fixed << std::setprecision(4)
                      << s.wall_seconds << "s"
                      << "  speedup=" << std::setprecision(3) << speedup
                      << "x"
                      << "  efficiency=" << std::setprecision(1)
                      << efficiency * 100.0 << "%\n";

            tbl << thr << ',' << par.Nx << ',' << par.Ny << ','
                << s.total_steps << ',' << s.wall_seconds << ','
                << speedup << ',' << efficiency << ',' << int_pct << '\n';

            write_summary(root / "scaling" / par.run_name / "summary.txt",
                          s, par.run_name,
                          {{"speedup",    dstr(speedup)},
                           {"efficiency", dstr(efficiency)}});
        }
    }

    // ---- Grid-size scaling (serial) ----
    {
        std::ofstream tbl(root / "scaling" / "grid_scaling.csv");
        tbl << "Nx,Ny,steps,wall_s,interior_pct\n";

        for (int N : {50, 100, 200, 400, 600}) {
            Params par;
            par.Nx = N; par.Ny = N;
            par.alpha = 4e-6; par.CTE = 12e-6;
            par.T_initial = 1500.0; par.T_ambient = 300.0;
            par.total_time = 2.0;
            par.output_every = 999999;
            par.use_openmp = false;
            par.n_threads  = 1;
            par.run_name   = "grid_N" + std::to_string(N);

            std::cout << "  grid scaling N=" << N << '\n';
            Grid g = build_scaling_grid(par);
            const RunSummary s = run_simulation(g, par,
                root / "scaling" / par.run_name, false);
            const double int_pct = 100.0 * s.profile.update_interior
                / std::max(1e-12, s.profile.total());

            std::cout << "    --> wall=" << std::fixed << std::setprecision(4)
                      << s.wall_seconds << "s"
                      << "  steps=" << s.total_steps
                      << "  interior=" << std::setprecision(1)
                      << int_pct << "%\n";

            tbl << N << ',' << N << ',' << s.total_steps << ','
                << s.wall_seconds << ',' << int_pct << '\n';
        }
    }

    std::cout << "[scaling] done\n";
}

// ============================================================
//  CLI + main
// ============================================================
void print_usage() {
    std::cout <<
        "Usage: casting <mode> [output_dir]\n\n"
        "Modes:\n"
        "  steady_state   Verification: linear gradient\n"
        "  cooling_curve  Verification: energy decay\n"
        "  geometry       Insulator sweep experiment\n"
        "  scaling        Strong scaling + grid-size study\n"
        "  all            Run everything\n\n"
        "Windows build:\n"
        "  g++ -std=c++17 -O3 -fopenmp -o casting.exe casting_solver.cpp\n"
        "  .\\casting.exe scaling results\n\n"
        "Linux/macOS build:\n"
        "  g++ -std=c++17 -O3 -fopenmp -o casting casting_solver.cpp\n"
        "  ./casting scaling results\n";
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) { print_usage(); return 1; }
        const std::string mode = argv[1];
        const fs::path root = (argc >= 3)
            ? fs::path(argv[2]) : fs::path("results");

        if      (mode == "steady_state")  run_steady_state(root);
        else if (mode == "cooling_curve") run_cooling_curve(root);
        else if (mode == "geometry")      run_geometry_experiment(root);
        else if (mode == "scaling")       run_scaling(root);
        else if (mode == "all") {
            run_steady_state(root);
            run_cooling_curve(root);
            run_geometry_experiment(root);
            run_scaling(root);
        } else { print_usage(); return 1; }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 2;
    }
}