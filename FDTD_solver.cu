#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda_runtime.h>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// ── Parámetros fijos del dominio ─────────────────────────────────────────────
// NX y NY se reciben por argumento en tiempo de ejecución
#define NSTEPS    2000
#define PML_THICK 20
#define BLOCK_X   16
#define BLOCK_Y   16
#define C0        0.5f

// IDX usa ny como variable local — todos los kernels y funciones lo reciben
// como parámetro, por lo que la expansión del macro siempre encuentra ny en scope
#define IDX(i, j) ((i) * ny + (j))

// ── Macro de verificación de errores CUDA ───────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                              \
            std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__          \
                      << " — " << cudaGetErrorString(_err) << "\n";            \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ============================================================
//  KERNEL 1: precomputar coeficientes Ca y Cb por celda
// ============================================================
__global__ void precompute_coeffs(
    const float* __restrict__ eps,
    const float* __restrict__ sigma,
    float*       __restrict__ Ca,
    float*       __restrict__ Cb,
    int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    int   idx   = IDX(i, j);
    float denom = eps[idx] + 0.5f * sigma[idx];

    Ca[idx] = (eps[idx] - 0.5f * sigma[idx]) / denom;
    Cb[idx] = C0 / denom;
}

// ============================================================
//  KERNEL 2: actualizar Hz
//  Hz^{n+½} = Hz^{n-½} + C0 · (∂Ex/∂y − ∂Ey/∂x)
// ============================================================
__global__ void update_Hz(
    float*       __restrict__ Hz,
    const float* __restrict__ Ex,
    const float* __restrict__ Ey,
    int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx - 1 || j >= ny - 1) return;

    int idx = IDX(i, j);
    Hz[idx] += C0 * (
        (Ex[IDX(i, j + 1)] - Ex[idx]) -
        (Ey[IDX(i + 1, j)] - Ey[idx])
    );
}

// ============================================================
//  KERNEL 3: actualizar Ex
//  Ex^{n+1} = Ca · Ex^n + Cb · ∂Hz/∂y
// ============================================================
__global__ void update_Ex(
    float*       __restrict__ Ex,
    const float* __restrict__ Hz,
    const float* __restrict__ Ca,
    const float* __restrict__ Cb,
    int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j < 1 || j >= ny) return;

    int idx = IDX(i, j);
    Ex[idx] = Ca[idx] * Ex[idx]
            + Cb[idx] * (Hz[idx] - Hz[IDX(i, j - 1)]);
}

// ============================================================
//  KERNEL 4: actualizar Ey
//  Ey^{n+1} = Ca · Ey^n − Cb · ∂Hz/∂x
// ============================================================
__global__ void update_Ey(
    float*       __restrict__ Ey,
    const float* __restrict__ Hz,
    const float* __restrict__ Ca,
    const float* __restrict__ Cb,
    int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < 1 || i >= nx || j >= ny) return;

    int idx = IDX(i, j);
    Ey[idx] = Ca[idx] * Ey[idx]
            - Cb[idx] * (Hz[idx] - Hz[IDX(i - 1, j)]);
}

// ============================================================
//  KERNEL 5: inyectar fuente gaussiana (soft source)
// ============================================================
__global__ void inject_source(float* Hz, int t, int si, int sj, int ny)
{
    float pulse = expf(-0.5f * ((t - 40.0f) / 12.0f) * ((t - 40.0f) / 12.0f));
    Hz[IDX(si, sj)] += pulse;
}

// ============================================================
//  KERNEL 6: PML — amortiguación cuadrática en los 4 bordes
// ============================================================
__global__ void apply_pml(
    float* __restrict__ Hz,
    float* __restrict__ Ex,
    float* __restrict__ Ey,
    int nx, int ny, int pml)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    int dist = min(min(i, j), min(nx - 1 - i, ny - 1 - j));
    if (dist >= pml) return;

    float rho    = (float)(pml - 1 - dist) / (float)pml;
    float factor = 1.0f - 0.35f * rho * rho;

    int idx = IDX(i, j);
    Hz[idx] *= factor;
    Ex[idx] *= factor;
    Ey[idx] *= factor;
}

// ============================================================
//  FUNCIONES CPU — equivalentes directos de los kernels GPU
// ============================================================

void cpu_update_Hz(
    std::vector<float>& Hz,
    const std::vector<float>& Ex,
    const std::vector<float>& Ey,
    int nx, int ny)
{
    for (int i = 0; i < nx - 1; i++)
        for (int j = 0; j < ny - 1; j++) {
            int idx = IDX(i, j);
            Hz[idx] += C0 * (
                (Ex[IDX(i, j + 1)] - Ex[idx]) -
                (Ey[IDX(i + 1, j)] - Ey[idx])
            );
        }
}

void cpu_update_Ex(
    std::vector<float>& Ex,
    const std::vector<float>& Hz,
    const std::vector<float>& Ca,
    const std::vector<float>& Cb,
    int nx, int ny)
{
    for (int i = 0; i < nx; i++)
        for (int j = 1; j < ny; j++) {
            int idx = IDX(i, j);
            Ex[idx] = Ca[idx] * Ex[idx]
                    + Cb[idx] * (Hz[idx] - Hz[IDX(i, j - 1)]);
        }
}

void cpu_update_Ey(
    std::vector<float>& Ey,
    const std::vector<float>& Hz,
    const std::vector<float>& Ca,
    const std::vector<float>& Cb,
    int nx, int ny)
{
    for (int i = 1; i < nx; i++)
        for (int j = 0; j < ny; j++) {
            int idx = IDX(i, j);
            Ey[idx] = Ca[idx] * Ey[idx]
                    - Cb[idx] * (Hz[idx] - Hz[IDX(i - 1, j)]);
        }
}

void cpu_apply_pml(
    std::vector<float>& Hz,
    std::vector<float>& Ex,
    std::vector<float>& Ey,
    int nx, int ny, int pml)
{
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++) {
            int dist = std::min({i, j, nx - 1 - i, ny - 1 - j});
            if (dist >= pml) continue;
            float rho    = (float)(pml - 1 - dist) / (float)pml;
            float factor = 1.0f - 0.35f * rho * rho;
            int   idx    = IDX(i, j);
            Hz[idx] *= factor;
            Ex[idx] *= factor;
            Ey[idx] *= factor;
        }
}

// ── Inicializar materiales (CPU) ─────────────────────────────────────────────
void init_materials(std::vector<float>& eps, std::vector<float>& sigma,
                    int nx, int ny)
{
    std::fill(eps.begin(),   eps.end(),   1.0f);
    std::fill(sigma.begin(), sigma.end(), 0.0f);

    for (int i = 280; i < 360; i++)
        for (int j = 180; j < 330; j++)
            if (i < nx && j < ny) eps[IDX(i, j)] = 4.0f;

    for (int i = 140; i < 160; i++)
        for (int j = 100; j < 400; j++)
            if (i < nx && j < ny) sigma[IDX(i, j)] = 5.0f;
}

// ── Guardar snapshot de Hz en formato texto ──────────────────────────────────
void save_frame(const float* data, int t, const std::string& prefix,
                int nx, int ny)
{
    std::ofstream f(prefix + "_t" + std::to_string(t) + ".dat");
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++)
            f << data[IDX(i, j)] << ' ';
        f << '\n';
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    // ── Leer tamaño de grilla desde la línea de comandos ─────────────────────
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <N>\n"
                  << "Ejemplo: " << argv[0] << " 512\n";
        return 1;
    }
    const int nx = std::atoi(argv[1]);
    const int ny = nx;   // grilla cuadrada
    const int N  = nx * ny;

    std::cout << "Grilla: " << nx << " x " << ny
              << "  Pasos: " << NSTEPS << "\n\n";

    // ── Materiales ───────────────────────────────────────────────────────────
    std::vector<float> h_eps(N), h_sigma(N);
    init_materials(h_eps, h_sigma, nx, ny);

    // ── Precomputar Ca y Cb en CPU ───────────────────────────────────────────
    //    Los mismos arrays se usan en el bucle CPU y se copian a la GPU.
    //    No se incluyen en ninguno de los dos timers: son el mismo costo
    //    para ambas versiones y no forman parte de la simulación iterativa.
    std::vector<float> h_Ca(N), h_Cb(N);
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++) {
            int   idx   = IDX(i, j);
            float denom = h_eps[idx] + 0.5f * h_sigma[idx];
            h_Ca[idx]   = (h_eps[idx] - 0.5f * h_sigma[idx]) / denom;
            h_Cb[idx]   = C0 / denom;
        }

    // ════════════════════════════════════════════════════════════════════════
    //  BUCLE TEMPORAL — CPU
    //
    //  El timer arranca justo antes del primer paso y para justo después
    //  del último. No incluye init_materials ni precompute.
    //  Se ejecuta antes de la GPU para que la inicialización del driver
    //  CUDA no interfiera con el tiempo medido.
    // ════════════════════════════════════════════════════════════════════════
    std::cout << "── CPU ─────────────────────────────────────────────\n";

    std::vector<float> cpu_Hz(N, 0.0f);
    std::vector<float> cpu_Ex(N, 0.0f);
    std::vector<float> cpu_Ey(N, 0.0f);

    const int si = nx / 2, sj = ny / 2;

    // ── inicio del timer CPU ──
    auto cpu_t0 = Clock::now();

    for (int t = 0; t < NSTEPS; t++) {
        cpu_update_Hz(cpu_Hz, cpu_Ex, cpu_Ey, nx, ny);
        cpu_update_Ex(cpu_Ex, cpu_Hz, h_Ca, h_Cb, nx, ny);
        cpu_update_Ey(cpu_Ey, cpu_Hz, h_Ca, h_Cb, nx, ny);

        float pulse = std::exp(
            -0.5f * ((t - 40.0f) / 12.0f) * ((t - 40.0f) / 12.0f));
        cpu_Hz[IDX(si, sj)] += pulse;

        cpu_apply_pml(cpu_Hz, cpu_Ex, cpu_Ey, nx, ny, PML_THICK);
    }

    // ── fin del timer CPU ──
    double ms_cpu = Ms(Clock::now() - cpu_t0).count();

    std::cout << "Tiempo: " << ms_cpu << " ms\n\n";

    // ════════════════════════════════════════════════════════════════════════
    //  BUCLE TEMPORAL — GPU (código original sin modificaciones)
    //
    //  El timer arranca justo antes del primer lanzamiento de kernel y
    //  para después de cudaEventSynchronize del último paso.
    //  Un paso de calentamiento previo descarta el overhead de
    //  inicialización del contexto CUDA (~50 ms fijos, independientes
    //  del tamaño de grilla).
    // ════════════════════════════════════════════════════════════════════════
    std::cout << "── GPU ─────────────────────────────────────────────\n";

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Dispositivo: " << prop.name << "\n";

    thrust::device_vector<float> d_Hz(N, 0.0f);
    thrust::device_vector<float> d_Ex(N, 0.0f);
    thrust::device_vector<float> d_Ey(N, 0.0f);
    thrust::device_vector<float> d_eps(h_eps);
    thrust::device_vector<float> d_sigma(h_sigma);
    thrust::device_vector<float> d_Ca(h_Ca);
    thrust::device_vector<float> d_Cb(h_Cb);

    float* Hz = thrust::raw_pointer_cast(d_Hz.data());
    float* Ex = thrust::raw_pointer_cast(d_Ex.data());
    float* Ey = thrust::raw_pointer_cast(d_Ey.data());
    float* Ca = thrust::raw_pointer_cast(d_Ca.data());
    float* Cb = thrust::raw_pointer_cast(d_Cb.data());

    dim3 threads(BLOCK_X, BLOCK_Y);
    dim3 blocks((nx + BLOCK_X - 1) / BLOCK_X,
                (ny + BLOCK_Y - 1) / BLOCK_Y);

    // Calentamiento: fuerza la inicialización del contexto CUDA antes del timer
    update_Hz<<<blocks, threads>>>(Hz, Ex, Ey, nx, ny);
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::fill(d_Hz.begin(), d_Hz.end(), 0.0f);
    thrust::fill(d_Ex.begin(), d_Ex.end(), 0.0f);
    thrust::fill(d_Ey.begin(), d_Ey.end(), 0.0f);

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    thrust::host_vector<float> h_Hz(N);

    // ── inicio del timer GPU ──
    CUDA_CHECK(cudaEventRecord(ev_start));

    for (int t = 0; t < NSTEPS; t++) {

        update_Hz   <<<blocks, threads>>>(Hz, Ex, Ey, nx, ny);
        update_Ex   <<<blocks, threads>>>(Ex, Hz, Ca, Cb, nx, ny);
        update_Ey   <<<blocks, threads>>>(Ey, Hz, Ca, Cb, nx, ny);
        inject_source<<<1, 1>>>(Hz, t, nx / 2, ny / 2, ny);
        apply_pml   <<<blocks, threads>>>(Hz, Ex, Ey, nx, ny, PML_THICK);

        if (t % 50 == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());
            h_Hz = d_Hz;
            save_frame(thrust::raw_pointer_cast(h_Hz.data()), t, "frames/hz",
                       nx, ny);
            std::cout << "  t = " << t << " / " << NSTEPS << "\r" << std::flush;
        }
    }

    // ── fin del timer GPU ──
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms_gpu = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_gpu, ev_start, ev_stop));

    std::cout << "\nTiempo: " << ms_gpu << " ms\n\n";

    // ── Snapshot final ───────────────────────────────────────────────────────
    h_Hz = d_Hz;
    {
        std::ofstream f("hz_output.dat");
        const float* data = thrust::raw_pointer_cast(h_Hz.data());
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++)
                f << data[IDX(i, j)] << ' ';
            f << '\n';
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  GUARDAR RESULTADOS EN timing_results.csv
    //
    //  Modo append (std::ios::app): si el archivo no existe lo crea con
    //  encabezado; si ya existe agrega una línea al final.
    //  Workflow para comparar grillas:
    //    ./FDTD_solver 256
    //    ./FDTD_solver 512
    //    ./FDTD_solver 1024
    //    → cada ejecución agrega una fila al mismo CSV
    // ════════════════════════════════════════════════════════════════════════
    const std::string csv_path = "timing_results.csv";
    bool file_exists = std::ifstream(csv_path).good();

    std::ofstream csv(csv_path, std::ios::app);
    if (!file_exists)
        csv << "Nx,Ny,Nsteps,cpu_ms,gpu_ms\n";

    csv << nx << ","
        << ny << ","
        << NSTEPS << ","
        << ms_cpu << ","
        << ms_gpu << "\n";
    csv.close();

    // ── Reporte en consola ───────────────────────────────────────────────────
    std::cout << "════════════════════════════════════════════════════\n";
    std::cout << "  RESUMEN\n";
    std::cout << "════════════════════════════════════════════════════\n";
    std::cout << "  Grilla  : " << nx << " x " << ny << "\n";
    std::cout << "  Pasos   : " << NSTEPS << "\n";
    std::cout << "  CPU     : " << ms_cpu << " ms\n";
    std::cout << "  GPU     : " << ms_gpu << " ms\n";
    std::cout << "  Speedup : " << ms_cpu / ms_gpu << "x\n";
    std::cout << "  CSV     : " << csv_path << "\n";
    std::cout << "════════════════════════════════════════════════════\n";

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    return 0;
}
