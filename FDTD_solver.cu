#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda_runtime.h>

// ── Parámetros del dominio ───────────────────────────────────────────────────
#define NX        512       // celdas en x
#define NY        512       // celdas en y
#define NSTEPS    2000      // pasos temporales
#define PML_THICK 20        // grosor de la PML (celdas)
#define BLOCK_X   16        // hilos por bloque en x
#define BLOCK_Y   16        // hilos por bloque en y

// Coeficiente de Courant (2D estable con C0 ≤ 1/√2 ≈ 0.707)
// Con unidades normalizadas: dx=1, dt=0.5 → C0 = dt/dx = 0.5
#define C0 0.5f

// Indexado 2D → 1D row-major (x varía más lento → accesos coalescidos en j)
#define IDX(i, j) ((i) * NY + (j))

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
//
//  Discretización de la ecuación diferencial con pérdidas:
//    dE/dt = (1/ε)·rot(H) − (σ/ε)·E
//
//  Esquema Crank-Nicolson implícito en sigma:
//    Ca = (ε − σ·Δt/2) / (ε + σ·Δt/2)
//    Cb = C0 / (ε + σ·Δt/2)
//
//  Se lanza UNA SOLA VEZ antes del loop temporal.
//  Evita divisiones flotantes en cada paso dentro del loop.
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
    float denom = eps[idx] + 0.5f * sigma[idx];   // (ε + σΔt/2)

    Ca[idx] = (eps[idx] - 0.5f * sigma[idx]) / denom;
    Cb[idx] = C0 / denom;
}

// ============================================================
//  KERNEL 2: actualizar Hz
//
//  Hz^{n+½} = Hz^{n-½} + C0 · (∂Ex/∂y − ∂Ey/∂x)
//
//  Nota: Hz usa μ_r = 1 (sin materiales magnéticos).
//  Si necesitás μ_r ≠ 1, basta agregar un coeficiente
//  análogo a Ca/Cb calculado sobre mu.
// ============================================================
__global__ void update_Hz(
    float*       __restrict__ Hz,
    const float* __restrict__ Ex,
    const float* __restrict__ Ey,
    int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Hz se define en [0, nx-2] × [0, ny-2]
    if (i >= nx - 1 || j >= ny - 1) return;

    int idx = IDX(i, j);
    Hz[idx] += C0 * (
        (Ex[IDX(i, j + 1)] - Ex[idx]) -   // ∂Ex/∂y  (diferencia hacia adelante)
        (Ey[IDX(i + 1, j)] - Ey[idx])     // ∂Ey/∂x
    );
}

// ============================================================
//  KERNEL 3: actualizar Ex
//
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

    // Ex se actualiza en [0, nx-1] × [1, ny-1]
    if (i >= nx || j < 1 || j >= ny) return;

    int idx = IDX(i, j);
    Ex[idx] = Ca[idx] * Ex[idx]
            + Cb[idx] * (Hz[idx] - Hz[IDX(i, j - 1)]);  // ∂Hz/∂y
}

// ============================================================
//  KERNEL 4: actualizar Ey
//
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

    // Ey se actualiza en [1, nx-1] × [0, ny-1]
    if (i < 1 || i >= nx || j >= ny) return;

    int idx = IDX(i, j);
    Ey[idx] = Ca[idx] * Ey[idx]
            - Cb[idx] * (Hz[idx] - Hz[IDX(i - 1, j)]);  // ∂Hz/∂x
}

// ============================================================
//  KERNEL 5: inyectar fuente gaussiana (soft source)
//
//  Se lanza con <<<1,1>>> — solo escribe 1 celda.
//  "Soft source": suma al campo en lugar de sobreescribir,
//  permitiendo que la onda reflejada pase sin distorsión.
// ============================================================
__global__ void inject_source(float* Hz, int t, int si, int sj)
{
    float pulse = expf(-0.5f * ((t - 40.0f) / 12.0f) * ((t - 40.0f) / 12.0f));
    Hz[IDX(si, sj)] += pulse;
}

// ============================================================
//  KERNEL 6: PML — amortiguación cuadrática en los 4 bordes
//
//  En cada celda dentro de la capa PML, se calcula la
//  distancia al borde más cercano y se aplica un factor
//  de decaimiento: factor = 1 − α·ρ², con ρ = (pml−dist)/pml
//
//  ρ = 0 en el límite interior/PML → factor ≈ 1 (sin cambio)
//  ρ = 1 en el borde externo       → factor = 1 − α (máx. absorción)
//
//  α = 0.35 es un valor empírico que evita reflexión desde
//  el límite interior sin introducir inestabilidad.
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
    if (dist >= pml) return;   // fuera de la PML — no hacer nada

    float rho    = (float)(pml - 1 - dist) / (float)pml;
    float factor = 1.0f - 0.35f * rho * rho;

    int idx = IDX(i, j);
    Hz[idx] *= factor;
    Ex[idx] *= factor;
    Ey[idx] *= factor;
}

// ── Inicializar materiales (CPU) ─────────────────────────────────────────────
//
//  Retorna dos vectores: permitividad relativa y conductividad por celda.
//  Se define la geometría física aquí — fácil de modificar.
void init_materials(std::vector<float>& eps, std::vector<float>& sigma, int N)
{
    // Por defecto: aire (ε_r = 1, σ = 0)
    std::fill(eps.begin(),   eps.end(),   1.0f);
    std::fill(sigma.begin(), sigma.end(), 0.0f);

    // Bloque dieléctrico (tipo vidrio, ε_r = 4)
    for (int i = 280; i < 360; i++)
        for (int j = 180; j < 330; j++)
            eps[IDX(i, j)] = 4.0f;

    // Conductor con pérdidas (σ grande → absorbe la onda)
    for (int i = 140; i < 160; i++)
        for (int j = 100; j < 400; j++)
            sigma[IDX(i, j)] = 5.0f;
}

// ── Guardar snapshot de Hz en formato texto ──────────────────────────────────
void save_frame(const float* data, int t, const std::string& prefix)
{
    std::ofstream f(prefix + "_t" + std::to_string(t) + ".dat");
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++)
            f << data[IDX(i, j)] << ' ';
        f << '\n';
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main()
{
    const int N = NX * NY;

    // ── Materiales en CPU ────────────────────────────────────────────────────
    std::vector<float> h_eps(N), h_sigma(N);
    init_materials(h_eps, h_sigma, N);

    // ── Memoria GPU con Thrust ───────────────────────────────────────────────
    //
    //  thrust::device_vector<float>(N, 0.0f) reemplaza:
    //    cudaMalloc + cudaMemset + cudaFree (al salir del scope)
    //
    //  La copia desde host_vector a device_vector (d_eps, d_sigma)
    //  reemplaza el cudaMemcpy(HostToDevice).
    thrust::device_vector<float> d_Hz(N, 0.0f);
    thrust::device_vector<float> d_Ex(N, 0.0f);
    thrust::device_vector<float> d_Ey(N, 0.0f);
    thrust::device_vector<float> d_eps(h_eps);     // copia automática host→device
    thrust::device_vector<float> d_sigma(h_sigma);
    thrust::device_vector<float> d_Ca(N);
    thrust::device_vector<float> d_Cb(N);

    // raw_pointer_cast: obtener float* para pasarlos a los kernels CUDA
    float* Hz    = thrust::raw_pointer_cast(d_Hz.data());
    float* Ex    = thrust::raw_pointer_cast(d_Ex.data());
    float* Ey    = thrust::raw_pointer_cast(d_Ey.data());
    float* Ca    = thrust::raw_pointer_cast(d_Ca.data());
    float* Cb    = thrust::raw_pointer_cast(d_Cb.data());
    float* eps   = thrust::raw_pointer_cast(d_eps.data());
    float* sigma = thrust::raw_pointer_cast(d_sigma.data());

    // ── Configuración de lanzamiento ─────────────────────────────────────────
    dim3 threads(BLOCK_X, BLOCK_Y);
    // Techo de división: evita perder celdas si NX/NY no es múltiplo de BLOCK
    dim3 blocks((NX + BLOCK_X - 1) / BLOCK_X,
                (NY + BLOCK_Y - 1) / BLOCK_Y);

    // ── Precomputar coeficientes (una sola vez) ───────────────────────────────
    precompute_coeffs<<<blocks, threads>>>(eps, sigma, Ca, Cb, NX, NY);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // Buffer CPU para copias de diagnóstico
    thrust::host_vector<float> h_Hz(N);

    std::cout << "FDTD 2D — " << NX << "×" << NY
              << " — " << NSTEPS << " pasos\n";

    // ── Bucle temporal ────────────────────────────────────────────────────────
    for (int t = 0; t < NSTEPS; t++) {

        // Leapfrog: Hz primero (½ paso), luego E (paso completo)
        update_Hz   <<<blocks, threads>>>(Hz, Ex, Ey, NX, NY);
        update_Ex   <<<blocks, threads>>>(Ex, Hz, Ca, Cb, NX, NY);
        update_Ey   <<<blocks, threads>>>(Ey, Hz, Ca, Cb, NX, NY);

        // Fuente gaussiana en el centro del dominio
        inject_source<<<1, 1>>>(Hz, t, NX / 2, NY / 2);

        // PML al final de cada paso (atenúa campos en los bordes)
        apply_pml   <<<blocks, threads>>>(Hz, Ex, Ey, NX, NY, PML_THICK);

        // Diagnóstico cada 50 pasos: sincronizar, verificar y guardar frame
        if (t % 50 == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());

            // Copia device→host usando Thrust (reemplaza cudaMemcpy)
            h_Hz = d_Hz;
            save_frame(thrust::raw_pointer_cast(h_Hz.data()), t, "frames/hz");

            std::cout << "  t = " << t << " / " << NSTEPS << "\r" << std::flush;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Guardar snapshot final (equivalente a hz_output.dat del código original)
    h_Hz = d_Hz;
    {
        std::ofstream f("hz_output.dat");
        const float* data = thrust::raw_pointer_cast(h_Hz.data());
        for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY; j++)
                f << data[IDX(i, j)] << ' ';
            f << '\n';
        }
    }

    std::cout << "\nSimulación finalizada. Resultado final en hz_output.dat\n";

    // d_Hz, d_Ex, d_Ey, d_Ca, d_Cb, d_eps, d_sigma
    // se liberan automáticamente al salir del scope (RAII de Thrust)
    return 0;
}
