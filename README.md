# Solver-FDTD-2D

## Requisitos previos

Primero crear la carpeta `frames`, donde se almacenarán los "frames" generados durante la simulación:

```bash
mkdir frames
```

---

## Carga de módulos

Antes de compilar el código es necesario cargar los módulos correspondientes del clúster:

```bash
module load gcc/8.2.0
module load cuda/11.4.0
```

> **Importante:**  
> Es necesario utilizar `gcc/8.2.0`, ya que el código no compila correctamente con otras versiones de GCC.

---

## Compilación

Compilar el archivo `FDTD_solver.cu` utilizando `nvcc`:

```bash
nvcc -std=c++14 FDTD_solver.cu -o FDTD_solver
```

Esto generará el ejecutable:

```text
FDTD_solver
```

---

## Ejecución

En el archivo `jobGPU` el loop de ejecución está configurado para comenzar y terminar en `i=9`, lo que corresponde a una grilla de:

```text
512 x 512
```

Este código era solo para probar que se ejecute correctamente en el cluster, por lo que utilizar tamaños mayores no resulta necesario en este caso.

Sin embargo, el rango puede modificarse fácilmente para ejecutar simulaciones con grillas más grandes y comparar tiempos de ejecución CPU vs GPU.  
Hay que tener en cuenta que aumentar el tamaño de la grilla implica:

- Mayor tiempo de ejecución.
- Mayor consumo de memoria.
- Mayor tiempo de espera en la cola del clúster.

Por ejemplo:

```bash
for ((i=9;i<=12;i++))
do
    L=$((2**i))
    echo "Ejecutando simulacion con Grilla=$L"
    ./FDTD_solver $L
done
```

El ejemplo anterior ejecutaría simulaciones para los siguientes tamaños de grilla:

| i  | Tamaño de grilla |
|----|------------------|
| 9  | 512 × 512        |
| 10 | 1024 × 1024      |
| 11 | 2048 × 2048      |
| 12 | 4096 × 4096      |

---

## Notas

- La carpeta `frames` debe existir antes de ejecutar el programa para evitar errores.
- Los códigos de Python se adjuntan para mostrar que existen pero ni la animación ni el gráfico
  están pensados para correrse en el cluster, sino a lo sumo de manera local con los resultados generados.
