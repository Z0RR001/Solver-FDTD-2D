# Solver-FDTD-2D


- Primero crear carpeta "frames"
  

- Cargar modulos:
  module load gcc/8.2.0   //es necesario gcc 8.2 porque con otros no compila
  module load cuda/11.4.0
  

- Luego compilar el codigo para crear el .cu con:

  nvcc -std=c++14 FDTD_solver.cu -o FDTD_solver
  

- en jobGPU el loop de ejecución está seteado para que empiece y termine en i=9
  que corresponde a una grilla de 512x512, al cambiar el límite superior del loop
  se pueden simular grillas de otro tamaño. Hacer esto supondría un mayor tiempo de espera
  en la cola del cluste y además en este caso no tiene sentido hacerlo
  porque simplemente es una opción que se agrego para comparar el rendimiento
  de GPU vs CPU. De todas maneras se puede cambiar perfectamente, por ejemplo:

for ((i=9;i<=12;i++))
do
    L=$((2**i))
    echo "Ejecutando simulacion con Grilla=$L"
    ./FDTD_solver $L
done

Esto simularía grillas entre 512, ..., 4096.
