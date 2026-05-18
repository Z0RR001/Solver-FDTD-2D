# Solver-FDTD-2D


- Primero crear carpeta "frames"

- Luego compilar el codigo para crear el .cu

- en jobGPU el loop de ejecución está seteado para que empiece y termine en i=9
  que corresponde a una grilla de 512x512, al cambiar el límite superior del loop
  se pueden simular grillas de otro tamaño. Hacer esto supondría un mayor tiempo de espera
  en la cola del cluste y además en este caso no tiene sentido hacerlo
  porque simplemente es una opción que se agrego para comparar el rendimiento
  de GPU vs CPU. De todas maneras se puede cambiar perfectamente
