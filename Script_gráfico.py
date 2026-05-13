import pandas as pd
import matplotlib.pyplot as plt

# Cargar los resultados de timing_results.csv
timing_df = pd.read_csv('timing_results.csv')

# Asegurarse de que los datos estén ordenados por el tamaño de la grilla (Nx)
timing_df = timing_df.sort_values(by='Nx')

plt.figure(figsize=(10, 6))

# Graficar la curva de CPU
plt.plot(timing_df['Nx'], timing_df['cpu_ms'], label='CPU (ms)', marker='o')

# Graficar la curva de GPU
plt.plot(timing_df['Nx'], timing_df['gpu_ms'], label='GPU (ms)', marker='o')

plt.title('CPU vs GPU Performance')
plt.xlabel('Tamaño de Grilla')
plt.ylabel('Tiempo (ms)')
plt.xscale('log', base=2) # El tamaño de la grilla suele ser potencias de 2
plt.yscale('log') # El tiempo puede variar significativamente, escala logarítmica para mejor visualización
plt.xticks(timing_df['Nx'], labels=timing_df['Nx'].astype(str)) # Mostrar etiquetas de Nx
plt.grid(True, which="both", ls="--", c='0.7')
plt.legend()
plt.tight_layout()
plt.show()
