# Algoritmo genético para el TSP de las 32 capitales de México

Actividad 04 – Algoritmos Evolutivos.

Se resuelve el Problema del Agente Viajero (TSP) sobre las 32 capitales estatales de México
(distancias ortodrómicas, fórmula de Haversine) y se comparan tres variantes de tres etapas
del algoritmo genético:

| Etapa     | Variantes comparadas                         |
|-----------|----------------------------------------------|
| Selección | Ruleta, Torneo (k = 3), Ranking lineal (SP = 1.8) |
| Cruce     | OX, PMX, CX                                  |
| Mutación  | Intercambio, Inversión, Mezcla               |

Diseño factorial completo 3 × 3 × 3 = 27 configuraciones × 20 semillas = 540 corridas.

## Ejecución

```bash
pip install -r requirements.txt
python ag_tsp_mexico.py                                   # experimento completo (~1-2 min con 12 núcleos)
python ag_tsp_mexico.py --semillas 3 --generaciones 200   # prueba rápida
```

Los resultados (CSV, JSON y figuras PNG) se guardan en `../resultados`.
