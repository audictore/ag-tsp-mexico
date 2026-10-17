"""
Actividad 04 - Algoritmos Evolutivos
Algoritmo genetico para el Problema del Agente Viajero (TSP) sobre las
32 capitales estatales de Mexico.

Se comparan TRES variantes de TRES etapas del algoritmo genetico:
    * Seleccion : ruleta, torneo (k=3), ranking lineal
    * Cruce     : OX (orden), PMX (parcialmente mapeado), CX (ciclico)
    * Mutacion  : intercambio (swap), inversion, mezcla (scramble)

Diseno experimental factorial completo 3 x 3 x 3 = 27 configuraciones,
cada una ejecutada con varias semillas. Todas las configuraciones que comparten
semilla parten de la MISMA poblacion inicial (numeros aleatorios comunes), de
modo que las diferencias se deben al operador y no a la suerte inicial.

Uso:
    python ag_tsp_mexico.py                       # experimento completo
    python ag_tsp_mexico.py --semillas 3 --generaciones 200   # prueba rapida

Salidas (carpeta ../resultados por defecto):
    resultados_corridas.csv, convergencia.npz, resumen.json y figuras PNG.
"""

import argparse
import itertools
import json
import os
import time
from multiprocessing import Pool

import numpy as np

# ---------------------------------------------------------------------------
# 1. Datos del problema
# ---------------------------------------------------------------------------
# (ciudad, estado, latitud, longitud) - coordenadas aproximadas en grados
CAPITALES = [
    ("Aguascalientes", "Ags.", 21.88, -102.29),
    ("Mexicali", "B.C.", 32.62, -115.45),
    ("La Paz", "B.C.S.", 24.14, -110.31),
    ("Campeche", "Camp.", 19.85, -90.53),
    ("Saltillo", "Coah.", 25.42, -101.00),
    ("Colima", "Col.", 19.24, -103.72),
    ("Tuxtla Gutierrez", "Chis.", 16.75, -93.12),
    ("Chihuahua", "Chih.", 28.63, -106.07),
    ("Ciudad de Mexico", "CDMX", 19.43, -99.13),
    ("Durango", "Dgo.", 24.03, -104.65),
    ("Guanajuato", "Gto.", 21.02, -101.26),
    ("Chilpancingo", "Gro.", 17.55, -99.50),
    ("Pachuca", "Hgo.", 20.10, -98.76),
    ("Guadalajara", "Jal.", 20.67, -103.35),
    ("Toluca", "Mex.", 19.29, -99.66),
    ("Morelia", "Mich.", 19.70, -101.19),
    ("Cuernavaca", "Mor.", 18.92, -99.23),
    ("Tepic", "Nay.", 21.50, -104.89),
    ("Monterrey", "N.L.", 25.67, -100.31),
    ("Oaxaca", "Oax.", 17.07, -96.73),
    ("Puebla", "Pue.", 19.04, -98.21),
    ("Queretaro", "Qro.", 20.59, -100.39),
    ("Chetumal", "Q.Roo", 18.50, -88.30),
    ("San Luis Potosi", "S.L.P.", 22.15, -100.98),
    ("Culiacan", "Sin.", 24.80, -107.39),
    ("Hermosillo", "Son.", 29.07, -110.96),
    ("Villahermosa", "Tab.", 17.99, -92.93),
    ("Ciudad Victoria", "Tamps.", 23.74, -99.15),
    ("Tlaxcala", "Tlax.", 19.32, -98.24),
    ("Xalapa", "Ver.", 19.54, -96.91),
    ("Merida", "Yuc.", 20.97, -89.62),
    ("Zacatecas", "Zac.", 22.77, -102.58),
]
N_CIUDADES = len(CAPITALES)


def matriz_distancias():
    """Distancia ortodromica (formula de Haversine) en km entre cada par de ciudades."""
    lat = np.radians([c[2] for c in CAPITALES])
    lon = np.radians([c[3] for c in CAPITALES])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


D = matriz_distancias()


# ---------------------------------------------------------------------------
# 2. Etapa 3: evaluacion (funcion de aptitud)
# ---------------------------------------------------------------------------
def longitud_rutas(pob):
    """Longitud total (km) de cada ruta cerrada. pob: matriz (n_individuos, n_ciudades)."""
    return D[pob, np.roll(pob, -1, axis=1)].sum(axis=1)


def aptitud(dist):
    """Aptitud a maximizar: inverso de la distancia."""
    return 1.0 / dist


# ---------------------------------------------------------------------------
# 3. Etapa 4: seleccion (tres variantes)
# ---------------------------------------------------------------------------
def seleccion_ruleta(dist, n, rng):
    """Seleccion proporcional a la aptitud (Holland, 1975)."""
    f = aptitud(dist)
    return rng.choice(len(dist), size=n, p=f / f.sum())


def seleccion_torneo(dist, n, rng, k=3):
    """Torneo determinista de tamano k: gana el de menor distancia."""
    cand = rng.integers(0, len(dist), size=(n, k))
    return cand[np.arange(n), np.argmin(dist[cand], axis=1)]


def seleccion_ranking(dist, n, rng, sp=1.8):
    """Ranking lineal de Baker (1985) con presion selectiva sp en [1, 2]."""
    m = len(dist)
    orden = np.argsort(dist)  # orden[0] = mejor individuo
    r = np.arange(m)
    p = (sp - (2 * sp - 2) * r / (m - 1)) / m
    return orden[rng.choice(m, size=n, p=p)]


SELECCION = {"Ruleta": seleccion_ruleta, "Torneo": seleccion_torneo, "Ranking": seleccion_ranking}


# ---------------------------------------------------------------------------
# 4. Etapa 5: cruce para permutaciones (tres variantes)
# ---------------------------------------------------------------------------
def _cortes(n, rng):
    i, j = sorted(rng.choice(n + 1, size=2, replace=False))
    return i, j


def cruce_ox(p1, p2, rng):
    """Order Crossover (Davis, 1985): conserva un segmento de p1 y el orden relativo de p2."""
    n = len(p1)
    i, j = _cortes(n, rng)
    hijo = [-1] * n
    hijo[i:j] = p1[i:j]
    usados = set(p1[i:j])
    resto = [p2[(j + k) % n] for k in range(n) if p2[(j + k) % n] not in usados]
    for k, v in enumerate(resto):
        hijo[(j + k) % n] = v
    return hijo


def cruce_pmx(p1, p2, rng):
    """Partially Mapped Crossover (Goldberg y Lingle, 1985): conserva posiciones absolutas."""
    n = len(p1)
    i, j = _cortes(n, rng)
    hijo = [-1] * n
    hijo[i:j] = p1[i:j]
    pos2 = {v: k for k, v in enumerate(p2)}
    segmento = set(p1[i:j])
    for k in range(i, j):
        v = p2[k]
        if v in segmento:
            continue
        pos = k
        while i <= pos < j:  # sigue el mapeo hasta salir del segmento
            pos = pos2[p1[pos]]
        hijo[pos] = v
    for k in range(n):
        if hijo[k] == -1:
            hijo[k] = p2[k]
    return hijo


def cruce_cx(p1, p2, rng):
    """Cycle Crossover (Oliver, Smith y Holland, 1987): cada gen conserva la posicion de algun padre."""
    n = len(p1)
    hijo = [-1] * n
    pos1 = {v: k for k, v in enumerate(p1)}
    ciclo = 0
    for inicio in range(n):
        if hijo[inicio] != -1:
            continue
        fuente = p1 if ciclo % 2 == 0 else p2
        k = inicio
        while hijo[k] == -1:
            hijo[k] = fuente[k]
            k = pos1[p2[k]]
        ciclo += 1
    return hijo


CRUCE = {"OX": cruce_ox, "PMX": cruce_pmx, "CX": cruce_cx}


# ---------------------------------------------------------------------------
# 5. Etapa 6: mutacion para permutaciones (tres variantes)
# ---------------------------------------------------------------------------
def mutacion_intercambio(ind, rng):
    """Swap: intercambia dos ciudades."""
    i, j = rng.choice(len(ind), size=2, replace=False)
    ind[i], ind[j] = ind[j], ind[i]


def mutacion_inversion(ind, rng):
    """Inversion: invierte un segmento (equivale a un movimiento 2-opt)."""
    i, j = _cortes(len(ind), rng)
    ind[i:j] = ind[i:j][::-1]


def mutacion_mezcla(ind, rng):
    """Scramble: baraja aleatoriamente un segmento."""
    i, j = _cortes(len(ind), rng)
    seg = ind[i:j]
    rng.shuffle(seg)
    ind[i:j] = seg


MUTACION = {"Intercambio": mutacion_intercambio, "Inversion": mutacion_inversion, "Mezcla": mutacion_mezcla}


# ---------------------------------------------------------------------------
# 6. Algoritmo genetico (las 7 etapas)
# ---------------------------------------------------------------------------
def poblacion_inicial(n_pob, semilla):
    """Etapa 2: poblacion inicial de permutaciones aleatorias (identica para todas las configuraciones)."""
    rng = np.random.default_rng(10_000 + semilla)
    return np.array([rng.permutation(N_CIUDADES) for _ in range(n_pob)])


def algoritmo_genetico(sel, cru, mut, semilla, n_pob=100, generaciones=500,
                       pc=0.9, pm=0.2, n_elite=2):
    # Etapa 1: representacion por permutacion (camino); el gen k es la k-esima ciudad visitada.
    # Etapa 2: inicializacion
    pob = poblacion_inicial(n_pob, semilla)
    rng = np.random.default_rng([semilla, list(SELECCION).index(sel),
                                 list(CRUCE).index(cru), list(MUTACION).index(mut)])
    f_sel, f_cru, f_mut = SELECCION[sel], CRUCE[cru], MUTACION[mut]

    # Etapa 3: evaluacion
    dist = longitud_rutas(pob)
    mejor_hist = np.empty(generaciones + 1)
    media_hist = np.empty(generaciones + 1)
    divers_hist = np.empty(generaciones + 1)
    mejor_ind, mejor_dist = pob[np.argmin(dist)].copy(), dist.min()
    mejor_hist[0], media_hist[0] = mejor_dist, dist.mean()
    divers_hist[0] = len({tuple(r) for r in pob}) / n_pob

    for g in range(1, generaciones + 1):
        # Etapa 4: seleccion de padres
        padres = pob[f_sel(dist, n_pob - n_elite, rng)].tolist()
        hijos = []
        for k in range(0, len(padres) - 1, 2):
            p1, p2 = padres[k], padres[k + 1]
            # Etapa 5: cruce
            if rng.random() < pc:
                h1, h2 = f_cru(p1, p2, rng), f_cru(p2, p1, rng)
            else:
                h1, h2 = p1[:], p2[:]
            hijos.extend([h1, h2])
        if len(padres) % 2:
            hijos.append(padres[-1][:])
        # Etapa 6: mutacion
        for h in hijos:
            if rng.random() < pm:
                f_mut(h, rng)
        # Etapa 7: reemplazo generacional con elitismo
        elite = pob[np.argsort(dist)[:n_elite]]
        pob = np.vstack([elite, np.array(hijos)])
        dist = longitud_rutas(pob)

        i_min = np.argmin(dist)
        if dist[i_min] < mejor_dist:
            mejor_dist, mejor_ind = dist[i_min], pob[i_min].copy()
        mejor_hist[g], media_hist[g] = mejor_dist, dist.mean()
        divers_hist[g] = len({tuple(r) for r in pob}) / n_pob
    # Etapa 7 (criterio de paro): numero fijo de generaciones

    return mejor_dist, mejor_ind, mejor_hist, media_hist, divers_hist


# ---------------------------------------------------------------------------
# 7. Solucion de referencia (2-opt multiarranque) para medir la brecha
# ---------------------------------------------------------------------------
def dos_opt(ruta):
    ruta = list(ruta)
    n = len(ruta)
    mejora = True
    while mejora:
        mejora = False
        for i in range(n - 1):
            for j in range(i + 2, n if i > 0 else n - 1):
                a, b, c, d = ruta[i], ruta[i + 1], ruta[j], ruta[(j + 1) % n]
                if D[a, c] + D[b, d] < D[a, b] + D[c, d] - 1e-9:
                    ruta[i + 1:j + 1] = ruta[i + 1:j + 1][::-1]
                    mejora = True
    return ruta


def or_opt(ruta):
    """Reubica segmentos de 1 a 3 ciudades consecutivas en otra posicion si acorta la ruta."""
    ruta = list(ruta)
    n = len(ruta)
    mejora = True
    while mejora:
        mejora = False
        for L in (1, 2, 3):
            for i in range(n - L + 1):
                seg = ruta[i:i + L]
                prev, nxt = ruta[i - 1], ruta[(i + L) % n]
                ganancia_quitar = D[prev, seg[0]] + D[seg[-1], nxt] - D[prev, nxt]
                resto = ruta[:i] + ruta[i + L:]
                for k in range(len(resto)):
                    a, b = resto[k], resto[(k + 1) % len(resto)]
                    for s in (seg, seg[::-1]):
                        costo = D[a, s[0]] + D[s[-1], b] - D[a, b]
                        if costo < ganancia_quitar - 1e-9:
                            ruta = resto[:k + 1] + s + resto[k + 1:]
                            mejora = True
                            break
                    if mejora:
                        break
                if mejora:
                    break
            if mejora:
                break
    return ruta


def busqueda_local(ruta):
    """Alterna 2-opt y Or-opt hasta que ninguno mejore."""
    d_ant = np.inf
    while True:
        ruta = or_opt(dos_opt(ruta))
        d = longitud_rutas(np.array([ruta]))[0]
        if d >= d_ant - 1e-9:
            return ruta
        d_ant = d


def solucion_referencia(arranques=300):
    """Mejor optimo local 2-opt + Or-opt de varios arranques aleatorios."""
    rng = np.random.default_rng(2024)
    mejor, mejor_d = None, np.inf
    for _ in range(arranques):
        r = busqueda_local([int(x) for x in rng.permutation(N_CIUDADES)])
        d = longitud_rutas(np.array([r]))[0]
        if d < mejor_d:
            mejor, mejor_d = r, d
    return mejor_d, mejor


# ---------------------------------------------------------------------------
# 8. Experimento
# ---------------------------------------------------------------------------
def _corrida(args):
    sel, cru, mut, semilla, n_pob, gens = args
    t0 = time.perf_counter()
    d, ind, mh, medh, divh = algoritmo_genetico(sel, cru, mut, semilla, n_pob, gens)
    return dict(seleccion=sel, cruce=cru, mutacion=mut, semilla=semilla, distancia=d,
                tiempo=time.perf_counter() - t0, ruta=ind.tolist(),
                mejor_hist=mh, media_hist=medh, divers_hist=divh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, default=20)
    ap.add_argument("--generaciones", type=int, default=500)
    ap.add_argument("--poblacion", type=int, default=100)
    ap.add_argument("--procesos", type=int, default=os.cpu_count())
    ap.add_argument("--salida", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resultados"))
    a = ap.parse_args()
    os.makedirs(a.salida, exist_ok=True)

    tareas = [(s, c, m, k, a.poblacion, a.generaciones)
              for s, c, m in itertools.product(SELECCION, CRUCE, MUTACION) for k in range(a.semillas)]
    print(f"Ejecutando {len(tareas)} corridas ({len(tareas) // a.semillas} configuraciones x {a.semillas} semillas)...")
    t0 = time.time()
    with Pool(a.procesos) as pool:
        res = pool.map(_corrida, tareas, chunksize=4)
    print(f"Listo en {time.time() - t0:.1f} s")

    ref_d, ref_ruta = solucion_referencia()
    print(f"Solucion de referencia (2-opt multiarranque): {ref_d:.1f} km")

    import pandas as pd
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.endswith("hist") and k != "ruta"} for r in res])
    df["brecha_pct"] = 100 * (df.distancia - ref_d) / ref_d
    df.to_csv(os.path.join(a.salida, "resultados_corridas.csv"), index=False)
    np.savez_compressed(os.path.join(a.salida, "convergencia.npz"),
                        mejor=np.array([r["mejor_hist"] for r in res]),
                        media=np.array([r["media_hist"] for r in res]),
                        divers=np.array([r["divers_hist"] for r in res]))
    i_best = int(df.distancia.idxmin())
    with open(os.path.join(a.salida, "resumen.json"), "w", encoding="utf-8") as f:
        json.dump(dict(referencia_km=ref_d, ruta_referencia=[int(x) for x in ref_ruta],
                       mejor_ag=df.loc[i_best, ["seleccion", "cruce", "mutacion", "semilla", "distancia"]].to_dict(),
                       ruta_mejor_ag=res[i_best]["ruta"], parametros=vars(a)), f, indent=2, default=float)

    analizar(df, res, ref_d, res[i_best]["ruta"], a.salida)


# ---------------------------------------------------------------------------
# 9. Analisis estadistico y graficas
# ---------------------------------------------------------------------------
def analizar(df, res, ref_d, mejor_ruta, salida):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colores = ["#2b6cb0", "#dd6b20", "#2f855a"]
    factores = {"seleccion": list(SELECCION), "cruce": list(CRUCE), "mutacion": list(MUTACION)}
    titulos = {"seleccion": "Selección", "cruce": "Cruce", "mutacion": "Mutación"}
    reporte = {}

    # --- Efectos principales y pruebas de Friedman / Wilcoxon (diseno pareado por bloque) ---
    for fac, niveles in factores.items():
        otros = [f for f in factores if f != fac] + ["semilla"]
        tabla = df.pivot_table(index=otros, columns=fac, values="distancia")[niveles]
        fr = stats.friedmanchisquare(*[tabla[n] for n in niveles])
        pares = []
        for x, y in itertools.combinations(niveles, 2):
            w = stats.wilcoxon(tabla[x], tabla[y])
            pares.append([x, y, float(w.pvalue)])
        # correccion de Holm
        orden = np.argsort([p[2] for p in pares])
        m = len(pares)
        ajust, prev = [0.0] * m, 0.0
        for r, idx in enumerate(orden):
            prev = max(prev, min(1.0, (m - r) * pares[idx][2]))
            ajust[idx] = prev
        for p, pa in zip(pares, ajust):
            p.append(pa)
        g = df.groupby(fac)
        reporte[fac] = dict(
            resumen={n: dict(media=float(g.distancia.mean()[n]), de=float(g.distancia.std()[n]),
                             mediana=float(g.distancia.median()[n]), minimo=float(g.distancia.min()[n]),
                             brecha_media=float(g.brecha_pct.mean()[n]), tiempo_medio=float(g.tiempo.mean()[n]),
                             exito_5pct=float((g.brecha_pct.apply(lambda s: (s <= 5).mean()))[n]))
                     for n in niveles},
            friedman=dict(estadistico=float(fr.statistic), p=float(fr.pvalue)),
            wilcoxon_holm=pares)

    # --- Tabla de las 27 configuraciones ---
    conf = (df.groupby(["seleccion", "cruce", "mutacion"])
              .agg(media=("distancia", "mean"), de=("distancia", "std"), minimo=("distancia", "min"),
                   brecha=("brecha_pct", "mean"), exito=("brecha_pct", lambda s: (s <= 5).mean()),
                   tiempo=("tiempo", "mean"))
              .reset_index().sort_values("media"))
    conf.to_csv(os.path.join(salida, "tabla_configuraciones.csv"), index=False)
    reporte["configuraciones"] = conf.to_dict(orient="records")
    reporte["referencia_km"] = ref_d

    # --- Generaciones para llegar a 10 % de la referencia, por factor ---
    mejor = np.array([r["mejor_hist"] for r in res])
    reporte["gen_10pct"] = {}
    for fac, niveles in factores.items():
        reporte["gen_10pct"][fac] = {}
        for n in niveles:
            idx = df.index[df[fac] == n]
            alc = mejor[idx] <= 1.10 * ref_d
            gens = [int(np.argmax(f)) if f.any() else None for f in alc]
            ok = [x for x in gens if x is not None]
            reporte["gen_10pct"][fac][n] = dict(mediana=float(np.median(ok)) if ok else None,
                                                proporcion=len(ok) / len(gens))

    with open(os.path.join(salida, "analisis.json"), "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, default=float)

    # --- Figura 1: convergencia por factor ---
    media = np.array([r["media_hist"] for r in res])
    divers = np.array([r["divers_hist"] for r in res])
    fig, ejes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (fac, niveles) in zip(ejes, factores.items()):
        for c, n in zip(colores, niveles):
            idx = df.index[df[fac] == n]
            ax.plot(mejor[idx].mean(axis=0), color=c, lw=2, label=n)
        ax.axhline(ref_d, color="gray", ls="--", lw=1, label="Referencia")
        ax.set_title(titulos[fac]); ax.set_xlabel("Generación")
        ax.legend(frameon=False)
    ejes[0].set_ylabel("Mejor distancia promedio (km)")
    ejes[0].set_ylim(ref_d * 0.95, ref_d * 2.2)
    fig.tight_layout(); fig.savefig(os.path.join(salida, "fig_convergencia.png"), dpi=200); plt.close(fig)

    # --- Figura 2: diagramas de caja de la distancia final por factor ---
    fig, ejes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (fac, niveles) in zip(ejes, factores.items()):
        datos = [df.distancia[df[fac] == n] for n in niveles]
        bp = ax.boxplot(datos, patch_artist=True, widths=0.55)
        ax.set_xticks(range(1, len(niveles) + 1), niveles)
        for p, c in zip(bp["boxes"], colores):
            p.set_facecolor(c); p.set_alpha(0.55)
        for med in bp["medians"]:
            med.set_color("black")
        ax.axhline(ref_d, color="gray", ls="--", lw=1)
        ax.set_title(titulos[fac])
    ejes[0].set_ylabel("Distancia final (km)")
    fig.tight_layout(); fig.savefig(os.path.join(salida, "fig_cajas.png"), dpi=200); plt.close(fig)

    # --- Figura 3: interaccion cruce x mutacion (brecha media %) por seleccion ---
    fig, ejes = plt.subplots(1, 3, figsize=(13, 3.6))
    vmax = conf.brecha.max()
    for ax, s in zip(ejes, factores["seleccion"]):
        m = (conf[conf.seleccion == s].pivot(index="cruce", columns="mutacion", values="brecha")
             .loc[factores["cruce"], factores["mutacion"]])
        im = ax.imshow(m.values, cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_xticks(range(3), m.columns); ax.set_yticks(range(3), m.index)
        for i in range(3):
            for j in range(3):
                v = m.values[i, j]
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                        color="white" if v > vmax * 0.6 else "black", fontsize=9)
        ax.set_title(f"Selección: {s}")
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.colorbar(im, ax=ejes, shrink=0.85, label="Brecha media vs. referencia (%)")
    fig.savefig(os.path.join(salida, "fig_interaccion.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

    # --- Figura 4: diversidad de la poblacion por tipo de seleccion ---
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for c, n in zip(colores, factores["seleccion"]):
        idx = df.index[df.seleccion == n]
        ax.plot(100 * divers[idx].mean(axis=0), color=c, lw=2, label=n)
    ax.set_xlabel("Generación"); ax.set_ylabel("Individuos distintos (%)")
    ax.set_title("Diversidad de la población según el método de selección")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(salida, "fig_diversidad.png"), dpi=200); plt.close(fig)

    # --- Figura 5: mapa de la mejor ruta encontrada ---
    lat = np.array([c[2] for c in CAPITALES]); lon = np.array([c[3] for c in CAPITALES])
    r = list(mejor_ruta) + [mejor_ruta[0]]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    ax.plot(lon[r], lat[r], "-", color="#2b6cb0", lw=1.6, zorder=1)
    ax.scatter(lon, lat, s=28, color="#dd6b20", zorder=2)
    for k, c in enumerate(CAPITALES):
        ax.annotate(c[1], (lon[k], lat[k]), textcoords="offset points", xytext=(4, 3), fontsize=7.5)
    ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud"); ax.set_aspect(1 / np.cos(np.radians(23)))
    ax.set_title(f"Mejor ruta del AG: {longitud_rutas(np.array([mejor_ruta]))[0]:,.0f} km")
    fig.tight_layout(); fig.savefig(os.path.join(salida, "fig_ruta.png"), dpi=200); plt.close(fig)

    print(json.dumps({f: reporte[f]["friedman"] for f in factores}, indent=2))
    print(conf.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
