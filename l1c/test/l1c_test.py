# ============================================================================
# L1C TEST SCRIPTS
# ----------------------------------------------------------------------------
# run_grid_plot()
#     Plot 1: contorno de la rejilla L1B (negro) + puntos de la rejilla
#     L1C (rojo). La L1B es irregular; la L1C es regular a 100 m porque
#     se proyecta sobre la rejilla MGRS (zona UTM 31R).
#
# run_gsd_analysis()
#     Plot 2: GSD ACROSS TRACK, distancia Haversine entre columnas ACT
#             consecutivas de la fila central de la L1B -> 149 puntos.
#     Plot 3: GSD ALONG TRACK, distancia Haversine entre filas ALT
#             consecutivas de la columna central de la L1B -> 99 puntos.
#
# run_crossvalidation()
#     Compara los l1c_toa_<banda>.nc con los de referencia. Los puntos L1C
#     salen de un set de teselas MGRS, que NO esta ordenado, asi que antes
#     de comparar se ordenan los dos por (lat, lon). Se puede desactivar
#     con DO_CROSSVALIDATION = False.
#
# NOTA sobre "un plot por banda": la geolocalizacion del GM es UNICA para
# las 4 bandas (una sola matriz lat/lon), asi que los plots de geometria
# (rejillas y GSD) son unicos y no tiene sentido repetirlos por banda. El
# TOA si es por banda, y se grafica aparte con PLOT_TOA_PER_BAND.
#
# Ejecucion (desde la raiz del proyecto EODP_NOEL):
#     python l1c/test/l1c_test.py
# ============================================================================

import io
import math
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no abre ventanas, solo guarda los PNG
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.io.l1cProduct import readL1c              # noqa: E402
from common.io.readGeodetic import readGeodetic, getCorners   # noqa: E402
from config.globalConfig import globalConfig          # noqa: E402

# common/src/auxGeom.py importa pyproj a nivel de modulo (lo necesita para
# ecef2geo / geo2ecef, que aqui no se usan). Si pyproj no esta instalado se
# usa una copia local de la MISMA formula y el MISMO radio que
# auxGeom.haversine, para que el resultado numerico sea identico.
try:
    from common.src.auxGeom import haversine          # noqa: E402
    HAVERSINE_SOURCE = "common/src/auxGeom.py"

except ImportError:

    def haversine(lat1, lon1, lat2, lon2):
        """
        Distancia geodetica sobre el terreno entre dos puntos.

        Copia literal de common/src/auxGeom.haversine, para no depender de
        pyproj. R = 6378137 m, radio equatorial WGS84.

        :param lat1: latitud del primer punto [deg]
        :param lon1: longitud del primer punto [deg]
        :param lat2: latitud del segundo punto [deg]
        :param lon2: longitud del segundo punto [deg]
        :return: distancia entre los dos puntos [m]
        """
        R = 6378137  # [m] WGS84 Equatorial radius

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)

        dphi = np.abs(math.radians(lat2 - lat1))
        dlambda = np.abs(math.radians(lon2 - lon1))

        a = math.sin(dphi / 2) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2

        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))  # [m]

    HAVERSINE_SOURCE = "copia local identica (pyproj no instalado)"


# ============================================================================
# CONFIGURACION  (cambiar estas rutas si las carpetas no son exactamente asi)
# ============================================================================

_BASE = (r"C:\Users\Noel\Documents\NOEL\UNIVERSIDAD\MASTER\TELECO\ACADEMICO"
         r"\26-27\1er cuatri\EODP\FILES\EODP_TER_2021\EODP_TER_2021"
         r"\EODP-TS-L1C")

# Carpeta del GM, con geolocation.nc (la rejilla L1B)
GM_DIR = Path(_BASE) / "input" / "gm_alt100_act_150"

# Mis salidas del L1C
FOLDER_MINE = Path(_BASE) / "output_noel"

# Salidas de referencia del profesor (None para saltar la comparacion)
FOLDER_REFERENCE = Path(_BASE) / "output"

# Carpeta donde se guardan los PNG y el informe.
# None -> se usa FOLDER_MINE
OUTPUT_DIR = None
REPORT_FILENAME = "l1c_test_report.txt"

# Nombres de los PNG
PNG_GRID = "l1c_grid_l1b_vs_l1c.png"
PNG_GSD_ACT = "l1c_gsd_across_track.png"
PNG_GSD_ALT = "l1c_gsd_along_track.png"
PNG_TOA_TEMPLATE = "l1c_toa_{band}.png"

# Dibujar tambien todos los puntos de la rejilla L1B, no solo el contorno
PLOT_ALL_L1B_POINTS = True

# Un scatter del TOA L1C por banda (este si es por banda)
PLOT_TOA_PER_BAND = True

# Cross validation de los l1c_toa_*.nc
DO_CROSSVALIDATION = True

# Tolerancia para emparejar dos puntos L1C por sus coordenadas [deg]
COORD_TOL_DEG = 1e-5

# Requisito: diferencias relativas por debajo de 1e-3 %
REL_DIFF_THRESHOLD_PCT = 1e-3
MIN_PCT_POINTS_WITHIN = 100.0
ABS_FLOOR = 1e-12


# ============================================================================
# UTILIDADES
# ============================================================================

def get_output_dir():
    """
    Devuelve la carpeta donde se guardan los PNG y el informe.

    :return: carpeta de salida
    """
    return Path(OUTPUT_DIR) if OUTPUT_DIR is not None else Path(FOLDER_MINE)


def save_report(report_lines):
    """
    Guarda el informe de texto.

    :param report_lines: lista de lineas del informe
    """
    output_dir = get_output_dir()

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / REPORT_FILENAME

        with open(report_path, "w", encoding="utf-8") as handler:
            handler.write("\n".join(report_lines) + "\n")

        print(f"\nInforme guardado en: {report_path}")

    except OSError as error:
        print(f"\nAVISO: no se pudo guardar el informe: {error}")


def quiet_read_geodetic(folder, filename):
    """
    Lee geolocation.nc silenciando los prints del lector.

    :param folder: carpeta del GM
    :param filename: nombre del fichero
    :return: (lat, lon) en grados, o (None, None) si no se puede leer
    """
    full_path = os.path.join(str(folder), filename)

    if not os.path.isfile(full_path):
        print(f"  AVISO: no existe {full_path}")
        return None, None

    try:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            return readGeodetic(str(folder), filename)
    except (OSError, RuntimeError, KeyError, IndexError) as error:
        print(f"  AVISO: no se pudo leer {full_path} -> {error}")
        return None, None


def quiet_read_l1c(folder, filename):
    """
    Lee un producto L1C silenciando los prints del lector.

    :param folder: carpeta del producto
    :param filename: nombre del fichero con extension .nc
    :return: (toa, lat, lon), o (None, None, None) si no se puede leer
    """
    if folder is None:
        return None, None, None

    full_path = os.path.join(str(folder), filename)

    if not os.path.isfile(full_path):
        return None, None, None

    try:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            return readL1c(str(folder), filename)
    except (OSError, RuntimeError, KeyError, IndexError) as error:
        print(f"  AVISO: no se pudo leer {full_path} -> {error}")
        return None, None, None


def haversine_series(lat_vector, lon_vector):
    """
    Distancia Haversine entre puntos consecutivos de una serie.

    :param lat_vector: latitudes consecutivas [deg]
    :param lon_vector: longitudes consecutivas [deg]
    :return: vector de distancias [m], de longitud n-1
    """
    n = lat_vector.size
    distances = np.zeros(n - 1)

    for index in range(n - 1):
        distances[index] = haversine(
            float(lat_vector[index]), float(lon_vector[index]),
            float(lat_vector[index + 1]), float(lon_vector[index + 1]),
        )

    return distances


def describe_series(distances, label, log):
    """
    Imprime las estadisticas de una serie de distancias.

    Se usan 6 decimales porque la variacion de la GSD across track en una
    escena estrecha es del orden de fracciones de milimetro: con solo 3
    decimales se mostraria como 0.000 m y pareceria exactamente constante.

    :param distances: vector de distancias [m]
    :param label: etiqueta para los mensajes
    :param log: funcion que imprime y acumula
    """
    media = float(np.mean(distances))
    minimo = float(np.min(distances))
    maximo = float(np.max(distances))
    variacion = maximo - minimo

    log(f"  {label}")
    log(f"    numero de puntos : {distances.size}")
    log(f"    media            : {media:.6f} m")
    log(f"    minimo           : {minimo:.6f} m")
    log(f"    maximo           : {maximo:.6f} m")
    log(f"    desviacion tipica: {float(np.std(distances)):.6f} m")
    log(f"    variacion max-min: {variacion:.6f} m  "
        f"({variacion * 1000:.3f} mm)")

    if media > 0:
        log(f"    variacion rel.   : {variacion / media * 100:.6f} %")


# ============================================================================
# PLOT 1 - REJILLA L1B FRENTE A REJILLA L1C
# ============================================================================

def run_grid_plot(log):
    """
    Dibuja el contorno de la rejilla L1B y los puntos de la rejilla L1C.

    :param log: funcion que imprime y acumula una linea del informe
    :return: True si el plot se ha generado
    """
    global_config = globalConfig()

    log("")
    log("=" * 96)
    log("PLOT 1 - REJILLA L1B FRENTE A REJILLA L1C")
    log("=" * 96)

    lat_l1b, lon_l1b = quiet_read_geodetic(GM_DIR, global_config.gm_geoloc)

    if lat_l1b is None:
        log(f"  No se encuentra {global_config.gm_geoloc} en {GM_DIR}.")
        return False

    log(f"  Rejilla L1B: {lat_l1b.shape[0]} lineas ALT x "
        f"{lat_l1b.shape[1]} columnas ACT = {lat_l1b.size} puntos")

    # La geolocalizacion es unica para las 4 bandas, pero el TOA L1C no:
    # se coge la primera banda disponible solo para dibujar los puntos L1C
    lat_l1c = lon_l1c = None
    band_used = None

    for band in global_config.bands:
        _toa, lat_candidate, lon_candidate = quiet_read_l1c(
            FOLDER_MINE, global_config.l1c_toa + band + ".nc"
        )
        if lat_candidate is not None:
            lat_l1c, lon_l1c, band_used = lat_candidate, lon_candidate, band
            break

    if lat_l1c is None:
        log(f"  No se encuentra ningun {global_config.l1c_toa}<banda>.nc "
            f"en {FOLDER_MINE}.")
        return False

    log(f"  Rejilla L1C: {lat_l1c.size} puntos (leidos de {band_used})")
    log("  La rejilla L1C es regular porque se proyecta sobre las teselas "
        "MGRS; la L1B es irregular por la geometria de observacion.")

    plt.figure(figsize=(9, 9))

    if PLOT_ALL_L1B_POINTS:
        plt.scatter(lon_l1b.ravel(), lat_l1b.ravel(), s=2,
                    color="0.70", label="L1B grid (puntos)")

    # Contorno de la rejilla L1B, en negro
    lat_corners = getCorners(lat_l1b)
    lon_corners = getCorners(lon_l1b)
    plt.plot(lon_corners, lat_corners, color="black", linewidth=2.0,
             label="L1B grid (contorno)")

    # Puntos de la rejilla L1C, en rojo
    plt.scatter(lon_l1c, lat_l1c, s=4, color="red", label="L1C grid")

    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.title("L1B grid vs L1C grid (MGRS)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    output_dir = get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / PNG_GRID
    plt.savefig(output_path, dpi=150)
    plt.close()

    log(f"  Grafica guardada en: {output_path}")

    return True


# ============================================================================
# PLOTS 2 Y 3 - GSD ACROSS TRACK Y ALONG TRACK
# ============================================================================

def run_gsd_analysis(log):
    """
    Calcula y grafica la GSD across track y along track de la rejilla L1B.

    :param log: funcion que imprime y acumula una linea del informe
    :return: True si los plots se han generado
    """
    global_config = globalConfig()

    log("")
    log("=" * 96)
    log("PLOTS 2 Y 3 - GROUND SAMPLING DISTANCE (HAVERSINE)")
    log("=" * 96)

    lat, lon = quiet_read_geodetic(GM_DIR, global_config.gm_geoloc)

    if lat is None:
        log(f"  No se encuentra {global_config.gm_geoloc} en {GM_DIR}.")
        return False

    nalt, nact = lat.shape
    central_row = nalt // 2
    central_col = nact // 2

    log(f"  Rejilla L1B: {nalt} x {nact}")
    log(f"  Fila central (ALT)    : {central_row}")
    log(f"  Columna central (ACT) : {central_col}")
    log(f"  Funcion haversine     : {HAVERSINE_SOURCE}")
    log("  R = 6378137 m (radio equatorial WGS84)")
    log("")

    # --- ACROSS TRACK: entre columnas ACT consecutivas de la fila central ---
    gsd_act = haversine_series(lat[central_row, :], lon[central_row, :])
    describe_series(gsd_act, f"GSD ACROSS TRACK (fila ALT {central_row})", log)
    log("")

    # --- ALONG TRACK: entre filas ALT consecutivas de la columna central ---
    gsd_alt = haversine_series(lat[:, central_col], lon[:, central_col])
    describe_series(gsd_alt, f"GSD ALONG TRACK (columna ACT {central_col})", log)
    log("")

    output_dir = get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 2: across track
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(gsd_act.size), gsd_act, color="black")
    plt.xlabel("ACT pixel [-]")
    plt.ylabel("Ground sampling distance [m]")
    plt.title(f"GSD across track, central ALT row {central_row} of the L1B grid")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    path_act = output_dir / PNG_GSD_ACT
    plt.savefig(path_act, dpi=150)
    plt.close()
    log(f"  Grafica guardada en: {path_act}")

    # Plot 3: along track
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(gsd_alt.size), gsd_alt, color="black")
    plt.xlabel("ALT line [-]")
    plt.ylabel("Ground sampling distance [m]")
    plt.title(f"GSD along track, central ACT column {central_col} of the L1B grid")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    path_alt = output_dir / PNG_GSD_ALT
    plt.savefig(path_alt, dpi=150)
    plt.close()
    log(f"  Grafica guardada en: {path_alt}")

    log("")
    log("  Geometria del swath (explica la forma de la curva across track):")
    swath = float(np.mean(gsd_act)) * nact
    # Altura orbital implicita a partir de la GSD: GSD = h * pix / f
    pix_size, focal = 30e-6, 0.5262
    orbit_h = float(np.mean(gsd_act)) * focal / pix_size
    theta_edge = np.degrees(np.arctan((swath / 2.0) / orbit_h))
    gsd_edge_theory = float(np.mean(gsd_act)) / np.cos(np.radians(theta_edge)) ** 2
    log(f"    anchura del swath        : {swath / 1000:.3f} km "
        f"({nact} pixeles)")
    log(f"    altura orbital implicita : {orbit_h / 1000:.1f} km")
    log(f"    angulo al borde del swath: {theta_edge:.4f} deg")
    log(f"    variacion teorica 1/cos^2: "
        f"{(gsd_edge_theory - float(np.mean(gsd_act))) * 1000:.3f} mm "
        f"({(gsd_edge_theory / float(np.mean(gsd_act)) - 1) * 100:.6f} %)")
    log("    Con un swath tan estrecho el angulo de observacion apenas "
        "cambia de un extremo al otro, asi que la GSD across track es "
        "practicamente constante. La curvatura solo es apreciable en "
        "swaths mucho mas anchos.")

    log("")
    log("  Interpretacion:")
    log("    - La GSD along track sale practicamente constante: el satelite "
        "avanza a velocidad uniforme y el muestreo temporal es fijo.")
    log("    - La GSD across track deberia crecer hacia los bordes porque "
        "los pixeles extremos miran con mayor angulo y su huella en el "
        "suelo se alarga, pero en esta escena ese efecto es de fracciones "
        "de milimetro (ver la geometria del swath de arriba).")
    log("    - La rejilla L1C, en cambio, es regular a 100 m por "
        "construccion, porque se proyecta sobre las teselas MGRS en "
        "coordenadas UTM (zona 31R).")

    return True


# ============================================================================
# PLOT POR BANDA - TOA L1C
# ============================================================================

def run_toa_plots(log):
    """
    Dibuja un scatter del TOA L1C por banda.

    :param log: funcion que imprime y acumula una linea del informe
    :return: numero de graficas generadas
    """
    global_config = globalConfig()

    log("")
    log("=" * 96)
    log("PLOT POR BANDA - TOA L1C")
    log("=" * 96)

    output_dir = get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = 0

    for band in global_config.bands:

        toa, lat, lon = quiet_read_l1c(
            FOLDER_MINE, global_config.l1c_toa + band + ".nc"
        )

        if toa is None:
            log(f"  {band}: no se encuentra el fichero, se salta")
            continue

        toa = np.atleast_1d(np.asarray(toa, dtype=float))

        if toa.size != np.asarray(lat).size:
            log(f"  {band}: el TOA tiene {toa.size} valores pero hay "
                f"{np.asarray(lat).size} coordenadas. Revisa l1cProjtoa: "
                f"si dentro del bucle se hace 'toa_l1c = bisplev(...)' en "
                f"vez de 'toa_l1c[ii] = bisplev(...)', el vector se "
                f"sobreescribe con un escalar.")
            continue

        plt.figure(figsize=(9, 8))
        scatter = plt.scatter(lon, lat, c=toa, s=6, cmap="viridis")
        plt.colorbar(scatter, label="TOA [mW/m2/sr]")
        plt.xlabel("Longitude [deg]")
        plt.ylabel("Latitude [deg]")
        plt.title(f"L1C TOA {band}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = output_dir / PNG_TOA_TEMPLATE.format(band=band)
        plt.savefig(output_path, dpi=150)
        plt.close()

        log(f"  {band}: grafica guardada en {output_path} "
            f"({toa.size} puntos, TOA de {np.min(toa):.4f} a "
            f"{np.max(toa):.4f} mW/m2/sr)")

        # Una radiancia negativa no tiene sentido fisico: es un artefacto
        # del spline bivariante, que oscila donde la senal tiene un
        # escalon brusco (por ejemplo en los pixeles saturados del ISM).
        n_negative = int(np.count_nonzero(toa < 0))
        if n_negative > 0:
            log(f"      AVISO: {n_negative} punto(s) con radiancia negativa "
                f"(minimo {np.min(toa):.4f} mW/m2/sr). No tiene sentido "
                f"fisico: es el overshoot del spline de bisplrep en zonas "
                f"de gradiente fuerte o saturadas.")

        generated += 1

    log(f"  Graficas generadas: {generated} de {len(global_config.bands)}")

    return generated


# ============================================================================
# CROSS VALIDATION DE LOS PRODUCTOS L1C
# ============================================================================

def sort_by_coordinates(lat, lon, toa):
    """
    Ordena los puntos L1C por (lat, lon).

    Los puntos L1C provienen de un set de teselas MGRS, que en Python no
    conserva el orden, asi que dos ejecuciones pueden devolver los mismos
    puntos en orden distinto. Ordenando se pueden comparar.

    :param lat: latitudes [deg]
    :param lon: longitudes [deg]
    :param toa: radiancias
    :return: (lat, lon, toa) ordenados
    """
    lat = np.asarray(lat, dtype=float).ravel()
    lon = np.asarray(lon, dtype=float).ravel()
    toa = np.asarray(toa, dtype=float).ravel()

    order = np.lexsort((lon, lat))

    return lat[order], lon[order], toa[order]


def relative_difference_pct(mine, reference):
    """
    Diferencia relativa punto a punto en por ciento.

    :param mine: array con mis valores
    :param reference: array con los valores de referencia
    :return: array de diferencias relativas [%]
    """
    absolute_difference = np.abs(mine - reference)
    denominator = np.abs(reference)

    rel_diff = np.full(absolute_difference.shape, np.nan, dtype=float)

    significant = denominator > ABS_FLOOR
    rel_diff[significant] = (
        absolute_difference[significant] / denominator[significant] * 100.0
    )

    degenerate = ~significant
    rel_diff[degenerate] = np.where(
        absolute_difference[degenerate] <= ABS_FLOOR, 0.0, np.inf
    )

    return rel_diff


def run_crossvalidation(log):
    """
    Compara los productos L1C con los de referencia, ordenando los puntos.

    :param log: funcion que imprime y acumula una linea del informe
    :return: True si todas las bandas cumplen el criterio
    """
    global_config = globalConfig()

    log("")
    log("=" * 96)
    log("CROSS VALIDATION DE LOS PRODUCTOS L1C")
    log("=" * 96)

    if not DO_CROSSVALIDATION:
        log("  Desactivada (DO_CROSSVALIDATION = False).")
        return True

    if FOLDER_REFERENCE is None:
        log("  Sin carpeta de referencia, se salta.")
        return True

    log(f"  Mis salidas        : {FOLDER_MINE}")
    log(f"  Salidas referencia : {FOLDER_REFERENCE}")
    log(f"  Criterio           : diferencia relativa < "
        f"{REL_DIFF_THRESHOLD_PCT:g} % en el {MIN_PCT_POINTS_WITHIN:g} % "
        f"de los puntos")
    log("  Los puntos se ordenan por (lat, lon) antes de comparar, porque "
        "el set de teselas MGRS no conserva el orden.")
    log("")

    all_ok = True
    compared = 0

    for band in global_config.bands:

        filename = global_config.l1c_toa + band + ".nc"

        toa_mine, lat_mine, lon_mine = quiet_read_l1c(FOLDER_MINE, filename)
        toa_ref, lat_ref, lon_ref = quiet_read_l1c(FOLDER_REFERENCE, filename)

        if toa_mine is None or toa_ref is None:
            log(f"  {band}: falta alguno de los dos ficheros, se salta")
            continue

        lat_mine_s, lon_mine_s, toa_mine_s = sort_by_coordinates(
            lat_mine, lon_mine, toa_mine
        )
        lat_ref_s, lon_ref_s, toa_ref_s = sort_by_coordinates(
            lat_ref, lon_ref, toa_ref
        )

        if toa_mine_s.size != toa_ref_s.size:
            log(f"  {band}: DIFERENTES | numero de puntos distinto "
                f"({toa_mine_s.size} vs {toa_ref_s.size}), no se pueden "
                f"emparejar")
            all_ok = False
            continue

        # Comprobacion de que los puntos son los mismos
        max_coord_diff = max(
            float(np.max(np.abs(lat_mine_s - lat_ref_s))),
            float(np.max(np.abs(lon_mine_s - lon_ref_s))),
        )

        if max_coord_diff > COORD_TOL_DEG:
            log(f"  {band}: DIFERENTES | las coordenadas no coinciden tras "
                f"ordenar (max {max_coord_diff:.3e} deg), las rejillas no "
                f"son la misma")
            all_ok = False
            continue

        rel_diff = relative_difference_pct(toa_mine_s, toa_ref_s)
        absolute_difference = np.abs(toa_mine_s - toa_ref_s)

        finite_rel = rel_diff[np.isfinite(rel_diff)]
        max_rel = float(np.max(finite_rel)) if finite_rel.size else np.inf
        max_abs = float(np.max(absolute_difference))
        pct_within = (
            float(np.count_nonzero(rel_diff <= REL_DIFF_THRESHOLD_PCT))
            / rel_diff.size * 100.0
        )

        passed = pct_within >= MIN_PCT_POINTS_WITHIN
        verdict = "IGUALES" if passed else "DIFERENTES"
        all_ok = all_ok and passed
        compared += 1

        log(f"  {band}: {verdict} | {toa_mine_s.size} puntos | "
            f"pts OK {pct_within:.4f} % | max rel {max_rel:.6e} % | "
            f"max abs {max_abs:.6e} | coord max diff "
            f"{max_coord_diff:.3e} deg")

    log("")
    if compared == 0:
        log("  No se ha podido comparar ninguna banda.")
        return False

    if all_ok:
        log("  CONCLUSION: los productos L1C coinciden con los de "
            "referencia dentro del criterio aplicado.")
    else:
        log("  CONCLUSION: hay bandas que no cumplen el criterio. Si lo que "
            "falla es el numero de puntos o las coordenadas, el problema "
            "esta en la seleccion de teselas MGRS de l1cProjtoa. Si las "
            "coordenadas coinciden y lo que difiere son las radiancias, el "
            "problema esta en la interpolacion (parametros de bisplrep).")

    return all_ok


# ============================================================================
# EJECUCION
# ============================================================================

def main():
    """Ejecuta los test del L1C y guarda el informe."""
    report_lines = []

    def log(message=""):
        """Imprime por consola y acumula la linea para el informe."""
        print(message)
        report_lines.append(message)

    log("=" * 96)
    log("TEST DEL MODULO L1C")
    log("=" * 96)
    log(f"Carpeta del GM     : {GM_DIR}")
    log(f"Mis salidas L1C    : {FOLDER_MINE}")
    log(f"Salidas referencia : {FOLDER_REFERENCE}")
    log("")
    log("La geolocalizacion del GM es unica para las 4 bandas, por lo que "
        "los plots de geometria (rejillas y GSD) son unicos. El TOA si se "
        "grafica por banda.")
    log("=" * 96)

    grid_ok = run_grid_plot(log)
    gsd_ok = run_gsd_analysis(log)

    toa_plots = 0
    if PLOT_TOA_PER_BAND:
        toa_plots = run_toa_plots(log)

    crossval_ok = run_crossvalidation(log)

    log("")
    log("=" * 96)
    log("RESUMEN GLOBAL DEL TEST DEL L1C")
    log("=" * 96)
    log(f"Plot 1 - Rejillas L1B vs L1C : "
        f"{'GENERADO' if grid_ok else 'NO GENERADO'}")
    log(f"Plots 2 y 3 - GSD ACT y ALT  : "
        f"{'GENERADOS' if gsd_ok else 'NO GENERADOS'}")
    log(f"Plots de TOA por banda       : {toa_plots}")
    log(f"Cross validation             : "
        f"{'CORRECTO' if crossval_ok else 'CON DIFERENCIAS'}")
    log("=" * 96)

    save_report(report_lines)


if __name__ == "__main__":
    main()
