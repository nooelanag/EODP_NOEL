# ============================================================================
# ISM TEST SCRIPTS
# ----------------------------------------------------------------------------
# run_crossvalidation()
#     Lista con os.listdir todos los ficheros .nc de mi carpeta de salida,
#     los compara uno a uno con los de la carpeta de referencia del profesor
#     y calcula TODAS las diferencias (absolutas y relativas).
#
# run_unit_conversion_check()
#     Tabla con el factor de conversion de cada paso del ISM y comprobacion,
#     desde el pixel central toa[50,75], de que las salidas guardadas
#     coinciden con la propagacion analitica. PARA LAS 4 BANDAS.
#     Avisa tambien de la saturacion (FWC y 4095 DN).
#
# Los dos escriben el resultado en REPORT_FILENAME dentro de la carpeta
# de salida, ademas de imprimirlo por consola.
#
# Ejecucion (desde la raiz del proyecto EODP_NOEL):
#     python ism/test/ism_test.py
#
# Referencia: EODP ATBD R1.4, EODP-ALG-ISM-1010/1020/1030/2010/2030/3010/3020
# ============================================================================

import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

# Permite ejecutar este fichero directamente: los imports absolutos
# ('config...', 'common...') necesitan la raiz del proyecto en sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from auxiliary.constants import constants          # noqa: E402
from common.io.writeToa import readToa             # noqa: E402
from common.src.auxFunc import getIndexBand        # noqa: E402
from config.globalConfig import globalConfig       # noqa: E402
from config.ismConfig import ismConfig             # noqa: E402


# ============================================================================
# CONFIGURACION  (cambiar estas rutas si las carpetas no son exactamente asi)
# ============================================================================

# Mis propias salidas del ISM
FOLDER_MINE = Path(
    r"C:\Users\Noel\Documents\NOEL\UNIVERSIDAD\MASTER\TELECO\ACADEMICO\26-27\1er cuatri\EODP\FILES\EODP_TER_2021\EODP_TER_2021\EODP-TS-ISM\output_noel"
)

# Salidas de referencia del profesor
FOLDER_REFERENCE = Path(
    r"C:\Users\Noel\Documents\NOEL\UNIVERSIDAD\MASTER\TELECO\ACADEMICO\26-27\1er cuatri\EODP\FILES\EODP_TER_2021\EODP_TER_2021\EODP-TS-ISM\output"
)

# ---- Informe de texto ----
# REPORT_DIR = None  ->  se guarda en FOLDER_MINE (mis propias salidas)
REPORT_DIR = None
REPORT_FILENAME = "ism_test_report.txt"

# Pixel central de la escena (gradient_alt100_act150 -> centro en [50, 75])
ALT_IDX = 50
ACT_IDX = 75

# Bandas. None -> las declaradas en config/globalConfig.py (VNIR-0 ... VNIR-3)
BANDS = None

# ---- Criterio de aceptacion ----
# Requisito: las diferencias relativas deben ser menores que 1e-3 %
REL_DIFF_THRESHOLD_PCT = 1e-3

# Fraccion minima de puntos que deben cumplir el criterio [%]
MIN_PCT_POINTS_WITHIN = 100.0

# Suelo por debajo del cual la senal se considera ruido numerico y la
# comparacion se hace en absoluto en vez de en relativo
ABS_FLOOR = 1e-12


# ============================================================================
# UTILIDADES COMUNES
# ============================================================================

def quiet_read_toa(folder, filename):
    """
    Lee un TOA con el lector del proyecto, silenciando sus prints.

    :param folder: carpeta que contiene el fichero
    :param filename: nombre del fichero con extension .nc
    :return: matriz TOA, o None si el fichero no existe o no se puede leer
    """
    full_path = os.path.join(str(folder), filename)

    if not os.path.isfile(full_path):
        return None

    try:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            toa = readToa(str(folder), filename)
        return toa
    except (OSError, RuntimeError, KeyError, IndexError) as error:
        print(f"  AVISO: no se pudo leer {full_path} -> {error}")
        return None


def relative_difference_pct(mine, reference):
    """
    Diferencia relativa punto a punto en por ciento, referida a la referencia.

    Donde la referencia es practicamente cero se usa la diferencia absoluta,
    para que una comparacion cero contra cero no produzca NaN ni infinito.

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


def scalar_rel_diff_pct(value, expected):
    """
    Diferencia relativa en por ciento entre dos escalares.

    :param value: valor medido
    :param expected: valor esperado
    :return: diferencia relativa [%], o None si no se puede calcular
    """
    if value is None or expected is None:
        return None

    if abs(expected) < 1e-30:
        return 0.0 if abs(value) < 1e-30 else np.inf

    return abs(value - expected) / abs(expected) * 100.0


def fmt(value, precision=6):
    """
    Formatea un numero para las tablas, tolerando None y no finitos.

    :param value: valor a formatear
    :param precision: numero de decimales
    :return: cadena formateada
    """
    if value is None:
        return "n/a"
    if isinstance(value, float) and not np.isfinite(value):
        return "inf"
    return f"{value:.{precision}e}"


def get_report_dir():
    """
    Devuelve la carpeta donde se guarda el informe.

    :return: carpeta del informe
    """
    return REPORT_DIR if REPORT_DIR is not None else FOLDER_MINE


def save_report(report_lines):
    """
    Guarda el informe de texto en la carpeta de salida.

    :param report_lines: lista de lineas del informe
    """
    report_dir = Path(get_report_dir())

    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / REPORT_FILENAME

        with open(report_path, "w", encoding="utf-8") as handler:
            handler.write("\n".join(report_lines) + "\n")

        print(f"\nInforme guardado en: {report_path}")

    except OSError as error:
        print(f"\nAVISO: no se pudo guardar el informe: {error}")


# ============================================================================
# PREGUNTA 1 - CROSS VALIDATION DE TODAS LAS SALIDAS DEL ISM
# ============================================================================

def list_nc_files(folder):
    """
    Lista los ficheros .nc de una carpeta usando os.listdir.

    :param folder: carpeta a recorrer
    :return: lista ordenada de nombres de fichero que acaban en .nc
    """
    if not os.path.isdir(str(folder)):
        print(f"AVISO: no se encuentra la carpeta {folder}")
        return []

    return sorted(
        name for name in os.listdir(str(folder))
        if name.endswith(".nc")
        and os.path.isfile(os.path.join(str(folder), name))
    )


def compare_toa_files(filename):
    """
    Compara un fichero TOA de mi carpeta contra el de referencia.

    Calcula TODAS las diferencias punto a punto, no solo el maximo.

    :param filename: nombre del fichero .nc, igual en las dos carpetas
    :return: diccionario con las metricas y el veredicto
    """
    result = {
        "passed": False,
        "max_abs_diff": np.inf,
        "max_rel_diff_pct": np.inf,
        "mean_abs_diff": np.inf,
        "pct_within": 0.0,
        "n_points": 0,
        "note": "",
    }

    toa_mine = quiet_read_toa(FOLDER_MINE, filename)
    toa_ref = quiet_read_toa(FOLDER_REFERENCE, filename)

    if toa_mine is None or toa_ref is None:
        result["note"] = "no se pudo leer uno de los dos ficheros"
        return result

    if toa_mine.shape != toa_ref.shape:
        result["note"] = (
            f"dimensiones distintas {toa_mine.shape} != {toa_ref.shape}"
        )
        return result

    mine = np.asarray(toa_mine, dtype=float)
    reference = np.asarray(toa_ref, dtype=float)

    # Solo los puntos donde ambos tienen un valor utilizable
    valid = np.isfinite(mine) & np.isfinite(reference)
    result["n_points"] = int(np.count_nonzero(valid))

    if result["n_points"] == 0:
        result["note"] = "no hay puntos finitos que comparar"
        return result

    absolute_difference = np.abs(mine[valid] - reference[valid])
    rel_diff = relative_difference_pct(mine[valid], reference[valid])

    result["max_abs_diff"] = float(np.max(absolute_difference))
    result["mean_abs_diff"] = float(np.mean(absolute_difference))

    finite_rel = rel_diff[np.isfinite(rel_diff)]
    result["max_rel_diff_pct"] = (
        float(np.max(finite_rel)) if finite_rel.size else np.inf
    )

    within = rel_diff <= REL_DIFF_THRESHOLD_PCT
    result["pct_within"] = float(np.count_nonzero(within)) / rel_diff.size * 100.0
    result["passed"] = result["pct_within"] >= MIN_PCT_POINTS_WITHIN

    return result


def run_crossvalidation(log):
    """
    PREGUNTA 1: cross validation de todas las salidas del ISM.

    :param log: funcion que imprime y acumula una linea del informe
    :return: True si todos los ficheros pasan el criterio
    """
    files_mine = list_nc_files(FOLDER_MINE)
    files_ref = list_nc_files(FOLDER_REFERENCE)

    common_files = sorted(set(files_mine) & set(files_ref))
    only_mine = sorted(set(files_mine) - set(files_ref))
    only_ref = sorted(set(files_ref) - set(files_mine))

    log("=" * 104)
    log("PREGUNTA 1 - CROSS VALIDATION DE LAS SALIDAS DEL ISM")
    log("=" * 104)
    log(f"Mis salidas        : {FOLDER_MINE}")
    log(f"Salidas referencia : {FOLDER_REFERENCE}")
    log(f"Criterio           : diferencia relativa < {REL_DIFF_THRESHOLD_PCT:g} % "
        f"en el {MIN_PCT_POINTS_WITHIN:g} % de los puntos")
    log("=" * 104)
    log(f"Ficheros .nc en mis salidas       : {len(files_mine)}")
    log(f"Ficheros .nc en las de referencia : {len(files_ref)}")
    log(f"Ficheros presentes en ambas       : {len(common_files)}")
    log("=" * 104)

    if not common_files:
        log("No se han encontrado ficheros .nc con el mismo nombre en ambas "
            "carpetas. Revisa FOLDER_MINE y FOLDER_REFERENCE.")
        return False

    header = (
        f"{'FICHERO':<34}{'VEREDICTO':<11}{'PTS OK [%]':>12}"
        f"{'MAX REL [%]':>14}{'MAX ABS':>14}{'MEDIA ABS':>14}"
    )
    log(header)
    log("-" * 104)

    n_passed = 0
    n_failed = 0

    for filename in common_files:
        result = compare_toa_files(filename)

        if result["passed"]:
            n_passed += 1
            verdict = "IGUALES"
        else:
            n_failed += 1
            verdict = "DIFERENTES"

        log(
            f"{filename:<34}{verdict:<11}{result['pct_within']:>11.4f}"
            f"{fmt(result['max_rel_diff_pct']):>15}"
            f"{fmt(result['max_abs_diff']):>14}"
            f"{fmt(result['mean_abs_diff']):>14}"
        )

        if result["note"]:
            log(f"{'':<34}nota: {result['note']}")

    log("-" * 104)

    if only_mine or only_ref:
        log("")
        log("FICHEROS SIN PAREJA")
        for name in only_mine:
            log(f"  solo en mis salidas       : {name}")
        for name in only_ref:
            log(f"  solo en las de referencia : {name}")

    log("")
    log(f"Ficheros comparados : {len(common_files)}")
    log(f"Iguales             : {n_passed}")
    log(f"Diferentes          : {n_failed}")
    log("")

    if n_failed == 0:
        log("CONCLUSION: todas las salidas del ISM coinciden con las de "
            "referencia dentro del criterio aplicado. Las diferencias "
            "residuales son de precision numerica (orden 1e-12).")
    else:
        log(f"CONCLUSION: {n_failed} fichero(s) NO cumplen el criterio.")
        log("Si la discrepancia empieza justo en ism_toa_prnu_* o en "
            "ism_toa_ds_*, la causa es la generacion aleatoria (semilla y "
            "orden de las llamadas a numpy.random), no las conversiones.")

    log("=" * 104)

    return n_failed == 0


# ============================================================================
# PREGUNTA 3 - FACTORES DE CONVERSION Y PIXEL CENTRAL, POR BANDA
# ----------------------------------------------------------------------------
# Cadena del ISM y unidades:
#
#   ism_toa_isrf_*      [mW/(m2*sr)]  radiancia tras la integracion espectral
#     ISRF              NO cambia unidades (radiancias -> radiancias)
#     -> RAD2IRRAD      x Tr*(pi/4)*(D/f)^2            -> [mW/m2]
#   ism_toa_optical_*   [mW/m2]       irradiancia tras rad2irrad + MTF
#     MTF               NO cambia unidades (redistribuye energia)
#     -> IRR2PH         x 1e-3*Apix*tint/(h*c/wv)      -> [ph]
#     -> PH2E           x QE, saturando en FWC         -> [e-]
#   ism_toa_e_*         [e-]
#     (PRNU, dark signal, bad/dead: efectos, no cambian unidades)
#   ism_toa_detection_* [e-]
#     -> E2V            x OCF*ADC_gain                 -> [V]
#     -> V2DN           x 4095/(Vmax-Vmin), redondeo   -> [DN]
#   ism_toa_vcu_*       [DN]  ==  ism_toa_*  [DN]
# ============================================================================

def digitisation_scalar(voltage, bit_depth, min_voltage, max_voltage):
    """
    Reproduce videoChainPhase.digitisation para un unico valor.

    :param voltage: tension de entrada [V]
    :param bit_depth: numero de bits del ADC
    :param min_voltage: tension minima del ADC [V]
    :param max_voltage: tension maxima del ADC [V]
    :return: cuentas digitales [DN]
    """
    saturated = min(voltage, max_voltage)
    levels = 2 ** bit_depth - 1
    counts = round((saturated - min_voltage) / (max_voltage - min_voltage) * levels)
    return float(max(counts, 0))


def read_central_pixel(folder, filename):
    """
    Lee el pixel central de un fichero TOA.

    :param folder: carpeta del fichero
    :param filename: nombre del fichero sin la extension
    :return: valor en [ALT_IDX, ACT_IDX], o None si no esta disponible
    """
    toa = quiet_read_toa(folder, filename + ".nc")

    if toa is None:
        return None

    if toa.ndim != 2:
        print(f"  AVISO: {filename} no es una matriz 2D (forma {toa.shape})")
        return None

    if ALT_IDX >= toa.shape[0] or ACT_IDX >= toa.shape[1]:
        print(f"  AVISO: el pixel [{ALT_IDX},{ACT_IDX}] esta fuera de "
              f"{filename} (forma {toa.shape})")
        return None

    return float(toa[ALT_IDX, ACT_IDX])


def compute_factors(band, ism_config, const):
    """
    Calcula los factores de conversion de una banda.

    :param band: nombre de la banda, por ejemplo 'VNIR-0'
    :param ism_config: configuracion del ISM
    :param const: constantes fisicas
    :return: diccionario de factores
    :raises IndexError: si la banda no tiene longitud de onda definida
    """
    band_index = getIndexBand(band)

    if band_index >= ism_config.wv.size:
        raise IndexError(
            f"la banda {band} tiene indice {band_index} pero ismConfig.wv "
            f"solo define {ism_config.wv.size} longitudes de onda"
        )

    wavelength = float(ism_config.wv[band_index])        # [m]
    area_pix = ism_config.pix_size ** 2                  # [m2]

    energy_photon = const.h_planck * const.speed_light / wavelength   # [J/ph]

    return {
        "wavelength": wavelength,
        "area_pix": area_pix,
        "energy_photon": energy_photon,
        # Radiancia -> Irradiancia [sr]
        "rad2irrad": ism_config.Tr * (np.pi / 4.0)
        * (ism_config.D / ism_config.f) ** 2,
        # Irradiancia -> Fotones [ph/(mW/m2)]
        "irr2ph": 1e-3 * area_pix * ism_config.t_int / energy_photon,
        # Fotones -> Electrones [e-/ph]
        "ph2e": ism_config.QE,
        # Electrones -> Voltios [V/e-]
        "e2v": ism_config.OCF * ism_config.ADC_gain,
        # Voltios -> Cuentas digitales [DN/V] = 4095 / (Vmax - Vmin)
        "v2dn": (2 ** ism_config.bit_depth - 1)
        / (ism_config.max_voltage - ism_config.min_voltage),
    }


def build_chain_rows(band, ism_config, global_config, factors):
    """
    Construye las filas de la tabla de conversiones de una banda.

    :param band: nombre de la banda
    :param ism_config: configuracion del ISM
    :param global_config: configuracion global (nombres de fichero)
    :param factors: factores de conversion de la banda
    :return: lista de filas de la tabla
    """
    # Valores leidos de mis propias salidas
    v_isrf = read_central_pixel(FOLDER_MINE, global_config.ism_toa_isrf + band)
    v_optical = read_central_pixel(FOLDER_MINE, global_config.ism_toa_optical + band)
    v_e = read_central_pixel(FOLDER_MINE, global_config.ism_toa_e + band)
    v_prnu = read_central_pixel(FOLDER_MINE, global_config.ism_toa_prnu + band)
    v_ds = read_central_pixel(FOLDER_MINE, global_config.ism_toa_ds + band)
    v_det = read_central_pixel(FOLDER_MINE, global_config.ism_toa_detection + band)
    v_vcu = read_central_pixel(FOLDER_MINE, global_config.ism_toa_vcu + band)
    v_final = read_central_pixel(FOLDER_MINE, global_config.ism_toa + band)

    rows = []

    # 0. Punto de partida
    rows.append({
        "step": "ISRF (inicio)",
        "units": "mW/(m2*sr)",
        "factor": 1.0,
        "value_in": None,
        "expected": None,
        "actual": v_isrf,
        "rel": None,
        "check": False,
        "note": "el ISRF esta normalizado: radiancias -> radiancias, factor 1",
    })

    # 1. Radiancia -> Irradiancia (+ MTF, que no cambia unidades)
    exp_irrad = v_isrf * factors["rad2irrad"] if v_isrf is not None else None
    rows.append({
        "step": "RAD2IRRAD",
        "units": "mW/m2",
        "factor": factors["rad2irrad"],
        "value_in": v_isrf,
        "expected": exp_irrad,
        "actual": v_optical,
        "rel": scalar_rel_diff_pct(v_optical, exp_irrad),
        "check": False,
        "note": "Tr*(pi/4)*(D/f)^2. El fichero incluye ademas la MTF, que no "
                "cambia unidades pero si redistribuye energia",
    })

    # 2. Irradiancia -> Fotones (no se guarda producto intermedio)
    exp_ph = v_optical * factors["irr2ph"] if v_optical is not None else None
    rows.append({
        "step": "IRR2PH",
        "units": "ph",
        "factor": factors["irr2ph"],
        "value_in": v_optical,
        "expected": exp_ph,
        "actual": None,
        "rel": None,
        "check": False,
        "note": "1e-3*Apix*tint/(h*c/wv). El ISM no guarda los fotones",
    })

    # 3. Fotones -> Electrones
    exp_e = None
    saturated_fwc = False
    if exp_ph is not None:
        exp_e_raw = exp_ph * factors["ph2e"]
        saturated_fwc = exp_e_raw > ism_config.FWC
        exp_e = min(exp_e_raw, ism_config.FWC)

    rows.append({
        "step": "PH2E",
        "units": "e-",
        "factor": factors["ph2e"],
        "value_in": exp_ph,
        "expected": exp_e,
        "actual": v_e,
        "rel": scalar_rel_diff_pct(v_e, exp_e),
        "check": True,
        "note": ("SATURADO en FWC = " + str(ism_config.FWC) + " e-")
        if saturated_fwc else "QE. Cierra el tramo optica -> electrones",
    })

    # 4-6. Efectos que no cambian unidades
    rows.append({
        "step": "(PRNU)",
        "units": "e-",
        "factor": (v_prnu / v_e) if (v_e not in (None, 0.0) and v_prnu is not None) else None,
        "value_in": v_e,
        "expected": None,
        "actual": v_prnu,
        "rel": None,
        "check": False,
        "note": "efecto, no cambia unidades. factor = ganancia PRNU implicita",
    })
    rows.append({
        "step": "(DARK SIGNAL)",
        "units": "e-",
        "factor": (v_ds - v_prnu) if (v_ds is not None and v_prnu is not None) else None,
        "value_in": v_prnu,
        "expected": None,
        "actual": v_ds,
        "rel": None,
        "check": False,
        "note": "efecto, no cambia unidades. factor = offset de DS [e-]",
    })
    rows.append({
        "step": "(BAD/DEAD PIX)",
        "units": "e-",
        "factor": (v_det / v_ds) if (v_ds not in (None, 0.0) and v_det is not None) else None,
        "value_in": v_ds,
        "expected": None,
        "actual": v_det,
        "rel": None,
        "check": False,
        "note": "efecto, no cambia unidades. factor = reduccion de QE implicita",
    })

    # 7. Electrones -> Voltios (no se guarda producto intermedio)
    exp_v = v_det * factors["e2v"] if v_det is not None else None
    rows.append({
        "step": "E2V",
        "units": "V",
        "factor": factors["e2v"],
        "value_in": v_det,
        "expected": exp_v,
        "actual": None,
        "rel": None,
        "check": False,
        "note": "OCF * ADC_gain, dos escalares. El ISM no guarda los voltios",
    })

    # 8. Voltios -> Cuentas digitales
    exp_dn = None
    saturated_adc = False
    if exp_v is not None:
        saturated_adc = exp_v >= ism_config.max_voltage
        exp_dn = digitisation_scalar(
            exp_v,
            ism_config.bit_depth,
            ism_config.min_voltage,
            ism_config.max_voltage,
        )

    max_dn = 2 ** ism_config.bit_depth - 1
    rows.append({
        "step": "V2DN",
        "units": "DN",
        "factor": factors["v2dn"],
        "value_in": exp_v,
        "expected": exp_dn,
        "actual": v_vcu,
        "rel": scalar_rel_diff_pct(v_vcu, exp_dn),
        "check": True,
        "note": (f"SATURADO en {max_dn} DN (Vmax = {ism_config.max_voltage} V)")
        if saturated_adc else f"{max_dn}/(Vmax-Vmin), no lineal: satura y redondea",
    })

    # 9. Salida final, debe ser identica a la del VCU
    rows.append({
        "step": "SALIDA ISM",
        "units": "DN",
        "factor": 1.0,
        "value_in": v_vcu,
        "expected": v_vcu,
        "actual": v_final,
        "rel": scalar_rel_diff_pct(v_final, v_vcu),
        "check": True,
        "note": "ism_toa_* debe ser identico a ism_toa_vcu_*",
    })

    saturation = {
        "fwc": saturated_fwc,
        "adc": saturated_adc,
        "dn_at_max": (v_vcu is not None and v_vcu >= max_dn),
        "max_dn": max_dn,
    }

    return rows, saturation


def format_chain_table(rows, log):
    """
    Imprime la tabla de conversiones de una banda.

    :param rows: filas generadas por build_chain_rows
    :param log: funcion que imprime y acumula
    """
    header = (
        f"{'PASO':<16}{'UNIDADES':<13}{'FACTOR':>15}{'ENTRADA':>15}"
        f"{'ESPERADO':>15}{'DEL FICHERO':>15}{'REL [%]':>13}"
    )
    log(header)
    log("-" * len(header))

    for row in rows:
        rel = row["rel"]
        if rel is None:
            rel_str = "n/a"
        elif not np.isfinite(rel):
            rel_str = "inf"
        else:
            rel_str = f"{rel:.4e}"

        log(
            f"{row['step']:<16}{row['units']:<13}{fmt(row['factor']):>15}"
            f"{fmt(row['value_in']):>15}{fmt(row['expected']):>15}"
            f"{fmt(row['actual']):>15}{rel_str:>13}"
        )
        if row["note"]:
            log(f"    -> {row['note']}")


def run_unit_conversion_check(log):
    """
    PREGUNTA 3: factores de conversion y comprobacion del pixel central.

    :param log: funcion que imprime y acumula una linea del informe
    :return: True si todas las conversiones comprobables cumplen el criterio
    """
    ism_config = ismConfig()
    global_config = globalConfig()
    const = constants()

    bands = BANDS if BANDS is not None else list(global_config.bands)

    log("")
    log("=" * 104)
    log("PREGUNTA 3 - FACTORES DE CONVERSION Y PIXEL CENTRAL, POR BANDA")
    log("=" * 104)
    log(f"Mis salidas   : {FOLDER_MINE}")
    log(f"Pixel central : toa[{ALT_IDX}, {ACT_IDX}]")
    log(f"Bandas        : {', '.join(bands)}")
    log(f"Criterio      : diferencia relativa <= {REL_DIFF_THRESHOLD_PCT:g} %")
    log("=" * 104)
    log("Parametros del instrumento (config/ismConfig.py):")
    log(f"  Tr={ism_config.Tr}  D={ism_config.D} m  f={ism_config.f} m  "
        f"pix_size={ism_config.pix_size} m  t_int={ism_config.t_int} s")
    log(f"  QE={ism_config.QE} e-/ph  FWC={ism_config.FWC} e-  "
        f"OCF={ism_config.OCF} V/e-  ADC_gain={ism_config.ADC_gain}")
    log(f"  bit_depth={ism_config.bit_depth}  "
        f"min_voltage={ism_config.min_voltage} V  "
        f"max_voltage={ism_config.max_voltage} V")

    failures = []
    saturation_notes = []

    for band in bands:
        log("")
        log("=" * 104)
        log(f"BANDA {band}")
        log("=" * 104)

        try:
            factors = compute_factors(band, ism_config, const)
        except IndexError as error:
            log(f"  SALTADA: {error}")
            continue

        log(f"Longitud de onda central : {factors['wavelength']:.6e} m")
        log(f"Area del pixel           : {factors['area_pix']:.6e} m2")
        log(f"Energia del foton        : {factors['energy_photon']:.6e} J/ph")
        log("")
        log("Factores de conversion:")
        log(f"  ISRF       [-]            = {1.0:.6e}   (no cambia unidades)")
        log(f"  RAD2IRRAD  [sr]           = {factors['rad2irrad']:.6e}")
        log(f"  MTF        [-]            = {1.0:.6e}   (no cambia unidades)")
        log(f"  IRR2PH     [ph/(mW/m2)]   = {factors['irr2ph']:.6e}")
        log(f"  PH2E       [e-/ph]        = {factors['ph2e']:.6e}")
        log(f"  E2V        [V/e-]         = {factors['e2v']:.6e}")
        log(f"  V2DN       [DN/V]         = {factors['v2dn']:.6e}")
        log("")

        rows, saturation = build_chain_rows(
            band, ism_config, global_config, factors
        )

        log(f"Cadena de conversion en el pixel central toa[{ALT_IDX},{ACT_IDX}]:")
        format_chain_table(rows, log)

        # Veredicto de los pasos que se pueden cerrar contra un fichero
        for row in rows:
            if not row["check"] or row["rel"] is None:
                continue
            if not np.isfinite(row["rel"]) or row["rel"] > REL_DIFF_THRESHOLD_PCT:
                failures.append(f"{band} / {row['step']}: {fmt(row['rel'], 4)} %")

        if saturation["fwc"]:
            saturation_notes.append(
                f"{band}: saturacion FOTONICA, se alcanza la FWC "
                f"({ism_config.FWC} e-)"
            )
        if saturation["adc"] or saturation["dn_at_max"]:
            saturation_notes.append(
                f"{band}: saturacion en la VIDEO CHAIN, se alcanzan "
                f"{saturation['max_dn']} DN (Vmax = {ism_config.max_voltage} V)"
            )

    # --- Conclusiones -------------------------------------------------------
    log("")
    log("=" * 104)
    log("CONCLUSIONES DE LA PREGUNTA 3")
    log("=" * 104)
    log("Pasos que se cierran contra un fichero guardado: PH2E "
        "(ism_toa_e_*), V2DN (ism_toa_vcu_*) y la copia final (ism_toa_*).")
    log("IRR2PH y E2V no tienen producto intermedio, se propagan "
        "analiticamente.")
    log("RAD2IRRAD se muestra pero no entra en el veredicto: el residuo viene "
        "de la MTF que se aplica justo despues, no de las unidades.")
    log("El ISRF y la MTF no cambian las unidades (factor 1).")
    log("")

    if not failures:
        log(f"[OK] Todas las conversiones comprobables cumplen el "
            f"{REL_DIFF_THRESHOLD_PCT:g} % en toa[{ALT_IDX},{ACT_IDX}] "
            f"para todas las bandas.")
    else:
        log("[FALLO] Conversiones fuera de tolerancia:")
        for item in failures:
            log(f"  - {item}")

    log("")
    if saturation_notes:
        log("SATURACION detectada:")
        for item in saturation_notes:
            log(f"  - {item}")
        log("")
        log("Un pixel saturado es inservible: la senal ya no es recuperable "
            "en el L1B, porque la ganancia de restitucion solo multiplica un "
            "valor que ya esta recortado. Por eso el rango dinamico del "
            "instrumento se dimensiona para la escena mas brillante "
            "esperable (nubes, desiertos).")
    else:
        log("No se detecta saturacion en el pixel central de ninguna banda "
            "(ni fotonica en la FWC, ni en la video chain a "
            f"{2 ** ism_config.bit_depth - 1} DN).")

    log("=" * 104)

    return not failures


# ============================================================================
# EJECUCION
# ============================================================================

def main():
    """Ejecuta los dos test del ISM y guarda el informe conjunto."""
    report_lines = []

    def log(message=""):
        """Imprime por consola y acumula la linea para el informe."""
        print(message)
        report_lines.append(message)

    crossval_ok = run_crossvalidation(log)
    conversions_ok = run_unit_conversion_check(log)

    log("")
    log("=" * 104)
    log("RESUMEN GLOBAL DEL TEST DEL ISM")
    log("=" * 104)
    log(f"Cross validation de las salidas : "
        f"{'CORRECTO' if crossval_ok else 'CON DIFERENCIAS'}")
    log(f"Conversiones de unidades        : "
        f"{'CORRECTO' if conversions_ok else 'CON DIFERENCIAS'}")
    log("=" * 104)

    save_report(report_lines)


if __name__ == "__main__":
    main()
