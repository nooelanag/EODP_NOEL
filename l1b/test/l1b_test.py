# ============================================================
# L1B TEST SCRIPTS
# ============================================================
# run_crossvalidation()
#     Cross validate los ficheros .nc de mis salidas L1B (ecualizadas)
#     contra las salidas de referencia del profesor.
#     Cubre automaticamente todas las bandas presentes en la carpeta.
#     Vuelca el informe a REPORT_FILENAME dentro de la carpeta de salida.
#
# plot_equalization_effect()
#     Genera UN PNG POR BANDA (VNIR-0 ... VNIR-3) comparando la TOA L1B
#     ecualizada (negro), la no ecualizada (rojo) y la verdad / salida
#     del ISRF (azul).
#
# Ejecucion:
#     python l1b/test/l1b_test.py
# ============================================================

import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACION - BANDAS
# ============================================================

# Las 4 bandas del instrumento (las mismas que config/globalConfig.py)
BANDS = ["VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3"]


# ============================================================
# CONFIGURACION - CROSS VALIDATION
# ============================================================

# Carpetas que contienen los ficheros .nc
FOLDER_1 = Path(
    r"C:\Users\Noel\Documents\NOEL\UNIVERSIDAD\MASTER\TELECO\ACADEMICO\26-27\1er cuatri\EODP\FILES\EODP_TER_2021\EODP_TER_2021\EODP-TS-L1B\output_noel"
)

FOLDER_2 = Path(
    r"C:\Users\Noel\Documents\NOEL\UNIVERSIDAD\MASTER\TELECO\ACADEMICO\26-27\1er cuatri\EODP\FILES\EODP_TER_2021\EODP_TER_2021\EODP-TS-L1B\output"
)

# Tolerancia para considerar que dos valores son iguales.
# rtol = 1e-5 equivale a una diferencia relativa de 1e-3 %,
# que es el requisito de cross-validation.
RTOL = 1e-5
ATOL = 1e-8

# ---- Informe de texto ----
# Carpeta donde se guarda el informe.
# REPORT_DIR = None  ->  se guarda en FOLDER_1 (mis propias salidas)
REPORT_DIR = None
REPORT_FILENAME = "l1b_crossvalidation_report.txt"


# ============================================================
# CONFIGURACION - PLOT DE LA ECUALIZACION (4 BANDAS)
# ============================================================
# Ahora se trabaja con CARPETAS + prefijos, y el nombre de cada fichero
# se construye por banda. Asi se generan las 4 graficas de una pasada.

# Carpeta con las salidas L1B CON equalización (linea negra)
FOLDER_EQ = Path(
    r"C:\Users\Noel\Documents\NOEL\UNIVERSIDAD\MASTER\TELECO\ACADEMICO\26-27\1er cuatri\EODP\FILES\EODP_TER_2021\EODP_TER_2021\EODP-TS-L1B\output_noel"
)

# Carpeta con las salidas L1B SIN equalización (linea roja)
FOLDER_NO_EQ = Path(
    r"C:\Users\Noel\Documents\NOEL\UNIVERSIDAD\MASTER\TELECO\ACADEMICO\26-27\1er cuatri\EODP\FILES\EODP_TER_2021\EODP_TER_2021\EODP-TS-L1B\output_noel_no_equalized"
)

# Carpeta de ENTRADA, con la salida del ISRF = verdad (linea azul)
FOLDER_INPUT = Path(
    r"C:\Users\Noel\Documents\NOEL\UNIVERSIDAD\MASTER\TELECO\ACADEMICO\26-27\1er cuatri\EODP\FILES\EODP_TER_2021\EODP_TER_2021\EODP-TS-L1B\input"
)

# Prefijos de los nombres de fichero (igual que en config/globalConfig.py)
L1B_TOA_PREFIX = "l1b_toa_"
ISRF_TOA_PREFIX = "ism_toa_isrf_"

# Carpeta y nombre de los PNG de salida.
# Se guardan dentro de output_noel para que entren en el data pack.
OUTPUT_DIR = Path(
    r"C:\Users\Noel\Documents\NOEL\UNIVERSIDAD\MASTER\TELECO\ACADEMICO\26-27\1er cuatri\EODP\FILES\EODP_TER_2021\EODP_TER_2021\EODP-TS-L1B\output_noel"
)

# {band} se sustituye por el nombre de la banda
OUTPUT_FILENAME_TEMPLATE = "equalization_effect_{band}.png"

# Fila (row) a extraer de cada fichero
ROW_INDEX = 50

# Nombre de la variable a leer dentro del .nc.
# Si se deja en None, se usa automáticamente la única variable de datos
# (no-coordenada) que tenga el fichero.
VARIABLE_NAME = None


# ============================================================
# CROSS VALIDATION - FUNCIONES
# ============================================================

def compare_netcdf_files(file1, file2):

    try:
        with xr.open_dataset(file1) as ds1, xr.open_dataset(file2) as ds2:

            # Comprobar que tienen las mismas variables
            vars1 = set(ds1.variables)
            vars2 = set(ds2.variables)

            if vars1 != vars2:
                missing_in_1 = vars2 - vars1
                missing_in_2 = vars1 - vars2

                print("  Variables diferentes:")
                if missing_in_1:
                    print(f"    - Faltan en fichero 1: {missing_in_1}")
                if missing_in_2:
                    print(f"    - Faltan en fichero 2: {missing_in_2}")

                return False, np.inf

            max_difference = 0.0

            # Comparar cada variable
            for var_name in vars1:

                data1 = ds1[var_name].values
                data2 = ds2[var_name].values

                # Comprobar dimensiones/tamaño
                if data1.shape != data2.shape:
                    print(
                        f"  Variable '{var_name}' tiene diferentes dimensiones: "
                        f"{data1.shape} != {data2.shape}"
                    )
                    return False, np.inf

                # Variables numéricas
                if np.issubdtype(data1.dtype, np.number) and np.issubdtype(
                    data2.dtype, np.number
                ):
                    # Diferencia absoluta ignorando NaN
                    with np.errstate(invalid="ignore"):
                        difference = np.abs(data1 - data2)

                    if np.any(np.isfinite(difference)):
                        variable_max_difference = np.nanmax(difference)
                        max_difference = max(
                            max_difference, variable_max_difference
                        )

                    # Comparación con tolerancia, considerando NaN == NaN
                    equal = np.allclose(
                        data1,
                        data2,
                        rtol=RTOL,
                        atol=ATOL,
                        equal_nan=True,
                    )

                    if not equal:
                        return False, max_difference

                # Variables no numéricas
                else:
                    if not np.array_equal(data1, data2):
                        return False, np.inf

            return True, max_difference

    except Exception as e:
        print(f"  ERROR al comparar: {e}")
        return False, np.inf


def save_report(report_lines):
    """
    Guarda el informe de cross validation en un fichero de texto.
    :param report_lines: lista de lineas del informe
    """
    report_dir = REPORT_DIR if REPORT_DIR is not None else FOLDER_1

    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / REPORT_FILENAME

        with open(report_path, "w", encoding="utf-8") as handler:
            handler.write("\n".join(report_lines) + "\n")

        print(f"\nInforme guardado en: {report_path}")

    except OSError as error:
        print(f"\nAVISO: no se pudo guardar el informe: {error}")


def run_crossvalidation():
    """Compara todos los .nc comunes a FOLDER_1 y FOLDER_2."""

    report_lines = []

    def log(message=""):
        """Imprime por consola y acumula la linea para el informe."""
        print(message)
        report_lines.append(message)

    # Obtener SOLO ficheros .nc
    files_1 = {
        file.name: file
        for file in FOLDER_1.glob("*.nc")
        if file.is_file()
    }

    files_2 = {
        file.name: file
        for file in FOLDER_2.glob("*.nc")
        if file.is_file()
    }

    # Buscar únicamente los ficheros que existen en ambas carpetas
    common_files = sorted(set(files_1) & set(files_2))

    log("=" * 80)
    log("CROSS VALIDATION DE FICHEROS NETCDF")
    log("=" * 80)
    log(f"Mis salidas       : {FOLDER_1}")
    log(f"Salidas referencia: {FOLDER_2}")
    log(f"Criterio aplicado : numpy.allclose(rtol={RTOL:g}, atol={ATOL:g})")
    log(f"                    rtol={RTOL:g} equivale a una diferencia "
        f"relativa de 1e-3 %")
    log("=" * 80)

    log(f"Ficheros .nc en carpeta 1: {len(files_1)}")
    log(f"Ficheros .nc en carpeta 2: {len(files_2)}")
    log(f"Ficheros encontrados en ambas carpetas: {len(common_files)}")
    log("=" * 80)

    if not common_files:
        log("No se han encontrado ficheros .nc con el mismo nombre en ambas carpetas.")
        save_report(report_lines)
        return

    equal_count = 0
    different_count = 0

    # Comparar cada pareja de ficheros con el mismo nombre
    for filename in common_files:

        file1 = files_1[filename]
        file2 = files_2[filename]

        log("")
        log(filename)

        # Se captura lo que imprime compare_netcdf_files para que los
        # avisos tambien queden recogidos en el informe
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            are_equal, max_difference = compare_netcdf_files(file1, file2)

        captured = buffer.getvalue().rstrip()
        if captured:
            log(captured)

        if are_equal:
            equal_count += 1
            log(f"  IGUALES     | Diferencia máxima: {max_difference:.12e}")
        else:
            different_count += 1

            if np.isfinite(max_difference):
                log(f"  DIFERENTES  | Diferencia máxima: {max_difference:.12e}")
            else:
                log("  DIFERENTES  | No se puede calcular la diferencia máxima")

    # ========================================================
    # RESUMEN
    # ========================================================

    log("")
    log("=" * 80)
    log("RESUMEN")
    log("=" * 80)
    log(f"Ficheros comparados: {len(common_files)}")
    log(f"Iguales:             {equal_count}")
    log(f"Diferentes:          {different_count}")
    log("=" * 80)

    if different_count == 0:
        log("CONCLUSION: todas las salidas L1B coinciden con las de "
            "referencia bajo el criterio aplicado.")
    else:
        log(f"CONCLUSION: {different_count} fichero(s) NO coinciden con las "
            "salidas de referencia.")
    log("=" * 80)

    save_report(report_lines)


# ============================================================
# PLOT DE LA ECUALIZACION - FUNCIONES
# ============================================================
# PLOT FROM YOUR OUTPUTS THE EQUALISED OUTPUT VERSUS NOT EQUALISED
# VERSUS THE TRUTH, PARA LAS 4 BANDAS
# TRUTH = EODP-TS-L1B\input\ism_toa_isrf_<band>.nc
# ============================================================

def get_row_from_nc(file_path: Path, row_index: int,
                    variable_name: Optional[str]):
    with xr.open_dataset(file_path) as ds:

        if variable_name is None:
            # Variables que no son coordenadas
            data_vars = list(ds.data_vars)

            if len(data_vars) != 1:
                raise ValueError(
                    f"El fichero {file_path.name} tiene {len(data_vars)} "
                    f"variables de datos ({data_vars}); especifica "
                    f"VARIABLE_NAME manualmente."
                )

            variable_name = data_vars[0]

        data = ds[variable_name].values

        if data.ndim != 2:
            raise ValueError(
                f"La variable '{variable_name}' en {file_path.name} no es "
                f"2D (tiene forma {data.shape})."
            )

        if row_index >= data.shape[0]:
            raise IndexError(
                f"La fila {row_index} esta fuera de {file_path.name} "
                f"(forma {data.shape})."
            )

        return data[row_index, :]


def plot_equalization_effect_band(band: str) -> bool:
    """
    Genera el PNG ecualizada vs no ecualizada vs verdad para UNA banda.
    :param band: nombre de la banda, por ejemplo 'VNIR-0'
    :return: True si la grafica se ha generado, False si se ha saltado
    """
    file_eq = FOLDER_EQ / (L1B_TOA_PREFIX + band + ".nc")
    file_no_eq = FOLDER_NO_EQ / (L1B_TOA_PREFIX + band + ".nc")
    file_isrf = FOLDER_INPUT / (ISRF_TOA_PREFIX + band + ".nc")

    # Si falta algun fichero de esta banda, se avisa y se continua con las demas
    for file_path in (file_eq, file_no_eq, file_isrf):
        if not file_path.is_file():
            print(f"  AVISO [{band}]: no se encuentra {file_path}")
            return False

    try:
        row_eq = get_row_from_nc(file_eq, ROW_INDEX, VARIABLE_NAME)
        row_no_eq = get_row_from_nc(file_no_eq, ROW_INDEX, VARIABLE_NAME)
        row_isrf = get_row_from_nc(file_isrf, ROW_INDEX, VARIABLE_NAME)
    except (ValueError, IndexError, OSError) as error:
        print(f"  AVISO [{band}]: no se pudo leer la fila -> {error}")
        return False

    plt.figure(figsize=(10, 6))

    plt.plot(np.arange(row_eq.size), row_eq,
             color="black", label="TOA L1B with eq")
    plt.plot(np.arange(row_no_eq.size), row_no_eq,
             color="red", label="TOA L1B no eq")
    plt.plot(np.arange(row_isrf.size), row_isrf,
             color="blue", label="TOA after the ISRF")

    plt.title(f"Effect of the Equalization for {band}")
    plt.xlabel("ACT pixel [-]")
    plt.ylabel("TOA [mW/m2/sr]")
    plt.legend(loc="upper left")
    plt.grid(True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_FILENAME_TEMPLATE.format(band=band)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"  Grafica guardada en: {output_path}")
    return True


def plot_equalization_effect():
    """Genera el PNG del efecto de la ecualizacion para las 4 bandas."""

    print("=" * 80)
    print("PLOT DEL EFECTO DE LA ECUALIZACION - 4 BANDAS")
    print("=" * 80)
    print(f"Fila (ALT) utilizada: {ROW_INDEX}")
    print("=" * 80)

    generated = 0

    for band in BANDS:
        print(f"\n{band}")
        if plot_equalization_effect_band(band):
            generated += 1

    print("\n" + "=" * 80)
    print(f"Graficas generadas: {generated} de {len(BANDS)}")
    print("=" * 80)


# ============================================================
# EJECUCION
# ============================================================

if __name__ == "__main__":
    # Cross validation de las salidas L1B (todas las bandas)
    run_crossvalidation()

    print()

    # Plot del efecto de la ecualizacion (las 4 bandas)
    plot_equalization_effect()
