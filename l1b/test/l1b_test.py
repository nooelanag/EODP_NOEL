# CROSS VALIDATE L1B OUTPUTS EQUALIZED
from pathlib import Path
import numpy as np
import xarray as xr

# ============================================================
# CONFIGURACIÓN
# ============================================================

# Carpetas que contienen los ficheros .nc
FOLDER_1 = Path(
    r"C:\Users\Noel\Documents\NOEL\UNIVERSIDAD\MASTER\TELECO\ACADEMICO\26-27\1er cuatri\EODP\FILES\EODP_TER_2021\EODP_TER_2021\EODP-TS-L1B\output_noel"
)

FOLDER_2 = Path(
    r"C:\Users\Noel\Documents\NOEL\UNIVERSIDAD\MASTER\TELECO\ACADEMICO\26-27\1er cuatri\EODP\FILES\EODP_TER_2021\EODP_TER_2021\EODP-TS-L1B\output"
)

# Tolerancia para considerar que dos valores son iguales
RTOL = 1e-5
ATOL = 1e-8


# ============================================================
# FUNCIONES
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


# ============================================================
# CROSS VALIDATION
# ============================================================

def main():

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

    print("=" * 80)
    print("CROSS VALIDATION DE FICHEROS NETCDF")
    print("=" * 80)

    print(f"Ficheros .nc en carpeta 1: {len(files_1)}")
    print(f"Ficheros .nc en carpeta 2: {len(files_2)}")
    print(f"Ficheros encontrados en ambas carpetas: {len(common_files)}")
    print("=" * 80)

    if not common_files:
        print("No se han encontrado ficheros .nc con el mismo nombre en ambas carpetas.")
        return

    equal_count = 0
    different_count = 0

    # Comparar cada pareja de ficheros con el mismo nombre
    for filename in common_files:

        file1 = files_1[filename]
        file2 = files_2[filename]

        print(f"\n{filename}")

        are_equal, max_difference = compare_netcdf_files(file1, file2)

        if are_equal:
            equal_count += 1
            print(f"  IGUALES     | Diferencia máxima: {max_difference:.12e}")
        else:
            different_count += 1

            if np.isfinite(max_difference):
                print(f"  DIFERENTES  | Diferencia máxima: {max_difference:.12e}")
            else:
                print("  DIFERENTES  | No se puede calcular la diferencia máxima")

    # ========================================================
    # RESUMEN
    # ========================================================

    print("\n" + "=" * 80)
    print("RESUMEN")
    print("=" * 80)
    print(f"Ficheros comparados: {len(common_files)}")
    print(f"Iguales:             {equal_count}")
    print(f"Diferentes:          {different_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()


# PLOT FROM YOUR OUTPUTS THE EQUALISED OUTPUT VERSUS NOT EQUALISED VERSUS THE TRUTH
# TRUTH = EODP-TS-L1B\input\ism_toa_isrf_VNIR-0.nc
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACIÓN
# ============================================================

# Fichero SIN equalización (en rojo)
NO_EQ_FILE = Path(
    r"C:\Users\Noel\Documents\NOEL\UNIVERSIDAD\MASTER\TELECO\ACADEMICO\26-27\1er cuatri\EODP\FILES\EODP_TER_2021\EODP_TER_2021\EODP-TS-L1B\output_noel_no_equalized\l1b_toa_VNIR-0.nc"
)

# Fichero CON equalización (en negro)
EQ_FILE = Path(
    r"C:\Users\Noel\Documents\NOEL\UNIVERSIDAD\MASTER\TELECO\ACADEMICO\26-27\1er cuatri\EODP\FILES\EODP_TER_2021\EODP_TER_2021\EODP-TS-L1B\output_noel\l1b_toa_VNIR-0.nc"
)

# Fichero de ENTRADA / ISRF (en azul)
ISRF_FILE = Path(
    r"C:\Users\Noel\Documents\NOEL\UNIVERSIDAD\MASTER\TELECO\ACADEMICO\26-27\1er cuatri\EODP\FILES\EODP_TER_2021\EODP_TER_2021\EODP-TS-L1B\input\ism_toa_isrf_VNIR-0.nc"
)

# Carpeta y nombre del PNG de salida
OUTPUT_DIR = Path(
    r"C:\Users\Noel\Documents\NOEL\UNIVERSIDAD\MASTER\TELECO\ACADEMICO\26-27\1er cuatri\EODP\FILES\EODP_TER_2021\EODP_TER_2021\EODP-TS-L1B"
)
OUTPUT_FILENAME = "equalization_effect_VNIR-0.png"

# Fila (row) a extraer de cada fichero
ROW_INDEX = 50

# Nombre de la variable a leer dentro del .nc.
# Si se deja en None, se usa automáticamente la única variable de datos
# (no-coordenada) que tenga el fichero.
VARIABLE_NAME = None

# Nombre de la banda, usado solo para el título de la gráfica
BAND_NAME = "VNIR-0"


# ============================================================
# FUNCIONES
# ============================================================

def get_row_from_nc(file_path: Path, row_index: int, variable_name: str | None):
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

        return data[row_index, :]


# ============================================================
# GENERAR GRÁFICA
# ============================================================

def main():

    row_no_eq = get_row_from_nc(NO_EQ_FILE, ROW_INDEX, VARIABLE_NAME)
    row_eq = get_row_from_nc(EQ_FILE, ROW_INDEX, VARIABLE_NAME)
    row_isrf = get_row_from_nc(ISRF_FILE, ROW_INDEX, VARIABLE_NAME)

    act_pixels_no_eq = np.arange(row_no_eq.size)
    act_pixels_eq = np.arange(row_eq.size)
    act_pixels_isrf = np.arange(row_isrf.size)

    plt.figure(figsize=(10, 6))

    plt.plot(act_pixels_eq, row_eq, color="black", label="TOA L1B with eq")
    plt.plot(act_pixels_no_eq, row_no_eq, color="red", label="TOA L1B no eq")
    plt.plot(act_pixels_isrf, row_isrf, color="blue", label="TOA after the ISRF")

    plt.title(f"Effect of the Equalization for {BAND_NAME}")
    plt.xlabel("ACT pixel [-]")
    plt.ylabel("TOA [mW/m2/sr]")
    plt.legend(loc="upper left")
    plt.grid(True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_FILENAME

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Gráfica guardada en: {output_path}")


if __name__ == "__main__":
    main()

