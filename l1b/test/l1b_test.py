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
    """
    Compara dos ficheros NetCDF variable por variable y devuelve:
        - True/False indicando si son iguales dentro de la tolerancia.
        - La diferencia máxima absoluta encontrada.
    """

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


