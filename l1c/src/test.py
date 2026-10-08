import numpy as np
import matplotlib.pyplot as plt

EARTH_RADIUS_M = 6371000.0  # radio medio de la Tierra [m]


def haversine(lat1, lon1, lat2, lon2):
    """
    Distancia sobre la esfera entre puntos (lat, lon) en grados.
    Admite escalares o arrays (operación elemento a elemento).
    :return: distancia en metros
    """
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def plot_l1b_vs_l1c(lat_l1b, lon_l1b, lat_l1c, lon_l1c, output_dir=None):
    """
    Plot 1: rejilla L1B (rojo) vs rejilla L1C (azul).
    Plot 2: distancia de muestreo espacial (Haversine) de la fila central L1B.
    :param lat_l1b, lon_l1b: arrays 2D (ALT x ACT) en grados
    :param lat_l1c, lon_l1c: arrays 1D en grados
    :param output_dir: si no es None, guarda los PNG en esa carpeta
    """

    # ---------------- Plot 1: L1B (rojo) vs L1C (azul) ----------------
    plt.figure(figsize=(8, 8))
    plt.scatter(lon_l1b.ravel(), lat_l1b.ravel(), s=4, color="red", label="L1B grid")
    plt.scatter(lon_l1c, lat_l1c, s=4, color="blue", label="L1C grid")
    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.title("L1B grid vs L1C grid")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(f"{output_dir}/l1b_vs_l1c_grid.png", dpi=150)

    # ---------- Plot 2: distancia de muestreo, fila central L1B ----------
    central_row = lat_l1b.shape[0] // 2
    lat_row = lat_l1b[central_row, :]
    lon_row = lon_l1b[central_row, :]

    # Distancia entre píxeles ACT consecutivos
    distance = haversine(lat_row[:-1], lon_row[:-1], lat_row[1:], lon_row[1:])

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(distance.size), distance, color="black")
    plt.xlabel("ACT pixel [-]")
    plt.ylabel("Spatial sampling distance [m]")
    plt.title(f"Spatial sampling distance (Haversine), central row {central_row} of L1B")
    plt.grid(True)
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(f"{output_dir}/l1b_sampling_distance.png", dpi=150)

    print(f"Distancia de muestreo (fila {central_row}): "
          f"media = {np.mean(distance):.3f} m, "
          f"min = {np.min(distance):.3f} m, max = {np.max(distance):.3f} m")

    plt.show()


# Ejemplo de uso (tras llamar a l1cProjtoa):
# lat_l1c, lon_l1c, toa_l1c = l1c.l1cProjtoa(lat, lon, toa, band)
# plot_l1b_vs_l1c(lat, lon, lat_l1c, lon_l1c, output_dir=r"C:\ruta\salida")