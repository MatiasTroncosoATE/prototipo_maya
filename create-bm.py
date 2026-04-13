"""
create_bm.py
=============
Construye un DataFrame "base de monitoreo" (BM) a partir de los CSVs
generados por simulate_yucatan_edu.py.

Cada fila representa un estudiante. Las columnas incluyen:
  - Info personal (de estudiantes.csv)
  - Por cada sesión:   sesion_N  → tipo de asistencia (str) o None
  - Por cada trabajo:  trabajo_N → 1 si entregado, 0 si no
  - Columnas derivadas de resumen: sesiones_asistidas, trabajos_entregados,
    pct_asistencia, pct_entrega, primera_actividad, ultima_actividad

Uso
---
    # Desde línea de comandos (usa defaults de data/)
    python create_bm.py

    # Importado
    from create_bm import build_base_monitoreo
    bm = build_base_monitoreo(
        students_path="data/estudiantes.csv",
        sesiones_path="data/sesiones_momento_2_final.csv",
        trabajos_path="data/trabajos_momento_2_final.csv",
    )

Dependencias
------------
    pip install polars
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def _pivot_sesiones(sesiones: pl.DataFrame) -> pl.DataFrame:
    """
    Convierte el DataFrame largo de sesiones en una columna por sesión.

    Entrada (una fila por evento):
        estudiante_id, sesion_num, tipo_asistencia, timestamp

    Salida (una fila por estudiante):
        estudiante_id,
        sesion_1,   ← valor: 'EN VIVO' | 'GRABACIÓN' | 'SIN ASISTENCIA' | null
        sesion_2,
        ...
    """
    sesion_nums = sorted(sesiones["sesion_num"].unique().to_list())

    # Cast sesion_num to String so pivot produces string column names
    wide = (
        sesiones
        .with_columns(pl.col("sesion_num").cast(pl.String))
        .pivot(
            on="sesion_num",
            index="estudiante_id",
            values="tipo_asistencia",
            aggregate_function="first",
        )
        .rename({str(n): f"sesion_{n}" for n in sesion_nums})
    )

    ordered_cols = ["estudiante_id"] + [f"sesion_{n}" for n in sesion_nums]
    existing = [c for c in ordered_cols if c in wide.columns]
    return wide.select(existing)


def _pivot_trabajos(trabajos: pl.DataFrame) -> pl.DataFrame:
    """
    Convierte el DataFrame largo de trabajos en una columna por trabajo.

    Entrada (una fila por evento):
        estudiante_id, trabajo_num, entregado, timestamp

    Salida (una fila por estudiante):
        estudiante_id,
        trabajo_1,   ← valor: 1 si entregado, 0 si no
        trabajo_2,
        ...
    """
    trabajo_nums = sorted(trabajos["trabajo_num"].unique().to_list())

    # Cast entregado → Int8 (1/0) y trabajo_num → String antes del pivot
    wide = (
        trabajos
        .with_columns(
            pl.col("entregado").cast(pl.Int8),
            pl.col("trabajo_num").cast(pl.String),
        )
        .pivot(
            on="trabajo_num",
            index="estudiante_id",
            values="entregado",
            aggregate_function="first",
        )
        .rename({str(n): f"trabajo_{n}" for n in trabajo_nums})
    )

    ordered_cols = ["estudiante_id"] + [f"trabajo_{n}" for n in trabajo_nums]
    existing = [c for c in ordered_cols if c in wide.columns]
    return wide.select(existing)


def _compute_summaries(
    sesiones: pl.DataFrame,
    trabajos: pl.DataFrame,
) -> pl.DataFrame:
    """
    Calcula columnas derivadas de resumen por estudiante.

    Retorna un DataFrame con:
        estudiante_id,
        sesiones_asistidas   (int)  — veces que el tipo != 'SIN ASISTENCIA'
        sesiones_en_vivo     (int)
        sesiones_grabacion   (int)
        trabajos_entregados  (int)
        pct_asistencia       (float) — sobre el total de sesiones disponibles
        pct_entrega          (float) — sobre el total de trabajos disponibles
        primera_actividad    (str)   — timestamp más antiguo entre todos los eventos
        ultima_actividad     (str)   — timestamp más reciente
    """
    n_sesiones = sesiones["sesion_num"].n_unique()
    n_trabajos = trabajos["trabajo_num"].n_unique()

    ses_summary = (
        sesiones.group_by("estudiante_id")
        .agg(
            (pl.col("tipo_asistencia") != "SIN ASISTENCIA")
            .sum()
            .alias("sesiones_asistidas"),
            (pl.col("tipo_asistencia") == "EN VIVO").sum().alias("sesiones_en_vivo"),
            (pl.col("tipo_asistencia") == "GRABACIÓN").sum().alias("sesiones_grabacion"),
            pl.col("timestamp").min().alias("_ses_primera"),
            pl.col("timestamp").max().alias("_ses_ultima"),
        )
        .with_columns(
            (pl.col("sesiones_asistidas") / n_sesiones * 100)
            .round(1)
            .alias("pct_asistencia")
        )
    )

    tra_summary = (
        trabajos.group_by("estudiante_id")
        .agg(
            pl.col("entregado").sum().alias("trabajos_entregados"),
            pl.col("timestamp").min().alias("_tra_primera"),
            pl.col("timestamp").max().alias("_tra_ultima"),
        )
        .with_columns(
            (pl.col("trabajos_entregados") / n_trabajos * 100)
            .round(1)
            .alias("pct_entrega")
        )
    )

    summary = (
        ses_summary.join(tra_summary, on="estudiante_id", how="full", coalesce=True)
        .with_columns(
            pl.min_horizontal("_ses_primera", "_tra_primera").alias("primera_actividad"),
            pl.max_horizontal("_ses_ultima", "_tra_ultima").alias("ultima_actividad"),
        )
        .drop("_ses_primera", "_ses_ultima", "_tra_primera", "_tra_ultima")
    )

    return summary


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def build_base_monitoreo(
    students_path: str | Path,
    sesiones_path: str | Path,
    trabajos_path: str | Path,
) -> pl.DataFrame:
    """
    Construye la base de monitoreo: un DataFrame ancho con una fila
    por estudiante y toda su información personal y de plataforma como columnas.

    Parámetros
    ----------
    students_path : str | Path
        Ruta al CSV de estudiantes generado por simulate_yucatan_edu.py.
    sesiones_path : str | Path
        Ruta al CSV de sesiones (ej. sesiones_momento_2_final.csv).
    trabajos_path : str | Path
        Ruta al CSV de trabajos (ej. trabajos_momento_2_final.csv).

    Retorna
    -------
    pl.DataFrame con columnas:
        [info personal] | [sesion_N (status str)] |
        [trabajo_N (1/0)] | [resúmenes]
    """
    # -- Carga --
    students = pl.read_csv(students_path)
    sesiones = pl.read_csv(sesiones_path)
    trabajos = pl.read_csv(trabajos_path)

    # Asegurar tipos
    sesiones = sesiones.with_columns(pl.col("estudiante_id").cast(pl.Int32))
    trabajos = trabajos.with_columns(
        pl.col("estudiante_id").cast(pl.Int32),
        pl.col("entregado").cast(pl.Boolean),
    )

    # -- Pivotes --
    wide_sesiones = _pivot_sesiones(sesiones)
    wide_trabajos = _pivot_trabajos(trabajos)
    summaries = _compute_summaries(sesiones, trabajos)

    # -- Join columnas naturales --
    bm = (
        students.rename({"id": "estudiante_id"})
        .join(wide_sesiones, on="estudiante_id", how="left")
        .join(wide_trabajos, on="estudiante_id", how="left")
        .join(summaries, on="estudiante_id", how="left")
        .rename({"estudiante_id": "id"})
    )

    # Orden de columnas: id primero, resúmenes al final
    summary_cols = [
        "sesiones_asistidas", "sesiones_en_vivo", "sesiones_grabacion",
        "trabajos_entregados", "pct_asistencia", "pct_entrega",
        "primera_actividad", "ultima_actividad",
    ]
    personal_cols = ["id", "name", "username", "email", "pregunta_1", "pregunta_2", "pregunta_3"]
    event_cols = [c for c in bm.columns if c not in personal_cols + summary_cols]

    final_order = personal_cols + event_cols + summary_cols
    return bm.select([c for c in final_order if c in bm.columns]).drop(['pregunta_1', 'pregunta_2', 'pregunta_3'])


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Paths always relative to this script, regardless of cwd
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    DATA_DIR.mkdir(exist_ok=True)

    momentos = [
        "momento_0_inicio",
        "momento_1_medio",
        "momento_2_final",
        "momento_2_optimista",
        "momento_2_pesimista",
        "momento_2_ordenado",
    ]

    for label in momentos:
        ses_path = DATA_DIR / f"sesiones_{label}.csv"
        tra_path = DATA_DIR / f"trabajos_{label}.csv"

        if not ses_path.exists() or not tra_path.exists():
            print(f"  [skip] Archivos no encontrados para '{label}'")
            continue

        print(f"Construyendo BM para {label}...")
        bm = build_base_monitoreo(
            students_path=DATA_DIR / "estudiantes.csv",
            sesiones_path=ses_path,
            trabajos_path=tra_path,
        )

        out_path = DATA_DIR / f"bm_{label}.csv"
        bm.write_csv(out_path)

        n_cols = len(bm.columns)
        print(f"  → {out_path}  ({len(bm)} filas × {n_cols} columnas)")
        print(f"     Columnas: {bm.columns[:6]} ... {bm.columns[-4:]}")
        print(f"     Asistencia media: {bm['pct_asistencia'].mean():.1f}%")
        print(f"     Entrega media:    {bm['pct_entrega'].mean():.1f}%")

    print("\n✓ Bases de monitoreo guardadas en data/")