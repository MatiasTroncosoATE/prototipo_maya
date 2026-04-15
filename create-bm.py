"""
create_bm.py
=============
Construye un DataFrame "base de monitoreo" (BM) a partir de los CSVs
generados por simulate_yucatan_edu.py.

Los CSVs de eventos contienen únicamente registros positivos:
  - sesiones: solo filas donde tipo_asistencia != 'SIN ASISTENCIA'
  - trabajos: solo filas donde entregado == True

La ausencia de registro se interpreta como no participación.

Uso
---
    python create_bm.py [--cutoff YYYY-MM-DD]

    # Importado
    from create_bm import build_base_monitoreo
    bm = build_base_monitoreo(
        students_path="data/estudiantes.csv",
        sesiones_path="data/sesiones_base.csv",
        trabajos_path="data/trabajos_base.csv",
        cutoff_date=date(2024, 2, 20)
    )
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import polars as pl


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def _pivot_sesiones(
    sesiones: pl.DataFrame,
    students_df: pl.DataFrame,
    max_sesion_num: int | None = None,
) -> pl.DataFrame:
    """
    Construye una columna por sesión para cada estudiante.

    Si un estudiante no tiene registro para una sesión, se asigna 'SIN ASISTENCIA'.
    """
    if sesiones.is_empty():
        # No hay eventos positivos, generamos todo como SIN ASISTENCIA
        students = students_df.select("id").rename({"id": "estudiante_id"})
        if max_sesion_num is None:
            max_sesion_num = 0
        # Crear columnas vacías para todas las sesiones esperadas
        for n in range(1, max_sesion_num + 1):
            students = students.with_columns(pl.lit("SIN ASISTENCIA").alias(f"sesion_{n}"))
        return students

    # Rango completo de sesiones
    if max_sesion_num is None:
        max_sesion_num = sesiones["sesion_num"].max()
    all_sessions = list(range(1, max_sesion_num + 1))

    # Cruzar estudiantes × sesiones para tener la cuadrícula completa
    students = students_df.select("id").rename({"id": "estudiante_id"})
    grid = students.join(
        pl.DataFrame({"sesion_num": all_sessions}).with_columns(pl.lit(1).alias("__join")),
        how="cross",
    ).drop("__join")

    # Unir eventos reales (left join)
    grid = grid.join(
        sesiones.select(["estudiante_id", "sesion_num", "tipo_asistencia"]),
        on=["estudiante_id", "sesion_num"],
        how="left",
    )

    # Rellenar ausencias con 'SIN ASISTENCIA'
    grid = grid.with_columns(
        pl.col("tipo_asistencia").fill_null("SIN ASISTENCIA")
    )

    # Pivotear
    wide = (
        grid.with_columns(pl.col("sesion_num").cast(pl.String))
        .pivot(
            on="sesion_num",
            index="estudiante_id",
            values="tipo_asistencia",
            aggregate_function="first",
        )
        .rename({str(n): f"sesion_{n}" for n in all_sessions})
    )

    # Ordenar columnas
    col_order = ["estudiante_id"] + [f"sesion_{n}" for n in all_sessions]
    return wide.select(col_order)


def _pivot_trabajos(
    trabajos: pl.DataFrame,
    students_df: pl.DataFrame,
    max_trabajo_num: int | None = None,
) -> pl.DataFrame:
    """
    Construye una columna por trabajo para cada estudiante.

    Si un estudiante no tiene registro de entrega, se asigna 0.
    """
    if trabajos.is_empty():
        students = students_df.select("id").rename({"id": "estudiante_id"})
        if max_trabajo_num is None:
            max_trabajo_num = 0
        for n in range(1, max_trabajo_num + 1):
            students = students.with_columns(pl.lit(0).cast(pl.Int8).alias(f"trabajo_{n}"))
        return students

    if max_trabajo_num is None:
        max_trabajo_num = trabajos["trabajo_num"].max()
    all_trabajos = list(range(1, max_trabajo_num + 1))

    students = students_df.select("id").rename({"id": "estudiante_id"})
    grid = students.join(
        pl.DataFrame({"trabajo_num": all_trabajos}).with_columns(pl.lit(1).alias("__join")),
        how="cross",
    ).drop("__join")

    # Los eventos positivos tienen entregado=True, pero en el CSV solo está esa columna
    grid = grid.join(
        trabajos.select(["estudiante_id", "trabajo_num"]).with_columns(pl.lit(1).alias("entregado")),
        on=["estudiante_id", "trabajo_num"],
        how="left",
    )

    grid = grid.with_columns(
        pl.col("entregado").fill_null(0).cast(pl.Int8)
    )

    wide = (
        grid.with_columns(pl.col("trabajo_num").cast(pl.String))
        .pivot(
            on="trabajo_num",
            index="estudiante_id",
            values="entregado",
            aggregate_function="first",
        )
        .rename({str(n): f"trabajo_{n}" for n in all_trabajos})
    )

    col_order = ["estudiante_id"] + [f"trabajo_{n}" for n in all_trabajos]
    return wide.select(col_order)


def _compute_summaries(
    sesiones: pl.DataFrame,
    trabajos: pl.DataFrame,
    total_sesiones: int,
    total_trabajos: int,
) -> pl.DataFrame:
    """
    Calcula columnas derivadas de resumen por estudiante.
    """
    # Agrupaciones sobre eventos reales (positivos)
    ses_summary = (
        sesiones.group_by("estudiante_id")
        .agg(
            pl.len().alias("sesiones_asistidas"),  # todas las filas aquí son positivas
            (pl.col("tipo_asistencia") == "EN VIVO").sum().alias("sesiones_en_vivo"),
            (pl.col("tipo_asistencia") == "GRABACIÓN").sum().alias("sesiones_grabacion"),
            pl.col("timestamp").min().alias("_ses_primera"),
            pl.col("timestamp").max().alias("_ses_ultima"),
        )
        .with_columns(
            (pl.col("sesiones_asistidas") / total_sesiones * 100)
            .round(1)
            .alias("pct_asistencia")
        )
    )

    tra_summary = (
        trabajos.group_by("estudiante_id")
        .agg(
            pl.len().alias("trabajos_entregados"),  # todas las filas son entregas
            pl.col("timestamp").min().alias("_tra_primera"),
            pl.col("timestamp").max().alias("_tra_ultima"),
        )
        .with_columns(
            (pl.col("trabajos_entregados") / total_trabajos * 100)
            .round(1)
            .alias("pct_entrega")
        )
    )

    # Unir summaries
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
    cutoff_date: Optional[date] = None,
    n_sessions: Optional[int] = None,
    n_trabajos: Optional[int] = None,
) -> pl.DataFrame:
    """
    Construye la base de monitoreo.

    Parámetros
    ----------
    students_path, sesiones_path, trabajos_path : rutas a los CSVs.
    cutoff_date : date, optional
        Solo eventos anteriores a esta fecha.
    n_sessions, n_trabajos : int, optional
        Número total de sesiones/trabajos del programa. Si no se da, se infiere
        de los datos (puede ser menor si se aplica cutoff).

    Retorna
    -------
    pl.DataFrame ancho con una fila por estudiante.
    """
    # -- Carga --
    students = pl.read_csv(students_path)
    sesiones = pl.read_csv(sesiones_path, try_parse_dates=True)
    trabajos = pl.read_csv(trabajos_path, try_parse_dates=True)

    # Quitamos preguntas de students
    students = students.select([c for c in students.columns if not c.startswith("pregunta")])

    # Convertir timestamp si es string
    if not sesiones.is_empty() and sesiones["timestamp"].dtype == pl.String:
        sesiones = sesiones.with_columns(pl.col("timestamp").str.to_datetime())
    if not trabajos.is_empty() and trabajos["timestamp"].dtype == pl.String:
        trabajos = trabajos.with_columns(pl.col("timestamp").str.to_datetime())

    # Aplicar cutoff
    if cutoff_date is not None:
        cutoff_dt = datetime.combine(cutoff_date, datetime.min.time())
        if not sesiones.is_empty():
            sesiones = sesiones.filter(pl.col("timestamp") < cutoff_dt)
        if not trabajos.is_empty():
            trabajos = trabajos.filter(pl.col("timestamp") < cutoff_dt)

    # Asegurar tipos
    sesiones = sesiones.with_columns(pl.col("estudiante_id").cast(pl.Int32))
    trabajos = trabajos.with_columns(pl.col("estudiante_id").cast(pl.Int32))

    # Determinar totales teóricos
    if n_sessions is None:
        # Si los datos tienen sesiones, tomamos el máximo número presente; si no, 0
        n_sessions = sesiones["sesion_num"].max() if not sesiones.is_empty() else 0
    if n_trabajos is None:
        n_trabajos = trabajos["trabajo_num"].max() if not trabajos.is_empty() else 0

    # Pivotes (rellenan cuadrícula completa)
    wide_sesiones = _pivot_sesiones(sesiones, students, max_sesion_num=n_sessions)
    wide_trabajos = _pivot_trabajos(trabajos, students, max_trabajo_num=n_trabajos)

    # Summaries
    summaries = _compute_summaries(sesiones, trabajos, n_sessions, n_trabajos)

    # Unir todo
    bm = (
        students.rename({"id": "estudiante_id"})
        .join(wide_sesiones, on="estudiante_id", how="left")
        .join(wide_trabajos, on="estudiante_id", how="left")
        .join(summaries, on="estudiante_id", how="left")
        .rename({"estudiante_id": "id"})
    )

    # Columnas de eventos (existentes tras el pivot)
    sesion_cols = [f"sesion_{i}" for i in range(1, n_sessions + 1)]
    trabajo_cols = [f"trabajo_{i}" for i in range(1, n_trabajos + 1)]

    # Métrica "complecion-min"
    asistencia_pct = pl.sum_horizontal(
        [pl.col(c).is_in(["EN VIVO", "GRABACIÓN"]).cast(pl.Float64) for c in sesion_cols]
    ) / n_sessions if n_sessions > 0 else pl.lit(0.0)
    entrega_pct = pl.sum_horizontal(
        [(pl.col(c) == 1).cast(pl.Float64) for c in trabajo_cols]
    ) / n_trabajos if n_trabajos > 0 else pl.lit(0.0)

    bm = bm.with_columns(
        (
            (pl.min_horizontal(asistencia_pct, pl.lit(1.0)) * 40.0)
            + (pl.min_horizontal(entrega_pct, pl.lit(1.0)) * 50.0)
            + (pl.col("encuesta_inicial") * 5.0)
            + (pl.col("encuesta_final") * 5.0)
        ).alias("complecion-min")
    )

    # Listas de pendientes (usando las columnas del DataFrame ancho)
    sesiones_no_vistas = pl.concat_list([
        pl.when(~pl.col(c).is_in(["EN VIVO", "GRABACIÓN"])).then(pl.lit(c))
        for c in sesion_cols
    ]).list.eval(pl.element().filter(pl.element().is_not_null()))

    trabajos_no_avalados = pl.concat_list([
        pl.when(pl.col(c) != 1).then(pl.lit(c))
        for c in trabajo_cols
    ]).list.eval(pl.element().filter(pl.element().is_not_null()))

    bm = bm.with_columns([
        sesiones_no_vistas.alias("sesiones no vistas"),
        trabajos_no_avalados.alias("trabajos no avalados"),
    ])

    # Orden final
    summary_cols = [
        "sesiones_asistidas", "sesiones_en_vivo", "sesiones_grabacion",
        "trabajos_entregados", "pct_asistencia", "pct_entrega",
        "ultima_actividad", "complecion-min", "sesiones no vistas", "trabajos no avalados"
    ]
    personal_cols = ["id", "name", "username", "email"]
    event_cols = [c for c in bm.columns if c not in personal_cols + summary_cols]

    final_order = personal_cols + event_cols + summary_cols
    return bm.select([c for c in final_order if c in bm.columns])


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=str, help="Fecha límite YYYY-MM-DD")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cutoff = None
    if args.cutoff:
        try:
            cutoff = datetime.strptime(args.cutoff, "%Y-%m-%d").date()
        except ValueError:
            print("Error: formato de fecha inválido. Use YYYY-MM-DD.")
            exit(1)

    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    DATA_DIR.mkdir(exist_ok=True)

    escenarios = ["base", "optimista", "pesimista", "ordenado"]

    for label in escenarios:
        ses_path = DATA_DIR / f"sesiones_{label}.csv"
        tra_path = DATA_DIR / f"trabajos_{label}.csv"

        # Fallback a nombres antiguos
        if not ses_path.exists() or not tra_path.exists():
            alt_label = f"momento_2_{label}" if label != "base" else "momento_2_final"
            ses_path = DATA_DIR / f"sesiones_{alt_label}.csv"
            tra_path = DATA_DIR / f"trabajos_{alt_label}.csv"
            if not ses_path.exists() or not tra_path.exists():
                print(f"  [skip] Archivos no encontrados para '{label}'")
                continue

        print(f"Construyendo BM para {label}...")
        bm = build_base_monitoreo(
            students_path=DATA_DIR / "estudiantes.csv",
            sesiones_path=ses_path,
            trabajos_path=tra_path,
            cutoff_date=cutoff,
        )

        suffix = f"_{args.cutoff}" if args.cutoff else ""
        out_path = DATA_DIR / f"bm_{label}{suffix}.csv"

        # Convertir listas a strings para CSV
        bm_csv = bm.with_columns(
            pl.col("sesiones no vistas").list.join(", "),
            pl.col("trabajos no avalados").list.join(", "),
        )
        bm_csv.write_csv(out_path)

        print(f"  → {out_path}  ({len(bm)} filas x {len(bm.columns)} columnas)")
        if "pct_asistencia" in bm.columns:
            print(f"     Asistencia media: {bm['pct_asistencia'].mean():.1f}%")
        if "pct_entrega" in bm.columns:
            print(f"     Entrega media:    {bm['pct_entrega'].mean():.1f}%")

    print("\n✓ Bases de monitoreo guardadas en data/")