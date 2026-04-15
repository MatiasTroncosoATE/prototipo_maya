"""
simulate_yucatan_edu.py
========================
Simulador de datos para prototipo de programa educativo en Yucatán.

Uso rápido:
    python simulate_yucatan_edu.py

Dependencias:
    pip install polars faker
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl
from faker import Faker

# ---------------------------------------------------------------------------
# Config global
# ---------------------------------------------------------------------------

faker_mx = Faker("es_MX")
Faker.seed(42)
random.seed(42)

# Ventana de tiempo por defecto del programa (8 semanas)
DEFAULT_DATE_RANGE = (datetime(2024, 2, 5, 9, 0, 0), datetime(2024, 3, 31, 20, 0, 0))

SESSION_OPTIONS = ["EN VIVO", "GRABACIÓN", "SIN ASISTENCIA"]


# ---------------------------------------------------------------------------
# 1. Generador de estudiantes
# ---------------------------------------------------------------------------


def generate_students(n_rows: int) -> pl.DataFrame:
    """
    Genera un DataFrame de estudiantes ficticios.

    Parámetros
    ----------
    n_rows : int
        Número de estudiantes a generar.

    Retorna
    -------
    pl.DataFrame con columnas:
        id, name, username, email,
        pregunta_1 (bool), pregunta_2 (str), pregunta_3 (int 1-5)
    """
    records = []
    for i in range(1, n_rows + 1):
        first = faker_mx.first_name()
        last = faker_mx.last_name()
        name = f"{first} {last}"

        # username: primera letra del nombre + apellido + número aleatorio
        base = (first[0] + last).lower().replace(" ", "")
        username = f"{base}{random.randint(10, 999)}"

        # email con dominios locales y comunes en MX
        domain = random.choice(
            ["gmail.com", "hotmail.com", "outlook.com", "uady.mx", "itmerida.mx"]
        )
        email = f"{username}@{domain}"

        records.append(
            {
                "id": i,
                "name": name,
                "username": username,
                "email": email,
                "pregunta_1": faker_mx.boolean(chance_of_getting_true=65),
                "pregunta_2": faker_mx.sentence(nb_words=random.randint(5, 12)),
                "pregunta_3": random.randint(1, 5),
                "encuesta_inicial": 1,
                "encuesta_final": 0,
            }
        )

    return pl.DataFrame(records).with_columns(
        pl.col("id").cast(pl.Int32),
        pl.col("pregunta_1").cast(pl.Boolean),
        pl.col("pregunta_3").cast(pl.Int32),
    )


# ---------------------------------------------------------------------------
# 2. Modelo de distribución de participación
# ---------------------------------------------------------------------------


def _generate_decaying_probs(n: int, start: float = 0.8, end: float = 0.25) -> list[float]:
    """
    Genera una lista de probabilidades que decaen exponencialmente de `start` a `end`
    a lo largo de `n` elementos.
    """
    if n <= 0:
        return []
    if n == 1:
        return [start]

    # Decaimiento exponencial: p[i] = end + (start - end) * exp(-k * i/(n-1))
    # Elegimos k tal que el último valor sea aproximadamente end.
    import math
    k = 3.0  # factor de caída (mayor k = caída más rápida)
    probs = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0
        p = end + (start - end) * math.exp(-k * t)
        probs.append(round(p, 4))
    return probs


@dataclass
class EngagementModel:
    """
    Encapsula las probabilidades de participación para sesiones y trabajos.

    Parámetros
    ----------
    n_sessions : int
        Número total de sesiones en el programa.
    n_trabajos : int
        Número total de trabajos/entregables.
    alpha : float
        Factor de escala que "afina" las distribuciones.
        > 1 aumenta participación, < 1 la reduce. Rango útil: 0.5 - 1.5.
    """

    n_sessions: int
    n_trabajos: int
    alpha: float = 1.0

    def __post_init__(self):
        # Generamos probabilidades base con caída realista
        self._base_session_probs = _generate_decaying_probs(
            self.n_sessions, start=0.80, end=0.27
        )
        self._base_trabajo_probs = _generate_decaying_probs(
            self.n_trabajos, start=0.60, end=0.28
        )

    def _clamp(self, p: float) -> float:
        return max(0.0, min(1.0, p * self.alpha))

    def session_prob(self, idx: int) -> float:
        return self._clamp(self._base_session_probs[idx])

    def trabajo_prob(self, idx: int) -> float:
        return self._clamp(self._base_trabajo_probs[idx])


# ---------------------------------------------------------------------------
# 3. Generador de eventos de plataforma
# ---------------------------------------------------------------------------


def _random_timestamp(
    event_idx: int,
    n_total: int,
    date_range: tuple[datetime, datetime],
    jitter_days: float = 2.0,
) -> datetime:
    """
    Genera un timestamp para un evento dentro de un rango de fechas.

    Los eventos más tempranos (event_idx bajo) se concentran en la primera
    mitad del rango; los tardíos en la segunda. El ruido (jitter) se escala
    proporcionalmente para no cruzar los límites del rango.

    Parámetros
    ----------
    event_idx : int
        Índice 0-based del evento dentro de la secuencia.
    n_total : int
        Total de eventos en la secuencia.
    date_range : tuple[datetime, datetime]
        (inicio, fin) del rango de fechas permitido.
    jitter_days : float
        Radio máximo de ruido aleatorio en días. Se recorta automáticamente
        para que el timestamp permanezca dentro del rango.
    """
    start, end = date_range
    if start >= end:
        raise ValueError("date_range[0] debe ser anterior a date_range[1].")

    span_secs = (end - start).total_seconds()
    fraction = event_idx / max(n_total - 1, 1)

    # Centro del slot temporal para este evento
    base_ts = start + timedelta(seconds=span_secs * fraction)

    # Jitter acotado: no salir más allá del ancho del slot
    slot_half = timedelta(seconds=span_secs / max(n_total, 1) / 2)
    max_jitter = min(timedelta(days=jitter_days), slot_half)
    jitter = timedelta(
        seconds=random.uniform(-max_jitter.total_seconds(), max_jitter.total_seconds())
    )
    ts = base_ts + jitter

    # Clamp final por seguridad
    return max(start, min(end, ts))


def _session_asistencia(prob_asistir: float) -> str:
    """
    Elige el tipo de asistencia dado que hay una sesión programada.
    Si no asiste → 'SIN ASISTENCIA'.
    Si asiste   → 60 % EN VIVO, 40 % GRABACIÓN.
    """
    if random.random() > prob_asistir:
        return "SIN ASISTENCIA"
    return random.choices(["EN VIVO", "GRABACIÓN"], weights=[0.6, 0.4])[0]


def generate_platform_events(
    students_df: pl.DataFrame,
    n_sessions: int,
    n_trabajos: int,
    alpha: float = 1.0,
    date_range: tuple[datetime, datetime] | None = None,
    jitter_days: float = 2.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Genera dos DataFrames de eventos de plataforma.

    Parámetros
    ----------
    students_df : pl.DataFrame
        DataFrame producido por `generate_students`.
    n_sessions : int
        Número total de sesiones en el programa.
    n_trabajos : int
        Número total de trabajos/entregables.
    alpha : float
        Factor de ajuste de distribuciones (0.5 - 1.5 es el rango útil).
    date_range : tuple[datetime, datetime] | None
        (inicio, fin) del rango de fechas para los timestamps.
        Si es None se usa DEFAULT_DATE_RANGE.
        Los eventos se distribuyen linealmente dentro del rango: sesión/trabajo 1
        tiende a caer al inicio y los últimos al final. El ruido entre eventos
        se controla con jitter_days.
    jitter_days : float
        Radio máximo de ruido aleatorio en días alrededor del slot de cada
        evento. Valores bajos (0.5) mantienen timestamps estrictamente
        ordenados; valores altos (5+) los mezclan más. Default: 2.0.

    Retorna
    -------
    (sesiones_df, trabajos_df) : tuple[pl.DataFrame, pl.DataFrame]

    sesiones_df columnas:
        evento_id, estudiante_id, sesion_num, tipo_asistencia (str), timestamp

    trabajos_df columnas:
        evento_id, estudiante_id, trabajo_num, entregado (bool), timestamp
    """
    dr = date_range if date_range is not None else DEFAULT_DATE_RANGE
    model = EngagementModel(
        n_sessions=n_sessions, n_trabajos=n_trabajos, alpha=alpha
    )
    student_ids: list[int] = students_df["id"].to_list()

    sesiones_rows: list[dict] = []
    trabajos_rows: list[dict] = []

    for sid in student_ids:
        # --- Sesiones (solo positivas) ---
        for s_idx in range(model.n_sessions):
            tipo = _session_asistencia(model.session_prob(s_idx))
            if tipo != "SIN ASISTENCIA":  # <-- Solo escribimos si asistió
                ts = _random_timestamp(s_idx, model.n_sessions, dr, jitter_days)
                sesiones_rows.append(
                    {
                        "evento_id": str(uuid.uuid4()),
                        "estudiante_id": sid,
                        "sesion_num": s_idx + 1,
                        "tipo_asistencia": tipo,
                        "timestamp": ts.isoformat(),
                    }
                )

        # --- Trabajos (solo entregados) ---
        for t_idx in range(model.n_trabajos):
            entregado = random.random() < model.trabajo_prob(t_idx)
            if entregado:  # <-- Solo escribimos si entregado
                mapped_idx = int(
                    t_idx * (model.n_sessions - 1) / max(model.n_trabajos - 1, 1)
                )
                ts = _random_timestamp(mapped_idx, model.n_sessions, dr, jitter_days)
                trabajos_rows.append(
                    {
                        "evento_id": str(uuid.uuid4()),
                        "estudiante_id": sid,
                        "trabajo_num": t_idx + 1,
                        "entregado": True,
                        "timestamp": ts.isoformat(),
                    }
                )

    sesiones_df = pl.DataFrame(sesiones_rows).with_columns(
        pl.col("estudiante_id").cast(pl.Int32),
        pl.col("sesion_num").cast(pl.Int32),
    )

    trabajos_df = pl.DataFrame(trabajos_rows).with_columns(
        pl.col("estudiante_id").cast(pl.Int32),
        pl.col("trabajo_num").cast(pl.Int32),
        pl.col("entregado").cast(pl.Boolean),
    )

    return sesiones_df, trabajos_df


# ---------------------------------------------------------------------------
# 4. Punto de entrada
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import os

    os.makedirs("data", exist_ok=True)

    # -- Parámetros de la campaña (ahora flexibles) --
    N_SESSIONS = 10      # Número total de sesiones
    N_TRABAJOS = 10      # Número total de trabajos

    # -- Estudiantes --
    print("Generando estudiantes...")
    students = generate_students(n_rows=120)
    students.write_csv("data/estudiantes.csv")
    print(f"  → data/estudiantes.csv  ({len(students)} filas)")

    # Rango de fechas del programa (puede cambiarse libremente)
    program_dates = (datetime(2024, 2, 5, 9, 0), datetime(2024, 3, 31, 20, 0))

    # -- Escenarios de simulación (alpha y jitter variados) --
    configs = [
        # (alpha, jitter_days, etiqueta)
        (1.0, 2.0, "base"),
        (1.3, 2.0, "optimista"),   # alpha alto → más participación
        (0.7, 2.0, "pesimista"),   # alpha bajo → menos participación
        (1.0, 0.5, "ordenado"),    # jitter bajo → timestamps más estrictos
    ]

    for alpha, jitter, label in configs:
        print(f"\nSimulando escenario — {label} (alpha={alpha}, jitter={jitter}d)...")
        sesiones, trabajos = generate_platform_events(
            students,
            n_sessions=N_SESSIONS,
            n_trabajos=N_TRABAJOS,
            alpha=alpha,
            date_range=program_dates,
            jitter_days=jitter,
        )

        ses_path = f"data/sesiones_{label}.csv"
        tra_path = f"data/trabajos_{label}.csv"
        sesiones.write_csv(ses_path)
        trabajos.write_csv(tra_path)

        print(f"  → {ses_path}  ({len(sesiones)} filas)")
        print(f"  → {tra_path}  ({len(trabajos)} filas)")

        # Mini-reporte de distribución
        n_students = len(students)
        asistencia_pct = (
            sesiones.filter(pl.col("tipo_asistencia") != "SIN ASISTENCIA")
            .group_by("sesion_num")
            .agg(pl.len().alias("cnt"))
            .with_columns((pl.col("cnt") / n_students * 100).round(1).alias("pct"))
            .sort("sesion_num")
        )
        entrega_pct = (
            trabajos.group_by("trabajo_num")
            .agg(pl.col("entregado").sum().alias("cnt"))
            .with_columns((pl.col("cnt") / n_students * 100).round(1).alias("pct"))
            .sort("trabajo_num")
        )
        print("  Asistencia por sesión (%):")
        for row in asistencia_pct.iter_rows(named=True):
            bar = "█" * int(row["pct"] / 5)
            print(f"    Sesión {row['sesion_num']:>2}: {row['pct']:>5.1f}%  {bar}")
        print("  Entrega por trabajo (%):")
        for row in entrega_pct.iter_rows(named=True):
            bar = "█" * int(row["pct"] / 5)
            print(f"    Trabajo {row['trabajo_num']:>2}: {row['pct']:>5.1f}%  {bar}")

    print("\n✓ Todos los archivos guardados en data/")