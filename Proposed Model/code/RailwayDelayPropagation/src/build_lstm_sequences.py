from pathlib import Path

import duckdb


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "data" / "raw" / "railway_delays_raw.parquet"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "lstm_sequences.parquet"

# Number of previous observed stations used to predict
# arrival delay at the current station.
SEQUENCE_LENGTH = 5

# Escape the output path before inserting it into SQL.
escaped_output = (
    str(OUTPUT_FILE)
    .replace("\\", "/")
    .replace("'", "''")
)


# =========================================================
# INITIALIZATION
# =========================================================

print("=" * 65)
print("BUILDING LSTM SEQUENCE DATASET")
print("=" * 65)

if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"Raw dataset not found: {RAW_FILE}\n"
        "Run src/download_and_prepare.py first."
    )

print(f"Input:  {RAW_FILE}")
print(f"Output: {OUTPUT_FILE}")
print(f"Sequence length: {SEQUENCE_LENGTH} previous stations")

con = duckdb.connect()

try:
    con.execute("SET threads TO 4")

    # =====================================================
    # BUILD THE SEQUENCE DATASET
    # =====================================================

    query = f"""
    COPY (

        WITH raw_stops AS (

            SELECT
                CAST(train_number AS VARCHAR) AS train_number,

                TRY_CAST(start_date AS DATE) AS service_date,

                CAST(station_code AS VARCHAR) AS station_code,

                TRY_CAST(seq AS INTEGER) AS station_seq,

                TRY_CAST(
                    scheduled_arrival AS TIMESTAMP
                ) AS scheduled_arrival,

                TRY_CAST(
                    scheduled_departure AS TIMESTAMP
                ) AS scheduled_departure,

                TRY_CAST(
                    arrival_delay_min AS DOUBLE
                ) AS arrival_delay_min

            FROM read_parquet(?)

            WHERE train_number IS NOT NULL
              AND start_date IS NOT NULL
              AND station_code IS NOT NULL
              AND seq IS NOT NULL
              AND scheduled_arrival IS NOT NULL
              AND arrival_delay_min IS NOT NULL

        ),

        -- -------------------------------------------------
        -- 1. Remove duplicate journey-stop records.
        -- -------------------------------------------------

        base_stops AS (

            SELECT
                train_number,
                service_date,
                station_code,
                station_seq,
                scheduled_arrival,
                scheduled_departure,
                arrival_delay_min

            FROM (

                SELECT
                    *,

                    ROW_NUMBER() OVER (
                        PARTITION BY
                            train_number,
                            service_date,
                            station_seq

                        ORDER BY scheduled_arrival
                    ) AS duplicate_rank

                FROM raw_stops

                WHERE service_date IS NOT NULL
                  AND station_seq IS NOT NULL

            )

            WHERE duplicate_rank = 1

        ),

        -- -------------------------------------------------
        -- 2. Obtain the previous scheduled departure.
        -- -------------------------------------------------

        previous_schedule AS (

            SELECT
                *,

                LAG(scheduled_departure) OVER (
                    PARTITION BY
                        train_number,
                        service_date

                    ORDER BY station_seq
                ) AS previous_scheduled_departure

            FROM base_stops

        ),

        -- -------------------------------------------------
        -- 3. Calculate input features.
        -- -------------------------------------------------

        features AS (

            SELECT
                train_number,
                service_date,
                station_code,
                station_seq,
                arrival_delay_min,

                -- Scheduled dwell time at the current stop.
                COALESCE(
                    GREATEST(
                        DATE_DIFF(
                            'second',
                            scheduled_arrival,
                            scheduled_departure
                        ) / 60.0,
                        0.0
                    ),
                    0.0
                ) AS scheduled_dwell_min,

                -- Scheduled travel time from the previous stop.
                CASE
                    WHEN previous_scheduled_departure IS NOT NULL
                    THEN GREATEST(
                        DATE_DIFF(
                            'second',
                            previous_scheduled_departure,
                            scheduled_arrival
                        ) / 60.0,
                        0.0
                    )
                    ELSE 0.0
                END AS scheduled_travel_min,

                -- Cyclic encoding of scheduled arrival time.
                SIN(
                    2 * PI() *
                    (
                        EXTRACT(
                            HOUR FROM scheduled_arrival
                        )
                        +
                        EXTRACT(
                            MINUTE FROM scheduled_arrival
                        ) / 60.0
                    ) / 24.0
                ) AS arrival_hour_sin,

                COS(
                    2 * PI() *
                    (
                        EXTRACT(
                            HOUR FROM scheduled_arrival
                        )
                        +
                        EXTRACT(
                            MINUTE FROM scheduled_arrival
                        ) / 60.0
                    ) / 24.0
                ) AS arrival_hour_cos

            FROM previous_schedule

        ),

        -- -------------------------------------------------
        -- 4. Build historical station windows.
        --
        -- Each target row uses the previous five observed
        -- stops from the SAME train journey.
        --
        -- Feature order for each historical stop:
        --   1. Arrival delay
        --   2. Scheduled dwell time
        --   3. Scheduled travel time
        --   4. Arrival hour sine
        --   5. Arrival hour cosine
        -- -------------------------------------------------

        windowed AS (

            SELECT
                train_number,
                service_date,

                station_code AS target_station_code,
                station_seq AS target_station_seq,

                arrival_delay_min AS target_delay_min,

                LAG(arrival_delay_min, 5) OVER journey
                    AS delay_t5,

                LAG(scheduled_dwell_min, 5) OVER journey
                    AS dwell_t5,

                LAG(scheduled_travel_min, 5) OVER journey
                    AS travel_t5,

                LAG(arrival_hour_sin, 5) OVER journey
                    AS hour_sin_t5,

                LAG(arrival_hour_cos, 5) OVER journey
                    AS hour_cos_t5,


                LAG(arrival_delay_min, 4) OVER journey
                    AS delay_t4,

                LAG(scheduled_dwell_min, 4) OVER journey
                    AS dwell_t4,

                LAG(scheduled_travel_min, 4) OVER journey
                    AS travel_t4,

                LAG(arrival_hour_sin, 4) OVER journey
                    AS hour_sin_t4,

                LAG(arrival_hour_cos, 4) OVER journey
                    AS hour_cos_t4,


                LAG(arrival_delay_min, 3) OVER journey
                    AS delay_t3,

                LAG(scheduled_dwell_min, 3) OVER journey
                    AS dwell_t3,

                LAG(scheduled_travel_min, 3) OVER journey
                    AS travel_t3,

                LAG(arrival_hour_sin, 3) OVER journey
                    AS hour_sin_t3,

                LAG(arrival_hour_cos, 3) OVER journey
                    AS hour_cos_t3,


                LAG(arrival_delay_min, 2) OVER journey
                    AS delay_t2,

                LAG(scheduled_dwell_min, 2) OVER journey
                    AS dwell_t2,

                LAG(scheduled_travel_min, 2) OVER journey
                    AS travel_t2,

                LAG(arrival_hour_sin, 2) OVER journey
                    AS hour_sin_t2,

                LAG(arrival_hour_cos, 2) OVER journey
                    AS hour_cos_t2,


                LAG(arrival_delay_min, 1) OVER journey
                    AS delay_t1,

                LAG(scheduled_dwell_min, 1) OVER journey
                    AS dwell_t1,

                LAG(scheduled_travel_min, 1) OVER journey
                    AS travel_t1,

                LAG(arrival_hour_sin, 1) OVER journey
                    AS hour_sin_t1,

                LAG(arrival_hour_cos, 1) OVER journey
                    AS hour_cos_t1

            FROM features

            WINDOW journey AS (
                PARTITION BY
                    train_number,
                    service_date

                ORDER BY station_seq
            )

        )

        -- -------------------------------------------------
        -- 5. Retain complete sequences only.
        -- -------------------------------------------------

        SELECT *

        FROM windowed

        WHERE target_delay_min IS NOT NULL

          AND delay_t5 IS NOT NULL
          AND delay_t4 IS NOT NULL
          AND delay_t3 IS NOT NULL
          AND delay_t2 IS NOT NULL
          AND delay_t1 IS NOT NULL

    ) TO '{escaped_output}'

    (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    print("\nGenerating sequences...")

    con.execute(query, [str(RAW_FILE)])

    print("Sequence generation complete.")

    # =====================================================
    # INSPECT THE GENERATED DATASET
    # =====================================================

    summary = con.execute(
        """
        SELECT
            COUNT(*) AS sequence_samples,

            COUNT(DISTINCT train_number)
                AS distinct_trains,

            COUNT(DISTINCT service_date)
                AS distinct_service_dates,

            COUNT(DISTINCT target_station_code)
                AS target_stations,

            MIN(service_date)
                AS first_service_date,

            MAX(service_date)
                AS last_service_date,

            AVG(target_delay_min)
                AS mean_target_delay_min,

            MEDIAN(target_delay_min)
                AS median_target_delay_min,

            COUNT(*) FILTER (
                WHERE target_delay_min < 0
            ) AS early_arrivals,

            COUNT(*) FILTER (
                WHERE target_delay_min = 0
            ) AS on_time_arrivals,

            COUNT(*) FILTER (
                WHERE target_delay_min > 0
            ) AS late_arrivals

        FROM read_parquet(?)
        """,
        [str(OUTPUT_FILE)]
    ).df()

    print("\n========== SEQUENCE DATASET SUMMARY ==========")
    print(summary.to_string(index=False))

    # =====================================================
    # SAMPLE RECORDS
    # =====================================================

    print("\n========== SAMPLE RECORDS ==========")

    sample = con.execute(
        """
        SELECT *
        FROM read_parquet(?)
        LIMIT 3
        """,
        [str(OUTPUT_FILE)]
    ).df()

    print(sample.to_string(index=False))

    # =====================================================
    # SAVE SUMMARY
    # =====================================================

    summary_path = OUTPUT_DIR / "lstm_sequence_summary.csv"

    summary.to_csv(summary_path, index=False)

    print("\n========== COMPLETED ==========")
    print(f"Sequence dataset: {OUTPUT_FILE}")
    print(f"Summary report:   {summary_path}")

finally:
    con.close()