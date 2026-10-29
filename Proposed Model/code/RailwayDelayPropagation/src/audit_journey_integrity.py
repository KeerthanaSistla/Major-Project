from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "data" / "raw" / "railway_delays_raw.parquet"
SEQ_FILE = ROOT / "data" / "processed" / "lstm_sequences.parquet"

con = duckdb.connect()

raw = f"read_parquet('{RAW_FILE.as_posix()}')"
seq = f"read_parquet('{SEQ_FILE.as_posix()}')"


def section(title):
    print(f"\n{'=' * 75}")
    print(title)
    print("=" * 75)


# ---------------------------------------------------------
# 1. Raw journey-key integrity
# ---------------------------------------------------------
section("1. RAW JOURNEY-KEY INTEGRITY")

query = f"""
WITH key_counts AS (
    SELECT
        train_number,
        start_date,
        seq,
        COUNT(*) AS n
    FROM {raw}
    GROUP BY train_number, start_date, seq
)
SELECT
    COUNT(*) AS distinct_station_keys,
    COUNT(*) FILTER (WHERE n > 1) AS duplicated_keys,
    SUM(n - 1) FILTER (WHERE n > 1) AS extra_duplicate_rows
FROM key_counts
"""

print(con.execute(query).fetchdf().to_string(index=False))


# ---------------------------------------------------------
# 2. Inspect train/date groups and station-sequence gaps
# ---------------------------------------------------------
section("2. RAW STATION-SEQUENCE GAPS")

query = f"""
WITH ordered AS (
    SELECT
        train_number,
        start_date,
        seq,
        LAG(seq) OVER (
            PARTITION BY train_number, start_date
            ORDER BY seq
        ) AS previous_seq
    FROM {raw}
),
journey_stats AS (
    SELECT
        train_number,
        start_date,
        COUNT(*) AS stops,
        MIN(seq) AS first_seq,
        MAX(seq) AS last_seq,
        COUNT(*) FILTER (
            WHERE previous_seq IS NOT NULL
              AND seq <> previous_seq + 1
        ) AS nonconsecutive_steps
    FROM ordered
    GROUP BY train_number, start_date
)
SELECT
    COUNT(*) AS train_date_groups,
    COUNT(*) FILTER (
        WHERE nonconsecutive_steps > 0
    ) AS groups_with_sequence_gaps,
    MAX(nonconsecutive_steps) AS max_gaps_in_one_group,
    MAX(stops) AS maximum_stops_in_group
FROM journey_stats
"""

print(con.execute(query).fetchdf().to_string(index=False))


# Show examples of gaps, if present.
query = f"""
WITH ordered AS (
    SELECT
        train_number,
        start_date,
        seq,
        LAG(seq) OVER (
            PARTITION BY train_number, start_date
            ORDER BY seq
        ) AS previous_seq
    FROM {raw}
)
SELECT
    train_number,
    start_date,
    previous_seq,
    seq
FROM ordered
WHERE previous_seq IS NOT NULL
  AND seq <> previous_seq + 1
LIMIT 20
"""

print("\nExample sequence gaps:")
print(con.execute(query).fetchdf().to_string(index=False))


# ---------------------------------------------------------
# 3. Scheduled and actual arrival timestamp order
# ---------------------------------------------------------
section("3. TIMESTAMP ORDER CHECKS")

query = f"""
WITH ordered AS (
    SELECT
        train_number,
        start_date,
        seq,
        scheduled_arrival,
        actual_arrival,

        LAG(scheduled_arrival) OVER (
            PARTITION BY train_number, start_date
            ORDER BY seq
        ) AS previous_scheduled_arrival,

        LAG(actual_arrival) OVER (
            PARTITION BY train_number, start_date
            ORDER BY seq
        ) AS previous_actual_arrival

    FROM {raw}
)
SELECT
    COUNT(*) AS rows_checked,

    COUNT(*) FILTER (
        WHERE scheduled_arrival IS NOT NULL
          AND previous_scheduled_arrival IS NOT NULL
          AND scheduled_arrival < previous_scheduled_arrival
    ) AS scheduled_arrival_reversals,

    COUNT(*) FILTER (
        WHERE actual_arrival IS NOT NULL
          AND previous_actual_arrival IS NOT NULL
          AND actual_arrival < previous_actual_arrival
    ) AS actual_arrival_reversals

FROM ordered
"""

print(con.execute(query).fetchdf().to_string(index=False))


# ---------------------------------------------------------
# 4. Verify target stations against raw data
# ---------------------------------------------------------
section("4. TARGET-STATION MATCH")

query = f"""
WITH matches AS (
    SELECT
        s.train_number,
        s.service_date,
        s.target_station_seq,
        s.target_station_code,
        s.target_delay_min,

        r.station_code AS raw_station_code,
        r.arrival_delay_min AS raw_delay

    FROM {seq} s

    LEFT JOIN {raw} r
      ON s.train_number = r.train_number
     AND s.service_date = r.start_date
     AND s.target_station_seq = r.seq

    WHERE hash(
        s.train_number,
        s.service_date,
        s.target_station_seq
    ) % 100 = 0
)
SELECT
    COUNT(*) AS sampled_sequences,

    COUNT(*) FILTER (
        WHERE raw_station_code IS NULL
    ) AS target_rows_not_found,

    COUNT(*) FILTER (
        WHERE raw_station_code IS NOT NULL
          AND target_station_code <> raw_station_code
    ) AS station_code_mismatches,

    COUNT(*) FILTER (
        WHERE raw_delay IS NULL
           OR target_delay_min IS NULL
    ) AS rows_with_missing_target_delay,

    COUNT(*) FILTER (
        WHERE raw_delay IS NOT NULL
          AND target_delay_min IS NOT NULL
          AND ABS(raw_delay - target_delay_min) > 0.001
    ) AS target_delay_mismatches

FROM matches
"""

print(con.execute(query).fetchdf().to_string(index=False))


# ---------------------------------------------------------
# 5. Compare previous delay features with raw records
# ---------------------------------------------------------
section("5. PREVIOUS-DELAY FEATURE MATCH")

query = f"""
WITH raw_ordered AS (
    SELECT
        train_number,
        start_date,
        seq,

        LAG(arrival_delay_min, 1) OVER (
            PARTITION BY train_number, start_date
            ORDER BY seq
        ) AS raw_delay_t1,

        LAG(arrival_delay_min, 2) OVER (
            PARTITION BY train_number, start_date
            ORDER BY seq
        ) AS raw_delay_t2,

        LAG(arrival_delay_min, 3) OVER (
            PARTITION BY train_number, start_date
            ORDER BY seq
        ) AS raw_delay_t3,

        LAG(arrival_delay_min, 4) OVER (
            PARTITION BY train_number, start_date
            ORDER BY seq
        ) AS raw_delay_t4,

        LAG(arrival_delay_min, 5) OVER (
            PARTITION BY train_number, start_date
            ORDER BY seq
        ) AS raw_delay_t5

    FROM {raw}
),
sample AS (
    SELECT *
    FROM {seq}
    WHERE hash(
        train_number,
        service_date,
        target_station_seq
    ) % 100 = 0
),
joined AS (
    SELECT
        s.*,
        r.raw_delay_t1,
        r.raw_delay_t2,
        r.raw_delay_t3,
        r.raw_delay_t4,
        r.raw_delay_t5

    FROM sample s

    LEFT JOIN raw_ordered r
      ON s.train_number = r.train_number
     AND s.service_date = r.start_date
     AND s.target_station_seq = r.seq
)
SELECT
    COUNT(*) AS sampled_rows,

    COUNT(*) FILTER (
        WHERE raw_delay_t1 IS NOT NULL
          AND ABS(delay_t1 - raw_delay_t1) > 0.001
    ) AS delay_t1_mismatches,

    COUNT(*) FILTER (
        WHERE raw_delay_t2 IS NOT NULL
          AND ABS(delay_t2 - raw_delay_t2) > 0.001
    ) AS delay_t2_mismatches,

    COUNT(*) FILTER (
        WHERE raw_delay_t3 IS NOT NULL
          AND ABS(delay_t3 - raw_delay_t3) > 0.001
    ) AS delay_t3_mismatches,

    COUNT(*) FILTER (
        WHERE raw_delay_t4 IS NOT NULL
          AND ABS(delay_t4 - raw_delay_t4) > 0.001
    ) AS delay_t4_mismatches,

    COUNT(*) FILTER (
        WHERE raw_delay_t5 IS NOT NULL
          AND ABS(delay_t5 - raw_delay_t5) > 0.001
    ) AS delay_t5_mismatches

FROM joined
"""

print(con.execute(query).fetchdf().to_string(index=False))

con.close()
print("\nJourney-integrity audit completed.")