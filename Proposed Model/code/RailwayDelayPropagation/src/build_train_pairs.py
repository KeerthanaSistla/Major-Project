
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "railway_delays_raw.parquet"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

PROCESSED.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

PAIR_FILE = PROCESSED / "candidate_train_pairs.parquet"

con = duckdb.connect()
con.execute("SET threads TO 4")

print("Building candidate train-to-train interactions...")
print("Stations: 30 busiest by observed arrival-delay records")
print("Scheduled arrival gap: 1 to 60 minutes")
print("Actual arrival order: A must arrive before B")

query = """
WITH top_stations AS (
    SELECT station_code
    FROM read_parquet(?)
    WHERE actual_arrival IS NOT NULL
      AND scheduled_arrival IS NOT NULL
      AND arrival_delay_min IS NOT NULL
    GROUP BY station_code
    ORDER BY COUNT(*) DESC
    LIMIT 30
),

arrivals AS (
    SELECT
        train_number,
        start_date,
        station_code,
        CAST(actual_arrival AS DATE) AS arrival_day,
        scheduled_arrival,
        actual_arrival,
        arrival_delay_min
    FROM read_parquet(?)
    WHERE actual_arrival IS NOT NULL
      AND scheduled_arrival IS NOT NULL
      AND arrival_delay_min IS NOT NULL
      AND station_code IN (
          SELECT station_code FROM top_stations
      )
),

pairs AS (
    SELECT
        a.station_code,
        a.arrival_day,

        a.train_number AS earlier_train_number,
        a.start_date AS earlier_train_start_date,
        b.train_number AS later_train_number,
        b.start_date AS later_train_start_date,

        a.scheduled_arrival AS earlier_scheduled_arrival,
        b.scheduled_arrival AS later_scheduled_arrival,
        a.actual_arrival AS earlier_actual_arrival,
        b.actual_arrival AS later_actual_arrival,

        a.arrival_delay_min AS earlier_delay_min,
        b.arrival_delay_min AS later_delay_min,

        DATE_DIFF(
            'second',
            a.scheduled_arrival,
            b.scheduled_arrival
        ) / 60.0 AS scheduled_gap_min,

        DATE_DIFF(
            'second',
            a.actual_arrival,
            b.actual_arrival
        ) / 60.0 AS actual_gap_min

    FROM arrivals a
    JOIN arrivals b
      ON a.station_code = b.station_code
     AND a.arrival_day = b.arrival_day
     AND a.scheduled_arrival < b.scheduled_arrival
     AND a.actual_arrival < b.actual_arrival

     -- A and B must be distinct train journeys.
     AND (
         a.train_number <> b.train_number
         OR a.start_date <> b.start_date
     )

     -- Scheduled separation: at most one hour.
     AND DATE_DIFF(
         'second',
         a.scheduled_arrival,
         b.scheduled_arrival
     ) BETWEEN 60 AND 3600

     -- Actual separation: no more than three hours.
     AND DATE_DIFF(
         'second',
         a.actual_arrival,
         b.actual_arrival
     ) BETWEEN 1 AND 10800
)

SELECT * FROM pairs
"""

# Write directly to Parquet so the full result need not
# be loaded into pandas memory.
escaped_output = str(PAIR_FILE).replace("\\", "/").replace("'", "''")

con.execute(
    f"COPY ({query}) TO '{escaped_output}' "
    "(FORMAT PARQUET, COMPRESSION ZSTD)",
    [str(RAW), str(RAW)],
)

print(f"\nCandidate pairs saved: {PAIR_FILE}")

# --------------------------------------------------
# Summarize relationships by station
# --------------------------------------------------

print("\n========== PAIR DATASET SUMMARY ==========")

summary = con.execute("""
    SELECT
        COUNT(*) AS candidate_pairs,
        COUNT(DISTINCT station_code) AS stations,
        COUNT(DISTINCT earlier_train_number) AS earlier_train_numbers,
        COUNT(DISTINCT later_train_number) AS later_train_numbers,
        CORR(earlier_delay_min, later_delay_min)
            AS overall_delay_correlation,
        AVG(earlier_delay_min) AS mean_earlier_delay_min,
        AVG(later_delay_min) AS mean_later_delay_min,
        MEDIAN(scheduled_gap_min) AS median_scheduled_gap_min,
        MEDIAN(actual_gap_min) AS median_actual_gap_min
    FROM read_parquet(?)
""", [str(PAIR_FILE)]).df()

print(summary.to_string(index=False))
summary.to_csv(OUTPUTS / "train_pair_overview.csv", index=False)

station_summary = con.execute("""
    SELECT
        station_code,
        COUNT(*) AS candidate_pairs,
        CORR(earlier_delay_min, later_delay_min)
            AS delay_correlation,
        AVG(earlier_delay_min) AS mean_earlier_delay_min,
        AVG(later_delay_min) AS mean_later_delay_min,
        AVG(later_delay_min) FILTER (
            WHERE earlier_delay_min > 15
        ) AS later_delay_when_earlier_over_15,
        AVG(later_delay_min) FILTER (
            WHERE earlier_delay_min <= 15
        ) AS later_delay_when_earlier_at_most_15
    FROM read_parquet(?)
    GROUP BY station_code
    HAVING COUNT(*) >= 100
    ORDER BY candidate_pairs DESC
""", [str(PAIR_FILE)]).df()

print("\n========== STATION-LEVEL RELATIONSHIPS ==========")
print(station_summary.to_string(index=False))

station_summary.to_csv(
    OUTPUTS / "station_delay_relationships.csv",
    index=False,
)

print("\nAnalysis complete.")
print("Review the pair count and correlation before interpreting results.")