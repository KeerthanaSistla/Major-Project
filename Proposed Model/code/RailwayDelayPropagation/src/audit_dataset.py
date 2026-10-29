
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "railway_delays_raw.parquet"
OUT = ROOT / "outputs"

OUT.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
con.execute("SET threads TO 4")

# --------------------------------------------------
# 1. Overall coverage and data quality
# --------------------------------------------------

print("\n========== DATASET OVERVIEW ==========")

overview = con.execute("""
    SELECT
        COUNT(*) AS total_records,
        COUNT(DISTINCT train_number) AS unique_train_numbers,
        COUNT(DISTINCT station_code) AS unique_stations,
        COUNT(DISTINCT start_date) AS distinct_start_dates,
        MIN(start_date) AS earliest_start_date,
        MAX(start_date) AS latest_start_date,
        COUNT(*) FILTER (
            WHERE actual_arrival IS NOT NULL
              AND arrival_delay_min IS NOT NULL
        ) AS usable_arrival_records,
        COUNT(*) FILTER (
            WHERE actual_departure IS NOT NULL
              AND departure_delay_min IS NOT NULL
        ) AS usable_departure_records,
        COUNT(*) FILTER (
            WHERE scheduled_arrival IS NOT NULL
              AND actual_arrival IS NOT NULL
        ) AS arrival_timestamp_pairs,
        COUNT(*) FILTER (
            WHERE scheduled_departure IS NOT NULL
              AND actual_departure IS NOT NULL
        ) AS departure_timestamp_pairs
    FROM read_parquet(?)
""", [str(RAW_FILE)]).df()

print(overview.to_string(index=False))
overview.to_csv(OUT / "audit_overview.csv", index=False)

# --------------------------------------------------
# 2. Status and delay distributions
# --------------------------------------------------

print("\n========== JOURNEY STATUS ==========")

status = con.execute("""
    SELECT status, COUNT(*) AS records
    FROM read_parquet(?)
    GROUP BY status
    ORDER BY records DESC
""", [str(RAW_FILE)]).df()

print(status.to_string(index=False))
status.to_csv(OUT / "audit_status.csv", index=False)

print("\n========== ARRIVAL DELAY DISTRIBUTION ==========")

delay_stats = con.execute("""
    SELECT
        COUNT(arrival_delay_min) AS observed_delays,
        MIN(arrival_delay_min) AS minimum_delay,
        QUANTILE_CONT(arrival_delay_min, 0.01) AS p01,
        QUANTILE_CONT(arrival_delay_min, 0.25) AS p25,
        MEDIAN(arrival_delay_min) AS median_delay,
        AVG(arrival_delay_min) AS mean_delay,
        QUANTILE_CONT(arrival_delay_min, 0.75) AS p75,
        QUANTILE_CONT(arrival_delay_min, 0.99) AS p99,
        MAX(arrival_delay_min) AS maximum_delay,
        COUNT(*) FILTER (
            WHERE arrival_delay_min < 0
        ) AS early_arrivals,
        COUNT(*) FILTER (
            WHERE arrival_delay_min = 0
        ) AS on_time_arrivals,
        COUNT(*) FILTER (
            WHERE arrival_delay_min > 0
        ) AS late_arrivals
    FROM read_parquet(?)
""", [str(RAW_FILE)]).df()

print(delay_stats.to_string(index=False))
delay_stats.to_csv(OUT / "audit_delay_statistics.csv", index=False)

# --------------------------------------------------
# 3. Check for duplicate journey-stop records
# --------------------------------------------------

print("\n========== DUPLICATE CHECK ==========")

duplicates = con.execute("""
    SELECT
        COUNT(*) AS duplicated_journey_stop_groups
    FROM (
        SELECT train_number, start_date, seq
        FROM read_parquet(?)
        GROUP BY train_number, start_date, seq
        HAVING COUNT(*) > 1
    )
""", [str(RAW_FILE)]).df()

print(duplicates.to_string(index=False))

# --------------------------------------------------
# 4. Stations with the most observations
# --------------------------------------------------

print("\n========== MOST FREQUENT STATIONS ==========")

stations = con.execute("""
    SELECT
        station_code,
        COUNT(*) AS total_stop_records,
        COUNT(arrival_delay_min) AS arrival_delay_records,
        COUNT(actual_departure) AS departure_records,
        COUNT(DISTINCT train_number) AS distinct_train_numbers
    FROM read_parquet(?)
    GROUP BY station_code
    ORDER BY total_stop_records DESC
    LIMIT 30
""", [str(RAW_FILE)]).df()

print(stations.to_string(index=False))
stations.to_csv(OUT / "audit_top_stations.csv", index=False)

# --------------------------------------------------
# 5. Look for multiple trains observed at the same
#    station on the same service date
# --------------------------------------------------

print("\n========== SHARED-STATION OPPORTUNITIES ==========")

shared = con.execute("""
    WITH station_day AS (
        SELECT
            start_date,
            station_code,
            COUNT(*) AS stop_records,
            COUNT(DISTINCT train_number) AS train_count,
            COUNT(actual_arrival) AS arrival_count
        FROM read_parquet(?)
        GROUP BY start_date, station_code
    )
    SELECT
        COUNT(*) AS station_days,
        COUNT(*) FILTER (
            WHERE train_count >= 2
        ) AS station_days_with_multiple_trains,
        COUNT(*) FILTER (
            WHERE train_count >= 5
        ) AS station_days_with_five_or_more_trains,
        MAX(train_count) AS maximum_trains_at_station_day
    FROM station_day
""", [str(RAW_FILE)]).df()

print(shared.to_string(index=False))
shared.to_csv(OUT / "audit_shared_stations.csv", index=False)

print("\nAudit complete. Reports saved in:", OUT)