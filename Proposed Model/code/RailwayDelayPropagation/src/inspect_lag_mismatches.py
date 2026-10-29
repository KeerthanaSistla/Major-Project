from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "data" / "raw" / "railway_delays_raw.parquet"
SEQ_FILE = ROOT / "data" / "processed" / "lstm_sequences.parquet"

con = duckdb.connect()

raw = RAW_FILE.as_posix().replace("'", "''")
seq = SEQ_FILE.as_posix().replace("'", "''")

query = f"""
WITH valid_raw AS (
    SELECT
        train_number,
        start_date,
        seq,
        station_code,
        arrival_delay_min
    FROM read_parquet('{raw}')
    WHERE arrival_delay_min IS NOT NULL
),
valid_lags AS (
    SELECT
        *,
        LAG(seq, 1) OVER w AS source_seq_1,
        LAG(seq, 2) OVER w AS source_seq_2,
        LAG(seq, 3) OVER w AS source_seq_3,
        LAG(seq, 4) OVER w AS source_seq_4,
        LAG(seq, 5) OVER w AS source_seq_5,

        LAG(arrival_delay_min, 1) OVER w AS valid_lag_1,
        LAG(arrival_delay_min, 2) OVER w AS valid_lag_2,
        LAG(arrival_delay_min, 3) OVER w AS valid_lag_3,
        LAG(arrival_delay_min, 4) OVER w AS valid_lag_4,
        LAG(arrival_delay_min, 5) OVER w AS valid_lag_5
    FROM valid_raw
    WINDOW w AS (
        PARTITION BY train_number, start_date
        ORDER BY seq
    )
),
sample AS (
    SELECT *
    FROM read_parquet('{seq}')
    WHERE hash(train_number, service_date, target_station_seq) % 100 = 0
),
joined AS (
    SELECT
        s.train_number,
        s.service_date,
        s.target_station_code,
        s.target_station_seq,

        s.delay_t1, s.delay_t2, s.delay_t3,
        s.delay_t4, s.delay_t5,

        r.source_seq_1, r.source_seq_2, r.source_seq_3,
        r.source_seq_4, r.source_seq_5,

        r.valid_lag_1, r.valid_lag_2, r.valid_lag_3,
        r.valid_lag_4, r.valid_lag_5

    FROM sample s
    JOIN valid_lags r
      ON r.train_number = s.train_number
     AND r.start_date = s.service_date
     AND r.seq = s.target_station_seq
)
SELECT *
FROM joined
WHERE
       delay_t1 IS DISTINCT FROM valid_lag_1
    OR delay_t2 IS DISTINCT FROM valid_lag_2
    OR delay_t3 IS DISTINCT FROM valid_lag_3
    OR delay_t4 IS DISTINCT FROM valid_lag_4
    OR delay_t5 IS DISTINCT FROM valid_lag_5
ORDER BY train_number, service_date, target_station_seq
LIMIT 25
"""

print("Checking whether stored features match the five most recent")
print("non-missing arrival-delay observations...")

df = con.execute(query).df()

if df.empty:
    print("\nPASS: All five stored delay features match the")
    print("five most recent non-missing raw arrival delays")
    print("in the sampled records.")
else:
    print(f"\nFound {len(df)} mismatching sampled rows (up to 25 shown):")
    print(df.to_string(index=False))

con.close()