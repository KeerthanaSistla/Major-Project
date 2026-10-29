from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "data" / "raw" / "railway_delays_raw.parquet"
SEQ_FILE = ROOT / "data" / "processed" / "lstm_sequences.parquet"

con = duckdb.connect()

def inspect_file(label, path):
    print(f"\n{'=' * 70}")
    print(f"{label}: {path}")
    print("=" * 70)

    if not path.exists():
        print("FILE NOT FOUND")
        return

    # Inspect actual column names and data types.
    schema = con.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)",
        [str(path)]
    ).fetchdf()

    print("\nCOLUMN SCHEMA")
    print(schema.to_string(index=False))

    # Number of rows.
    count = con.execute(
        "SELECT COUNT(*) FROM read_parquet(?)",
        [str(path)]
    ).fetchone()[0]

    print(f"\nTotal rows: {count:,}")

    # Show a few records to understand the structure.
    sample = con.execute(
        "SELECT * FROM read_parquet(?) LIMIT 5",
        [str(path)]
    ).fetchdf()

    print("\nSAMPLE RECORDS")
    print(sample.to_string(index=False))

    # Find likely key columns without assuming their names.
    columns = set(schema["column_name"].str.lower())

    train_col = next(
        (c for c in ["train_number", "train_no", "train_id"]
         if c in columns),
        None
    )

    date_col = next(
        (c for c in ["service_date", "date", "journey_date"]
         if c in columns),
        None
    )

    station_col = next(
        (c for c in ["station_seq", "target_station_seq",
                     "sequence_number", "stop_sequence"]
         if c in columns),
        None
    )

    delay_col = next(
        (c for c in ["arrival_delay_min", "arrival_delay",
                     "delay_minutes", "target_delay_min"]
         if c in columns),
        None
    )

    print("\nKEY-COLUMN DETECTION")
    print(f"Train column:   {train_col}")
    print(f"Date column:    {date_col}")
    print(f"Station column: {station_col}")
    print(f"Delay column:   {delay_col}")

    # Basic journey-key checks, when the required columns exist.
    if train_col and date_col:
        query = f"""
            SELECT
                COUNT(*) AS rows,
                COUNT(DISTINCT ({train_col}, {date_col}))
                    AS train_date_groups
            FROM read_parquet(?)
        """

        result = con.execute(query, [str(path)]).fetchone()

        print("\nTRAIN-DATE SUMMARY")
        print(f"Rows: {result[0]:,}")
        print(f"Distinct train/date combinations: {result[1]:,}")

        groups = con.execute(
            f"""
            SELECT
                {train_col} AS train_number,
                {date_col} AS service_date,
                COUNT(*) AS records
            FROM read_parquet(?)
            GROUP BY 1, 2
            ORDER BY records DESC
            LIMIT 10
            """,
            [str(path)]
        ).fetchdf()

        print("\nTEN LARGEST TRAIN-DATE GROUPS")
        print(groups.to_string(index=False))

    # Check whether station sequence values are duplicated.
    if train_col and date_col and station_col:
        duplicates = con.execute(
            f"""
            SELECT
                COUNT(*) AS duplicate_key_groups
            FROM (
                SELECT
                    {train_col},
                    {date_col},
                    {station_col}
                FROM read_parquet(?)
                GROUP BY 1, 2, 3
                HAVING COUNT(*) > 1
            ) d
            """,
            [str(path)]
        ).fetchone()[0]

        print("\nDUPLICATE STATION-KEY GROUPS")
        print(f"Duplicate groups: {duplicates:,}")

    # Inspect target delay distribution.
    if delay_col:
        stats = con.execute(
            f"""
            SELECT
                COUNT(*) AS rows,
                COUNT({delay_col}) AS non_null_delays,
                MIN({delay_col}) AS min_delay,
                QUANTILE_CONT({delay_col}, 0.01) AS p01,
                QUANTILE_CONT({delay_col}, 0.50) AS median,
                QUANTILE_CONT({delay_col}, 0.95) AS p95,
                QUANTILE_CONT({delay_col}, 0.99) AS p99,
                MAX({delay_col}) AS max_delay
            FROM read_parquet(?)
            """,
            [str(path)]
        ).fetchdf()

        print("\nDELAY DISTRIBUTION (MINUTES, IF STORED IN MINUTES)")
        print(stats.to_string(index=False))


inspect_file("RAW DATASET", RAW_FILE)
inspect_file("LSTM SEQUENCE DATASET", SEQ_FILE)

con.close()
print("\nAudit completed.")