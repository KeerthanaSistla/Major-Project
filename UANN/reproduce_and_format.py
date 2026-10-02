import subprocess
import ast
import re
import math


# ============================================================
# Configuration
# ============================================================

COMMAND = [
    "python",
    "run.py",
    "-m",
    "test_allfeatures",
    "-v",
    "allfeatures"
]


# ============================================================
# Helper functions
# ============================================================

def parse_metrics(output):
    """
    Extract all lines of the form:

        metrics {...}

    from run.py output.
    """

    metrics = []

    for line in output.splitlines():
        line = line.strip()

        if line.startswith("metrics "):

            dictionary_text = line[len("metrics "):]

            # ast.literal_eval cannot parse nan.
            dictionary_text = dictionary_text.replace(": nan", ": None")

            try:
                data = ast.literal_eval(dictionary_text)
                metrics.append(data)

            except Exception as e:
                print("Could not parse metrics line:")
                print(line)
                print("Error:", e)

    return metrics


def extract_value(pattern, output, default="N/A"):
    """
    Extract the first regex group from run.py output.
    """

    match = re.search(pattern, output)

    if match:
        return match.group(1)

    return default


def fmt(value, decimals=3):
    """
    Format numerical values safely.
    """

    if value is None:
        return "N/A"

    if isinstance(value, (int, float)):

        if isinstance(value, float) and math.isnan(value):
            return "N/A"

        return f"{value:.{decimals}f}"

    return str(value)


def safe_percent_change(value, reference):
    """
    Calculate percentage change relative to reference.
    """

    if value is None or reference is None or reference == 0:
        return None

    return ((value - reference) / reference) * 100


def print_separator(char="─", width=75):
    print(char * width)


def print_metric_row(name, metrics):
    """
    Print the regular test-results table row.
    """

    print(
        f"{name:<30}"
        f"{fmt(metrics.get('MAE')):>12}"
        f"{fmt(metrics.get('MSE')):>12}"
        f"{fmt(metrics.get('coverage')):>12}"
        f"{fmt(metrics.get('mean_unc')):>12}"
        f"{fmt(metrics.get('mean_pi_width')):>14}"
    )


# ============================================================
# Run original reproduction code
# ============================================================

print()

print_separator("═")
print("TRAIN DELAY MODEL REPRODUCTION")
print_separator("═")

print()
print("Running:")
print("  python run.py -m test_allfeatures -v allfeatures")
print()


result = subprocess.run(
    COMMAND,
    capture_output=True,
    text=True
)


stdout = result.stdout
stderr = result.stderr


# ============================================================
# Check whether run.py executed successfully
# ============================================================

if result.returncode != 0:

    print("❌ Reproduction failed.")
    print_separator()

    print(stdout)
    print(stderr)

    raise SystemExit(result.returncode)


# ============================================================
# Extract dataset information from run.py output
# ============================================================

train_shape = extract_value(
    r"train x and y \(([^)]+)\)",
    stdout
)

val_shape = extract_value(
    r"val x and y \(([^)]+)\)",
    stdout
)

test_shape = extract_value(
    r"test x and y \(([^)]+)\)",
    stdout
)


# Try to extract feature count from the train shape.

input_features = "N/A"

feature_match = re.search(
    r"train x and y \(\d+,\s*(\d+)\)",
    stdout
)

if feature_match:
    input_features = feature_match.group(1)


# ============================================================
# Extract model parameters
# ============================================================

params_match = re.search(
    r"PARAMS (\{.*?\})",
    stdout
)

params = {}

if params_match:

    try:
        params = ast.literal_eval(
            params_match.group(1)
        )

    except Exception:
        params = {}


# ============================================================
# Extract metrics
# ============================================================

all_metrics = parse_metrics(stdout)


# ============================================================
# Model labels
# ============================================================
#
# These labels correspond to the models actually evaluated by
# the current run.py configuration.
#
# The metric VALUES themselves are NOT hardcoded.
# They come directly from run.py.
# ============================================================

labels = [
    "NN",
    "NN + baseline uncertainty",
    "Simple Median",
    "Simple Mean",
    "Simple Average"
]


results = []

for label, metric in zip(labels, all_metrics):

    results.append(
        (label, metric)
    )


# ============================================================
# Dataset / evaluation setup
# ============================================================

print()
print_separator()
print("DATASET / EVALUATION SETUP")
print_separator()

print(
    f"{'Train samples':<25}: {train_shape}"
)

print(
    f"{'Validation samples':<25}: {val_shape}"
)

print(
    f"{'Test samples':<25}: {test_shape}"
)

print(
    f"{'Input features':<25}: {input_features}"
)

print(
    f"{'Feature version':<25}: allfeatures"
)

print(
    f"{'Model directory':<25}: test_allfeatures"
)


# ============================================================
# Neural network configuration
# ============================================================

print()
print_separator()
print("NEURAL NETWORK CONFIGURATION")
print_separator()


if params:

    first_layer = params.get(
        "first_layer_size",
        "N/A"
    )

    second_layer = params.get(
        "second_layer_size",
        "N/A"
    )

    nr_layers = params.get(
        "nr_layers",
        "N/A"
    )

    learning_rate = params.get(
        "learning_rate",
        "N/A"
    )

    dropout_rate = params.get(
        "dropout_rate",
        "N/A"
    )

else:

    first_layer = "N/A"
    second_layer = "N/A"
    nr_layers = "N/A"
    learning_rate = "N/A"
    dropout_rate = "N/A"


print(
    f"{'First layer':<25}: {first_layer}"
)

print(
    f"{'Second layer':<25}: {second_layer}"
)

print(
    f"{'Number of layers':<25}: {nr_layers}"
)

print(
    f"{'Learning rate':<25}: {learning_rate}"
)

print(
    f"{'Dropout rate':<25}: {dropout_rate}"
)


# Construct model name from extracted parameters.

if params:

    model_name = (
        f"nn-"
        f"{first_layer}-"
        f"{second_layer}-"
        f"{nr_layers}-"
        f"{learning_rate}-"
        f"{dropout_rate}"
    )

else:

    model_name = "N/A"


print(
    f"{'Model':<25}: {model_name}"
)


# ============================================================
# Regular test-set results
# ============================================================

print()
print_separator()
print("TEST SET RESULTS")
print_separator()


print(
    f"{'Model':<30}"
    f"{'MAE':>12}"
    f"{'MSE':>12}"
    f"{'Coverage':>12}"
    f"{'Mean Unc.':>12}"
    f"{'PI Width':>14}"
)

print_separator()


for label, metric in results:

    print_metric_row(
        label,
        metric
    )


print_separator()


# ============================================================
# Paper-style comparison table
# ============================================================
#
# This is the table that can be compared with Table 1
# from the paper.
#
# MAE:
#     get_metrics() returns minutes.
#     Convert minutes -> seconds.
#
# RMSE:
#     MSE is in minutes^2.
#     sqrt(MSE) -> minutes.
#     Convert minutes -> seconds.
#
# LoR Δ=30s:
#     The repository's Likely_30 corresponds to the
#     probability of being within ±30 seconds.
#
# % to NN:
#     Calculated relative to the NN result from THIS run.
#
# Nothing in this table is a hardcoded result value.
# ============================================================

if results:

    nn = results[0][1]

    # --------------------------------------------------------
    # NN reference values
    # --------------------------------------------------------

    nn_mae_min = nn.get("MAE")

    nn_mse_min2 = nn.get("MSE")

    nn_lor = nn.get("Likely_30")


    # MAE: minutes -> seconds

    if nn_mae_min is not None:

        nn_mae_sec = (
            nn_mae_min * 60
        )

    else:

        nn_mae_sec = None


    # MSE -> RMSE -> seconds

    if nn_mse_min2 is not None:

        nn_rmse_sec = (
            math.sqrt(nn_mse_min2) * 60
        )

    else:

        nn_rmse_sec = None


    # Likely_30 -> percentage

    if nn_lor is not None:

        nn_lor_pct = (
            nn_lor * 100
        )

    else:

        nn_lor_pct = None


    # --------------------------------------------------------
    # Print table
    # --------------------------------------------------------

    print()
    print_separator("═", 115)

    print(
        "PAPER-STYLE MODEL COMPARISON"
    )

    print_separator("═", 115)


    print(
        f"{'Model':<30}"
        f"{'MAE [s]':>12}"
        f"{'% to NN':>12}"
        f"{'RMSE [s]':>14}"
        f"{'% to NN':>12}"
        f"{'LoR Δ=30s [%]':>18}"
        f"{'% to NN':>12}"
    )

    print_separator("-", 115)


    # --------------------------------------------------------
    # Each model
    # --------------------------------------------------------

    for label, metric in results:

        mae_min = metric.get("MAE")

        mse_min2 = metric.get("MSE")

        lor = metric.get("Likely_30")


        # ----------------------------------------------------
        # MAE
        # ----------------------------------------------------

        if mae_min is not None:

            mae_sec = (
                mae_min * 60
            )

        else:

            mae_sec = None


        # ----------------------------------------------------
        # RMSE
        # ----------------------------------------------------

        if mse_min2 is not None:

            rmse_sec = (
                math.sqrt(mse_min2) * 60
            )

        else:

            rmse_sec = None


        # ----------------------------------------------------
        # LoR
        # ----------------------------------------------------

        if lor is not None:

            lor_pct = (
                lor * 100
            )

        else:

            lor_pct = None


        # ----------------------------------------------------
        # Percentage differences relative to NN
        # ----------------------------------------------------

        mae_change = safe_percent_change(
            mae_sec,
            nn_mae_sec
        )


        rmse_change = safe_percent_change(
            rmse_sec,
            nn_rmse_sec
        )


        lor_change = safe_percent_change(
            lor,
            nn_lor
        )


        # ----------------------------------------------------
        # Formatting
        # ----------------------------------------------------

        mae_text = (
            f"{mae_sec:.2f}"
            if mae_sec is not None
            else "N/A"
        )


        mae_change_text = (
            f"{mae_change:+.1f}%"
            if mae_change is not None
            else "N/A"
        )


        rmse_text = (
            f"{rmse_sec:.2f}"
            if rmse_sec is not None
            else "N/A"
        )


        rmse_change_text = (
            f"{rmse_change:+.1f}%"
            if rmse_change is not None
            else "N/A"
        )


        lor_text = (
            f"{lor_pct:.2f}"
            if lor_pct is not None
            else "N/A"
        )


        lor_change_text = (
            f"{lor_change:+.1f}%"
            if lor_change is not None
            else "N/A"
        )


        # ----------------------------------------------------
        # Print row
        # ----------------------------------------------------

        print(
            f"{label:<30}"
            f"{mae_text:>12}"
            f"{mae_change_text:>12}"
            f"{rmse_text:>14}"
            f"{rmse_change_text:>12}"
            f"{lor_text:>18}"
            f"{lor_change_text:>12}"
        )


    print_separator("-", 115)


    # --------------------------------------------------------
    # Explain the columns
    # --------------------------------------------------------

    print()
    print("Column definitions:")
    print("  MAE [s]        : Mean Absolute Error in seconds")
    print("  RMSE [s]       : Root Mean Squared Error in seconds")
    print("  LoR Δ=30s [%]  : Likelihood of prediction being within ±30 seconds")
    print("  % to NN        : Percentage difference relative to the NN")


# ============================================================
# Detailed primary NN results
# ============================================================

if results:

    nn = results[0][1]

    print()
    print_separator()
    print("PRIMARY NN RESULTS")
    print_separator()


    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    print(
        f"{'MAE':<30}: "
        f"{fmt(nn.get('MAE'))} minutes"
    )


    # --------------------------------------------------------
    # MSE
    # --------------------------------------------------------

    print(
        f"{'MSE':<30}: "
        f"{fmt(nn.get('MSE'))}"
    )


    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    coverage = nn.get("coverage")

    if coverage is not None:

        coverage = coverage * 100


    print(
        f"{'Coverage':<30}: "
        f"{fmt(coverage)}%"
    )


    # --------------------------------------------------------
    # Mean uncertainty
    # --------------------------------------------------------

    print(
        f"{'Mean uncertainty':<30}: "
        f"{fmt(nn.get('mean_unc'))}"
    )


    # --------------------------------------------------------
    # Prediction interval width
    # --------------------------------------------------------

    print(
        f"{'Mean prediction interval width':<30}: "
        f"{fmt(nn.get('mean_pi_width'))} minutes"
    )


    # --------------------------------------------------------
    # Likelihood metrics
    # --------------------------------------------------------

    print()
    print("Likelihood metrics:")


    likely_15 = nn.get("Likely_15")

    likely_30 = nn.get("Likely_30")

    likely_45 = nn.get("Likely_45")


    if likely_15 is not None:
        likely_15 *= 100

    if likely_30 is not None:
        likely_30 *= 100

    if likely_45 is not None:
        likely_45 *= 100


    print(
        f"{'  Likely within ±15 min':<30}: "
        f"{fmt(likely_15)}%"
    )


    print(
        f"{'  Likely within ±30 min':<30}: "
        f"{fmt(likely_30)}%"
    )


    print(
        f"{'  Likely within ±45 min':<30}: "
        f"{fmt(likely_45)}%"
    )


    # --------------------------------------------------------
    # Uncertainty correlations
    # --------------------------------------------------------

    print()
    print("Uncertainty correlation:")


    print(
        f"{'  Spearman correlation':<30}: "
        f"{fmt(nn.get('spearman_r'))}"
    )


    print(
        f"{'  Pearson correlation':<30}: "
        f"{fmt(nn.get('pearsonr'))}"
    )


# ============================================================
# Calibration information
# ============================================================

quantiles = re.findall(
    r"Quantiles ([0-9.eE+-]+)",
    stdout
)


factors = re.findall(
    r"Best factor for likelihood ([0-9.eE+-]+)",
    stdout
)


print()
print_separator()
print("CALIBRATION")
print_separator()


if quantiles:

    print(
        f"NN prediction interval quantile : "
        f"{quantiles[0]}"
    )


if factors:

    print(
        f"NN likelihood factor            : "
        f"{factors[0]}"
    )


# ============================================================
# Baseline comparison
# ============================================================

if len(results) >= 5:

    nn = results[0][1]

    median = results[2][1]

    mean = results[3][1]

    avg = results[4][1]


    print()
    print_separator()
    print("NN VS SIMPLE BASELINES")
    print_separator()


    print(
        f"{'Model':<25}"
        f"{'MAE':>12}"
        f"{'MSE':>14}"
    )


    print_separator()


    print(
        f"{'NN':<25}"
        f"{fmt(nn.get('MAE')):>12}"
        f"{fmt(nn.get('MSE')):>14}"
    )


    print(
        f"{'Simple Median':<25}"
        f"{fmt(median.get('MAE')):>12}"
        f"{fmt(median.get('MSE')):>14}"
    )


    print(
        f"{'Simple Mean':<25}"
        f"{fmt(mean.get('MAE')):>12}"
        f"{fmt(mean.get('MSE')):>14}"
    )


    print(
        f"{'Simple Average':<25}"
        f"{fmt(avg.get('MAE')):>12}"
        f"{fmt(avg.get('MSE')):>14}"
    )


    print_separator()


# ============================================================
# Completion
# ============================================================

print()

print_separator("═")
print("REPRODUCTION COMPLETED SUCCESSFULLY")
print_separator("═")

print()
print("The original run.py was executed unchanged.")
print("This script only reformats and calculates presentation metrics.")
print()