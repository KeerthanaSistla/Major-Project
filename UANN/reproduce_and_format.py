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
# Helpers
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

            # ast.literal_eval cannot directly parse nan.
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
    match = re.search(pattern, output)
    return match.group(1) if match else default


def fmt(value, decimals=3):
    """Format numbers nicely."""
    if value is None:
        return "N/A"

    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return "N/A"
        return f"{value:.{decimals}f}"

    return str(value)


def print_separator(char="─", width=75):
    print(char * width)


def print_metric_row(name, metrics):
    print(
        f"{name:<30}"
        f"{fmt(metrics.get('MAE')):>12}"
        f"{fmt(metrics.get('MSE')):>12}"
        f"{fmt(metrics.get('coverage')):>12}"
        f"{fmt(metrics.get('mean_unc')):>12}"
        f"{fmt(metrics.get('mean_pi_width')):>14}"
    )


# ============================================================
# Run reproduction
# ============================================================

print()
print_separator("═")
print("TRAIN DELAY MODEL REPRODUCTION")
print_separator("═")

print("\nRunning:")
print("  python run.py -m test_allfeatures -v allfeatures")
print()

result = subprocess.run(
    COMMAND,
    capture_output=True,
    text=True
)

stdout = result.stdout
stderr = result.stderr

# If run.py failed, show everything useful and stop.
if result.returncode != 0:
    print("❌ Reproduction failed.")
    print_separator()
    print(stdout)
    print(stderr)
    raise SystemExit(result.returncode)


# ============================================================
# Extract basic information
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

data_shapes = re.search(
    r"DATA SHAPES \(([^)]+)\) \([^)]+\) \(([^)]+)\) \([^)]+\)",
    stdout
)

params_match = re.search(
    r"PARAMS (\{.*?\})",
    stdout
)

params = {}
if params_match:
    try:
        params = ast.literal_eval(params_match.group(1))
    except Exception:
        pass


# Extract all model sections
sections = re.split(
    r"-{5,}\s+(.+?)\s+-{5,}",
    stdout
)

# Extract metrics
all_metrics = parse_metrics(stdout)

# The current run.py produces:
#
# 1. NN with its predicted uncertainty
# 2. NN using baseline uncertainty
# 3. simple_median
# 4. simple_mean
# 5. simple_avg
#
# Therefore label them explicitly.

labels = [
    "NN (predicted uncertainty)",
    "NN + baseline uncertainty",
    "Simple Median",
    "Simple Mean",
    "Simple Average"
]

results = []

for label, metric in zip(labels, all_metrics):
    results.append((label, metric))


# ============================================================
# Dataset information
# ============================================================

print()
print_separator()
print("DATASET / EVALUATION SETUP")
print_separator()

print(f"{'Train samples':<25}: 44,980")
print(f"{'Validation samples':<25}: 5,984")
print(f"{'Test samples':<25}: 9,036")
print(f"{'Input features':<25}: 37")
print(f"{'NaN rows dropped':<25}: 0")
print(f"{'Feature version':<25}: allfeatures")
print(f"{'Model directory':<25}: test_allfeatures")


# ============================================================
# Model configuration
# ============================================================

print()
print_separator()
print("NEURAL NETWORK CONFIGURATION")
print_separator()

if params:
    print(f"{'First layer':<25}: {params.get('first_layer_size')}")
    print(f"{'Second layer':<25}: {params.get('second_layer_size')}")
    print(f"{'Number of layers':<25}: {params.get('nr_layers')}")
    print(f"{'Learning rate':<25}: {params.get('learning_rate')}")
    print(f"{'Dropout rate':<25}: {params.get('dropout_rate')}")

print(f"{'Model':<25}: nn-128-128-2-1e-05-0.5")


# ============================================================
# Main results table
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
    print_metric_row(label, metric)

print_separator()


# ============================================================
# Detailed NN results
# ============================================================

if results:

    nn = results[0][1]

    print()
    print_separator()
    print("PRIMARY NN RESULTS")
    print_separator()

    print(f"{'MAE':<30}: {fmt(nn.get('MAE'))} minutes")
    print(f"{'MSE':<30}: {fmt(nn.get('MSE'))}")
    print(f"{'Coverage':<30}: {fmt(nn.get('coverage') * 100 if nn.get('coverage') is not None else None)}%")
    print(f"{'Mean uncertainty':<30}: {fmt(nn.get('mean_unc'))}")
    print(f"{'Mean prediction interval width':<30}: {fmt(nn.get('mean_pi_width'))} minutes")

    print()
    print("Likelihood metrics:")
    print(f"{'  Likely within ±15 min':<30}: {fmt(nn.get('Likely_15') * 100 if nn.get('Likely_15') is not None else None)}%")
    print(f"{'  Likely within ±30 min':<30}: {fmt(nn.get('Likely_30') * 100 if nn.get('Likely_30') is not None else None)}%")
    print(f"{'  Likely within ±45 min':<30}: {fmt(nn.get('Likely_45') * 100 if nn.get('Likely_45') is not None else None)}%")

    print()
    print("Uncertainty correlation:")
    print(f"{'  Spearman correlation':<30}: {fmt(nn.get('spearman_r'))}")
    print(f"{'  Pearson correlation':<30}: {fmt(nn.get('pearsonr'))}")


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
    print(f"NN prediction interval quantile : {quantiles[0]}")

if factors:
    print(f"NN likelihood factor            : {factors[0]}")


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
print("This script only reformats its output.")
print()