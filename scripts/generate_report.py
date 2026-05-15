#!/usr/bin/env python
"""Generate HTML report with all model metrics.

Creates a comprehensive dashboard showing:
- QSAR model performance
- GNN model results
- Generation model metrics (VAE, Diffusion)
- Docking results

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --output results/dashboard.html
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger


def load_metrics_from_directory(metrics_dir: Path) -> pd.DataFrame:
    """Load all metrics JSON files from directory.

    Args:
        metrics_dir: Directory containing metrics JSON files.

    Returns:
        DataFrame with metrics from all models.
    """
    rows = []

    if not metrics_dir.exists():
        logger.warning(f"Metrics directory not found: {metrics_dir}")
        return pd.DataFrame()

    for json_file in metrics_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Extract model name from filename
            model_name = json_file.stem

            # Flatten nested structure
            flat_data = {"model": model_name}

            # Handle different metric file formats
            if "test_metrics" in data:
                flat_data.update(data["test_metrics"])
            if "cv_results" in data:
                cv_data = {
                    f"cv_{k}": v
                    for k, v in data["cv_results"].items()
                    if not isinstance(v, (list, dict))
                }
                flat_data.update(cv_data)
            if "generation_metrics" in data:
                flat_data.update(data["generation_metrics"])
            if "reconstruction_accuracy" in data:
                flat_data["reconstruction_accuracy"] = data["reconstruction_accuracy"]

            # Add training info if available
            if "training" in data:
                flat_data.update(data["training"])

            rows.append(flat_data)
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def generate_html_report(df: pd.DataFrame, output_path: Path) -> None:
    """Generate HTML report from metrics DataFrame.

    Args:
        df: DataFrame with model metrics.
        output_path: Path to save HTML report.
    """
    # Create HTML with modern styling
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TB Drug Discovery - Model Dashboard</title>
    <style>
        :root {
            --primary: #2563eb;
            --secondary: #64748b;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg: #f8fafc;
            --card: #ffffff;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: #1e293b;
            line-height: 1.6;
        }

        .header {
            background: linear-gradient(135deg, var(--primary), #1e40af);
            color: white;
            padding: 2rem;
            text-align: center;
        }

        .header h1 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }

        .header p {
            opacity: 0.9;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }

        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }

        .card {
            background: var(--card);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        .card h3 {
            font-size: 0.875rem;
            text-transform: uppercase;
            color: var(--secondary);
            margin-bottom: 0.5rem;
        }

        .card .value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary);
        }

        .card .value.success { color: var(--success); }
        .card .value.warning { color: var(--warning); }
        .card .value.danger { color: var(--danger); }

        table {
            width: 100%;
            background: var(--card);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-collapse: collapse;
        }

        th, td {
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }

        th {
            background: #f1f5f9;
            font-weight: 600;
            font-size: 0.875rem;
            text-transform: uppercase;
            color: var(--secondary);
        }

        tr:hover {
            background: #f8fafc;
        }

        .badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .badge.success {
            background: #dcfce7;
            color: #166534;
        }

        .badge.warning {
            background: #fef3c7;
            color: #92400e;
        }

        .footer {
            text-align: center;
            padding: 2rem;
            color: var(--secondary);
            font-size: 0.875rem;
        }

        .numeric {
            font-family: 'SF Mono', Monaco, monospace;
        }

        @media (max-width: 768px) {
            .container {
                padding: 1rem;
            }
            .header h1 {
                font-size: 1.5rem;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>TB Drug Discovery Pipeline</h1>
        <p>Model Performance Dashboard • Generated on {timestamp}</p>
    </div>

    <div class="container">
        <div class="summary-cards">
            <div class="card">
                <h3>Total Models</h3>
                <div class="value">{total_models}</div>
            </div>
            <div class="card">
                <h3>QSAR Models</h3>
                <div class="value">{qsar_count}</div>
            </div>
            <div class="card">
                <h3>Generative Models</h3>
                <div class="value">{gen_count}</div>
            </div>
            <div class="card">
                <h3>Avg Validity</h3>
                <div class="value {validity_class}">{avg_validity:.1%}</div>
            </div>
        </div>

        <h2 style="margin-bottom: 1rem;">Model Metrics</h2>
        {table}
    </div>

    <div class="footer">
        <p>TB Drug Discovery Pipeline • QSAR/GNN/Generative Models</p>
    </div>
</body>
</html>
"""

    # Calculate summary statistics
    total_models = len(df)
    qsar_count = len([m for m in df.get("model", []) if "qsar" in str(m).lower()])
    gen_count = len([m for m in df.get("model", []) if any(x in str(m).lower() for x in ["vae", "diffusion", "gen"])])

    avg_validity = df.get("validity", pd.Series([0])).mean() if "validity" in df.columns else 0
    validity_class = "success" if avg_validity >= 0.9 else "warning" if avg_validity >= 0.7 else "danger"

    # Format DataFrame for display
    display_df = df.copy()

    # Round numeric columns
    numeric_cols = display_df.select_dtypes(include=["float64", "float32"]).columns
    for col in numeric_cols:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")

    # Generate table HTML
    table_html = display_df.to_html(
        index=False,
        classes="metrics-table",
        border=0,
        na_rep="-",
    )

    # Fill template
    html = html_template.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_models=total_models,
        qsar_count=qsar_count,
        gen_count=gen_count,
        avg_validity=avg_validity,
        validity_class=validity_class,
        table=table_html,
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"Report saved to: {output_path}")


def generate_markdown_report(df: pd.DataFrame, output_path: Path) -> None:
    """Generate Markdown report for README embedding.

    Args:
        df: DataFrame with model metrics.
        output_path: Path to save Markdown report.
    """
    lines = [
        "# TB Drug Discovery - Model Metrics Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- **Total Models**: {len(df)}",
        f"- **QSAR Models**: {len([m for m in df.get('model', []) if 'qsar' in str(m).lower()])}",
        f"- **Generative Models**: {len([m for m in df.get('model', []) if any(x in str(m).lower() for x in ['vae', 'diffusion', 'gen'])])}",
        "",
        "## Model Performance",
        "",
    ]

    if not df.empty:
        # Add table
        lines.append(df.to_markdown(index=False))
        lines.append("")

    lines.extend([
        "## Notes",
        "",
        "- ROC-AUC: Area Under ROC Curve (classification)",
        "- PR-AUC: Area Under Precision-Recall Curve (classification, imbalanced data)",
        "- R²: Coefficient of determination (regression)",
        "- Validity: Fraction of generated SMILES that are chemically valid",
        "- Uniqueness: Fraction of valid SMILES that are unique",
        "- Novelty: Fraction of unique SMILES not in training set",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Markdown report saved to: {output_path}")


def main():
    """Generate reports from metrics."""
    parser = argparse.ArgumentParser(
        description="Generate HTML/Markdown reports from model metrics"
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="results/metrics",
        help="Directory containing metrics JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/report.html",
        help="Output HTML report path",
    )
    parser.add_argument(
        "--markdown-output",
        type=str,
        default="results/report.md",
        help="Output Markdown report path",
    )

    args = parser.parse_args()

    logger.info("Generating reports...")

    # Load metrics
    metrics_dir = Path(args.metrics_dir)
    df = load_metrics_from_directory(metrics_dir)

    if df.empty:
        logger.warning("No metrics found. Run training scripts first.")
        # Create empty report with message
        df = pd.DataFrame({"message": ["No metrics available. Run training first."]})

    # Generate reports
    generate_html_report(df, Path(args.output))
    generate_markdown_report(df, Path(args.markdown_output))

    logger.info("Reports generated successfully!")
    logger.info(f"  HTML: {args.output}")
    logger.info(f"  Markdown: {args.markdown_output}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
