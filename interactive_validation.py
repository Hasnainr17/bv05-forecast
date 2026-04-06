import argparse

# Import from your main module (adjust file name accordingly)
from load_forecast_json_and_csv_upgraded import (
    perform_validation,
    train_models_from_historical_csv,
    LOCATIONS,
    DATA_DIR
)

def main():
    # -----------------------------
    # Argument Parser
    # -----------------------------
    parser = argparse.ArgumentParser(
        description="Custom Range Model Validation"
    )

    parser.add_argument(
        "--city",
        type=str,
        required=True,
        help="City name (must match available dataset)"
    )

    parser.add_argument(
        "--start_date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end_date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)"
    )

    args = parser.parse_args()

    city = args.city
    start_date = args.start_date
    end_date = args.end_date

    # -----------------------------
    # Load historical data path
    # -----------------------------
    if city not in LOCATIONS:
        raise ValueError(
            f"City '{city}' not found. Available: {list(LOCATIONS.keys())}"
        )

    csv_name = LOCATIONS[city]
    hist_path = DATA_DIR / csv_name   # <-- FIXED PATH HANDLING

    # -----------------------------
    # Train models
    # -----------------------------
    res_model, ci_model = train_models_from_historical_csv(hist_path)

    # -----------------------------
    # Run validation
    # -----------------------------
    metrics = perform_validation(
        res_model=res_model,
        ci_model=ci_model,
        output_csv="Interactive_model_validation.xlsx",
        hist_path=hist_path,
        city=city,
        start_date=start_date,
        end_date=end_date,
    )

    # -----------------------------
    # Print results
    # -----------------------------
    print("\nValidation Complete")
    print(f"City: {city}")
    print(f"Date Range: {start_date} → {end_date}")

    if metrics:
        print("\nResidential Metrics:")
        print(metrics.get("residential", {}))

        print("\nC&I Metrics:")
        print(metrics.get("ci", {}))
    else:
        print("No data available for selected range.")


if __name__ == "__main__":
    main()