import polars as pl
import numpy as np
from pathlib import Path

def process_sentinel1_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    
    Args:
        df (pl.DataFrame): Input DataFrame with Sentinel-1 columns:
                          'VH', 'VV', 'instrumentMode', 'orbitProperties_pass',
                          'relativeOrbitNumber_start', 'date'.
                          Assumes 'VH' and 'VV' are already in decibels (dB).

    Returns:
        pl.DataFrame: A new DataFrame with selected original and derived features.
    """
    
    # Create PID from row number
    if "PID" not in df.columns:
        df = df.with_row_index("PID")
    else:
        # If PID exists, ensure it's properly formatted
        pass
    # Parse date column
    df = df.with_columns([
        pl.col("date").str.to_datetime()
    ])
    
    # Ensure VH and VV are numeric and handle fill values
    sar_bands = ['VH', 'VV']
    
    cleaning_expressions = []
    for band in sar_bands:
        cleaning_expressions.append(
            pl.when(pl.col(band) < -50)  # Very low values likely fill or noise
            .then(None)
            .otherwise(pl.col(band).cast(pl.Float64))
            .alias(band)
        )
    
    df = df.with_columns(cleaning_expressions)
    
    # Derived SAR Features (convert to linear first for ratios/sums, then back to dB)
    epsilon = 1e-6  # Small value to avoid division by zero
    
    df = df.with_columns([
        # Convert to linear scale
        (10 ** (pl.col("VH") / 10)).alias("VH_linear"),
        (10 ** (pl.col("VV") / 10)).alias("VV_linear")
    ])
    
    df = df.with_columns([
        # Cross-polarization ratio (VH/VV)
        (pl.col("VH_linear") / (pl.col("VV_linear") + epsilon)).alias("VH_to_VV_Ratio_linear"),
        
        # Difference (VV - VH)
        (pl.col("VV") - pl.col("VH")).alias("VV_minus_VH_dB"),
        
        # Total Power (Span)
        (pl.col("VV_linear") + pl.col("VH_linear")).alias("Span_linear")
    ])
    
    df = df.with_columns([
        # Convert ratios and span back to dB
        (10 * (pl.col("VH_to_VV_Ratio_linear").log10())).alias("VH_to_VV_Ratio_dB"),
        (10 * (pl.col("Span_linear").log10())).alias("Span_dB")
    ])
    
    # Select and reorder relevant columns
    output_columns = [
        'PID', 'VH_to_VV_Ratio_dB', 'VV_minus_VH_dB', 'Span_dB'
    ]
    
    # Include categorical metadata that can be useful
    meta_cols = ['instrumentMode', 'orbitProperties_pass', 'relativeOrbitNumber_start']
    available_meta_cols = [col for col in meta_cols if col in df.columns]
    output_columns.extend(available_meta_cols)
    
    return df.select(output_columns)

def main():
    """Main function to process Sentinel-1 data and save as parquet."""
    
    # Input and output paths
    input_file = "../dataset/Sentinel1_data.csv"  # Adjust path as needed
    output_dir = Path("../processed_data")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "processed_sentinel1.parquet"
    
    try:
        # Read the data
        print(f"Reading data from {input_file}...")
        df = pl.read_csv(input_file)
        
        print(f"Original data shape: {df.shape}")
        print(f"Columns: {df.columns}")
        
        # Process the data
        print("Processing Sentinel-1 data...")
        processed_df = process_sentinel1_data(df)
        
        print(f"Processed data shape: {processed_df.shape}")
        print(f"Processed columns: {processed_df.columns}")
        
        # Save as parquet
        print(f"Saving processed data to {output_file}...")
        processed_df.write_parquet(output_file)
        
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        raise

if __name__ == "__main__":
    main()