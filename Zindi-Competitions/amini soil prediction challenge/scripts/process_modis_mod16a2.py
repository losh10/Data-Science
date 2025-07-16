import polars as pl
import numpy as np
from pathlib import Path

def process_modis_mod16a2_data(df: pl.DataFrame) -> pl.DataFrame:
    """

    Args:
        df (pl.DataFrame): Input DataFrame with MODIS MOD16A2 columns:
                          'ET', 'PET', 'date'.

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
    
    # Apply scaling factor for ET/PET (common for MOD16A2: 0.1 for kg/m²/8day)
    # Handle fill values (e.g., 32761 to 32767 for MOD16)
    et_pet_cols = ['ET', 'PET']
    
    scaling_expressions = []
    for col in et_pet_cols:
        scaling_expressions.extend([
            # Scale by 0.1 and handle fill values
            pl.when(pl.col(col) >= 32761)
            .then(None)
            .when(pl.col(col) < -100)
            .then(None)
            .otherwise(pl.col(col).cast(pl.Float64) * 0.1)
            .alias(col)
        ])
    
    df = df.with_columns(scaling_expressions)
    
    # Derived Hydrological Features
    df = df.with_columns([
        # Evaporative Stress Index (ESI) - handle division by zero
        pl.when(pl.col("PET") == 0)
        .then(None)
        .otherwise(pl.col("ET") / pl.col("PET"))
        .alias("ESI_mod16a2"),
        
        # Actual Evapotranspiration Deficit
        (pl.col("PET") - pl.col("ET")).alias("ETD_mod16a2")
    ])
    
    # Select and reorder relevant columns
    output_columns = [
        'PID', 'ET', 'PET','ESI_mod16a2', 'ETD_mod16a2'
    ]
    
    return df.select(output_columns)

def main():
    """Main function to process MODIS MOD16A2 data and save as parquet."""
    
    # Input and output paths
    input_file = "../dataset/MODIS_MOD16A2_data.csv"  # Adjust path as needed
    output_dir = Path("../processed_data")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "processed_modis_mod16a2.parquet"
    
    try:
        # Read the data
        print(f"Reading data from {input_file}...")
        df = pl.read_csv(input_file)
        
        print(f"Original data shape: {df.shape}")
        print(f"Columns: {df.columns}")
        
        # Process the data
        print("Processing MODIS MOD16A2 data...")
        processed_df = process_modis_mod16a2_data(df)
        
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