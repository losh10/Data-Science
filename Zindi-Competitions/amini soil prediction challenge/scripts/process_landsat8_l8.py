import polars as pl
import numpy as np
from pathlib import Path
import sys

def process_landsat8_data(df: pl.DataFrame) -> pl.DataFrame:
    """

    Args:
        df (pl.DataFrame): Input DataFrame with Landsat 8 columns:
                          'QA_PIXEL', 'QA_RADSAT', 'SR_B1', 'SR_B2', 'SR_B3', 'SR_B4',
                          'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10', 'date'.

    Returns:
        pl.DataFrame: A new DataFrame with selected original and derived features.
    """
    
    if "PID" not in df.columns:
        df = df.with_row_index("PID")
    else:
        # If PID exists, ensure it's properly formatted
        pass
    
    # Parse 'date' column
    df = df.with_columns(
        pl.col("date").str.to_datetime()
    )
    
    # Apply scaling factors for Surface Reflectance bands
    sr_bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
    
    # Scale and clean SR bands
    sr_expressions = []
    for band in sr_bands:
        sr_expressions.append(
            (pl.col(band).cast(pl.Float64) * 0.0001)
            .map_elements(lambda x: None if x == 0 or x > 1 else x, return_dtype=pl.Float64)
            .alias(band)
        )
    
    df = df.with_columns(sr_expressions)
    
    # Convert ST_B10 (Brightness Temperature) from Kelvin * 0.1 to Celsius
    df = df.with_columns([
        (pl.col("ST_B10").cast(pl.Float64) * 0.1)
        .map_elements(lambda x: None if x == 0 else x, return_dtype=pl.Float64)
        .alias("ST_B10"),
    ])
    
    # Convert to Celsius
    df = df.with_columns([
        (pl.col("ST_B10") - 273.15).alias("LST_Celsius")
    ])
    
    # Derived Spectral Features
    df = df.with_columns([
        # NDVI
        ((pl.col("SR_B5") - pl.col("SR_B4")) / (pl.col("SR_B5") + pl.col("SR_B4"))).alias("NDVI_ls8"),
        
        # EVI
        (2.5 * ((pl.col("SR_B5") - pl.col("SR_B4")) /
                (pl.col("SR_B5") + 6 * pl.col("SR_B4") - 7.5 * pl.col("SR_B2") + 1))).alias("EVI_ls8"),

        # SAVI
        (((pl.col("SR_B5") - pl.col("SR_B4")) / (pl.col("SR_B5") + pl.col("SR_B4") + 0.5)) * 1.5).alias("SAVI_ls8"),

        # NDWI
        ((pl.col("SR_B5") - pl.col("SR_B6")) / (pl.col("SR_B5") + pl.col("SR_B6"))).alias("NDWI_ls8"),
        
        # BSI
        (((pl.col("SR_B6") + pl.col("SR_B4")) - (pl.col("SR_B5") + pl.col("SR_B2"))) /
         ((pl.col("SR_B6") + pl.col("SR_B4")) + (pl.col("SR_B5") + pl.col("SR_B2")))).alias("BSI_ls8")
    ])
    
    # Temporal Features
    df = df.with_columns([
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.ordinal_day().alias("day_of_year"),
        pl.col("date").dt.weekday().alias("day_of_week")
    ])
    
    # Select and reorder relevant columns
    output_columns = ['PID', 'LST_Celsius', 'NDVI_ls8', 'EVI_ls8', 'SAVI_ls8', 'NDWI_ls8', 'BSI_ls8']

    return df.select(output_columns)

def main():
    """Main function to process Landsat 8 data."""
    
    # Check if input file is provided
    input_file = "../dataset/LANDSAT8_data_updated.csv"  # Adjust path as needed
    
    # Create output directory
    output_dir = Path("../processed_data")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Read the input file
        print(f"Reading Landsat 8 data from: {input_file}")
        
        # Determine file type and read accordingly
        if input_file.endswith('.parquet'):
            df = pl.read_parquet(input_file)
        elif input_file.endswith('.csv'):
            df = pl.read_csv(input_file)
        else:
            raise ValueError("Unsupported file format. Please use .parquet or .csv files.")
        
        print(f"Input data shape: {df.shape}")
        print(f"Input columns: {df.columns}")
        
        # Process the data
        print("Processing Landsat 8 data...")
        processed_df = process_landsat8_data(df)
        
        # Generate output filename
        input_path = Path(input_file)
        output_file = output_dir / f"processed_landsat8.parquet"
        
        # Save processed data
        print(f"Saving processed data to: {output_file}")
        processed_df.write_parquet(output_file)
        
        print(f"Processed data shape: {processed_df.shape}")
        print(f"Output columns: {processed_df.columns}")
        print("Landsat 8 processing completed successfully!")
        
    except Exception as e:
        print(f"Error processing Landsat 8 data: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()