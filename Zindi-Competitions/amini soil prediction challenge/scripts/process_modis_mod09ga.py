import polars as pl
import numpy as np
from pathlib import Path
import sys

def process_modis_mod09ga_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Args:
        df (pl.DataFrame): Input DataFrame with MODIS MOD09GA columns:
                          'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b04',
                          'sur_refl_b05', 'sur_refl_b06', 'sur_refl_b07', 'date'.

    Returns:
        pl.DataFrame: A new DataFrame with selected original and derived features.
    """
    
    # Create PID from row number
    if "PID" not in df.columns:
        df = df.with_row_index("PID")
    else:
        # If PID exists, ensure it's properly formatted
        pass
    # Parse 'date' column
    df = df.with_columns(
        pl.col("date").str.to_datetime()
    )
    
    # Apply scaling factor for Surface Reflectance bands
    refl_bands = ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b04',
                  'sur_refl_b05', 'sur_refl_b06', 'sur_refl_b07']
    
    # Scale and clean reflectance bands
    refl_expressions = []
    for band in refl_bands:
        if band in df.columns:
            refl_expressions.append(
                (pl.col(band).cast(pl.Float64) * 0.0001)
                .map_elements(lambda x: None if x < 0 or x > 1 else x, return_dtype=pl.Float64)
                .alias(band)
            )
    
    df = df.with_columns(refl_expressions)
    
    # Derived Spectral Features
    df = df.with_columns([
        # NDVI (NIR - Red) / (NIR + Red)
        ((pl.col("sur_refl_b02") - pl.col("sur_refl_b01")) / 
         (pl.col("sur_refl_b02") + pl.col("sur_refl_b01"))).alias("NDVI_mod09ga"),
        
        # EVI - Enhanced Vegetation Index
        (2.5 * ((pl.col("sur_refl_b02") - pl.col("sur_refl_b01")) / 
                (pl.col("sur_refl_b02") + 6 * pl.col("sur_refl_b01") - 7.5 * pl.col("sur_refl_b03") + 1))).alias("EVI_mod09ga"),
        
        # SAVI - Soil Adjusted Vegetation Index
        (((pl.col("sur_refl_b02") - pl.col("sur_refl_b01")) / 
          (pl.col("sur_refl_b02") + pl.col("sur_refl_b01") + 0.5)) * 1.5).alias("SAVI_mod09ga"),
        
        # NDWI - Normalized Difference Water Index
        ((pl.col("sur_refl_b02") - pl.col("sur_refl_b05")) / 
         (pl.col("sur_refl_b02") + pl.col("sur_refl_b05"))).alias("NDWI_mod09ga"),
        
        # BSI - Bare Soil Index
        (((pl.col("sur_refl_b06") + pl.col("sur_refl_b01")) - (pl.col("sur_refl_b02") + pl.col("sur_refl_b03"))) /
         ((pl.col("sur_refl_b06") + pl.col("sur_refl_b01")) + (pl.col("sur_refl_b02") + pl.col("sur_refl_b03")))).alias("BSI_mod09ga"),
        
        # Additional MODIS-specific indices
        # GEMI - Global Environment Monitoring Index
        (((2 * (pl.col("sur_refl_b02")**2 - pl.col("sur_refl_b01")**2) + 1.5 * pl.col("sur_refl_b02") + 0.5 * pl.col("sur_refl_b01")) /
          (pl.col("sur_refl_b02") + pl.col("sur_refl_b01") + 0.5)) * 
         (1 - 0.25 * ((2 * (pl.col("sur_refl_b02")**2 - pl.col("sur_refl_b01")**2) + 1.5 * pl.col("sur_refl_b02") + 0.5 * pl.col("sur_refl_b01")) /
                      (pl.col("sur_refl_b02") + pl.col("sur_refl_b01") + 0.5)))).alias("GEMI_mod09ga"),
        
        # ARVI - Atmospherically Resistant Vegetation Index
        ((pl.col("sur_refl_b02") - (2 * pl.col("sur_refl_b01") - pl.col("sur_refl_b03"))) /
         (pl.col("sur_refl_b02") + (2 * pl.col("sur_refl_b01") - pl.col("sur_refl_b03")))).alias("ARVI_mod09ga"),
        
        # SIPI - Structure Insensitive Pigment Index
        ((pl.col("sur_refl_b02") - pl.col("sur_refl_b03")) / 
         (pl.col("sur_refl_b02") - pl.col("sur_refl_b01"))).alias("SIPI_mod09ga")
    ])
    
    # Select and reorder relevant columns
    output_columns = ['PID', 'NDVI_mod09ga', 'EVI_mod09ga', 'SAVI_mod09ga', 'NDWI_mod09ga', 'BSI_mod09ga', 'GEMI_mod09ga', 'ARVI_mod09ga', 'SIPI_mod09ga']

    return df.select(output_columns)

def main():
    """Main function to process MODIS MOD09GA data."""
    
    # Check if input file is provided
    input_file = "../dataset/MODIS_MOD09GA_data.csv"  # Adjust path as needed
    output_dir = Path("../processed_data")
    
    # Create output directory
    output_dir = Path("../processed_data")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Read the input file
        print(f"Reading MODIS MOD09GA data from: {input_file}")
        
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
        print("Processing MODIS MOD09GA data...")
        processed_df = process_modis_mod09ga_data(df)
        
        # Generate output filename
        input_path = Path(input_file)
        output_file = output_dir / f"processed_modis_mod09ga.parquet"
        
        # Save processed data
        print(f"Saving processed data to: {output_file}")
        processed_df.write_parquet(output_file)
        
        print(f"Processed data shape: {processed_df.shape}")
        print(f"Output columns: {processed_df.columns}")
        print("MODIS MOD09GA processing completed successfully!")
        
    except Exception as e:
        print(f"Error processing MODIS MOD09GA data: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()