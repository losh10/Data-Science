import polars as pl
import numpy as np
from pathlib import Path
import sys

def process_modis_mcd43a4_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Process MODIS MCD43A4 (Nadir BRDF-Adjusted Reflectance) data.
    
    Args:
        df (pl.DataFrame): Input DataFrame with MODIS MCD43A4 columns:
                          'Nadir_Reflectance_Band1', 'Nadir_Reflectance_Band2', 
                          'Nadir_Reflectance_Band3', 'Nadir_Reflectance_Band4',
                          'date', 'PID'.

    Returns:
        pl.DataFrame: A new DataFrame with selected original and derived features.
    """
    
    # Ensure PID exists, create if not present
    if "PID" not in df.columns:
        df = df.with_row_index("PID")
    else:
        # If PID exists, ensure it's properly formatted
        pass
    # Parse 'date' column
    df = df.with_columns(
        pl.col("date").str.to_datetime()
    )
    
    # MODIS MCD43A4 reflectance bands (already BRDF-adjusted)
    # Scale factor is 0.0001 with valid range 0-10000
    reflectance_bands = ['Nadir_Reflectance_Band1', 'Nadir_Reflectance_Band2', 
                        'Nadir_Reflectance_Band3', 'Nadir_Reflectance_Band4']
    
    # Scale and clean reflectance bands
    refl_expressions = []
    for band in reflectance_bands:
        refl_expressions.append(
            (pl.col(band).cast(pl.Float64) * 0.0001)
            .map_elements(lambda x: None if x <= 0 or x > 1 else x, return_dtype=pl.Float64)
            .alias(band)
        )
    
    df = df.with_columns(refl_expressions)
    
    # Derived Spectral Features
    # Note: MODIS MCD43A4 bands correspond to:
    # Band 1: Red (620-670 nm)
    # Band 2: Near-IR (841-876 nm) 
    # Band 3: Blue (459-479 nm)
    # Band 4: Green (545-565 nm)
    
    df = df.with_columns([
        # NDVI (using Band2=NIR, Band1=Red)
        ((pl.col("Nadir_Reflectance_Band2") - pl.col("Nadir_Reflectance_Band1")) / 
         (pl.col("Nadir_Reflectance_Band2") + pl.col("Nadir_Reflectance_Band1"))).alias("NDVI_mcd43a4"),
        
        # EVI (using Band2=NIR, Band1=Red, Band3=Blue)
        (2.5 * ((pl.col("Nadir_Reflectance_Band2") - pl.col("Nadir_Reflectance_Band1")) / 
                (pl.col("Nadir_Reflectance_Band2") + 6 * pl.col("Nadir_Reflectance_Band1") - 
                 7.5 * pl.col("Nadir_Reflectance_Band3") + 1))).alias("EVI_mcd43a4"),
        
        # SAVI (using Band2=NIR, Band1=Red)
        (((pl.col("Nadir_Reflectance_Band2") - pl.col("Nadir_Reflectance_Band1")) / 
          (pl.col("Nadir_Reflectance_Band2") + pl.col("Nadir_Reflectance_Band1") + 0.5)) * 1.5).alias("SAVI_mcd43a4"),
        
        # GNDVI (Green NDVI using Band2=NIR, Band4=Green)
        ((pl.col("Nadir_Reflectance_Band2") - pl.col("Nadir_Reflectance_Band4")) / 
         (pl.col("Nadir_Reflectance_Band2") + pl.col("Nadir_Reflectance_Band4"))).alias("GNDVI_mcd43a4"),
        
        # Simple Ratio (SR) - NIR/Red (vegetation vigor indicator)
        (pl.col("Nadir_Reflectance_Band2") / pl.col("Nadir_Reflectance_Band1")).alias("SR_mcd43a4"),
        
        # Normalized Difference Blue-Red Index (soil/vegetation contrast)
        ((pl.col("Nadir_Reflectance_Band3") - pl.col("Nadir_Reflectance_Band1")) / 
         (pl.col("Nadir_Reflectance_Band3") + pl.col("Nadir_Reflectance_Band1"))).alias("NDBR_mcd43a4"),
        
        # Green-Red Vegetation Index (chlorophyll content proxy)
        ((pl.col("Nadir_Reflectance_Band4") - pl.col("Nadir_Reflectance_Band1")) / 
         (pl.col("Nadir_Reflectance_Band4") + pl.col("Nadir_Reflectance_Band1"))).alias("GRVI_mcd43a4"),
        
        # Soil-related indices (limited with available bands)
        # Brightness Index (soil brightness - proxy for organic matter)
        ((pl.col("Nadir_Reflectance_Band1") + pl.col("Nadir_Reflectance_Band2") + 
          pl.col("Nadir_Reflectance_Band3") + pl.col("Nadir_Reflectance_Band4")) / 4).alias("Brightness_Index_mcd43a4"),
        
        # Red-Green Ratio (soil color indicator)
        (pl.col("Nadir_Reflectance_Band1") / pl.col("Nadir_Reflectance_Band4")).alias("Red_Green_Ratio_mcd43a4"),
        
        # Vegetation health indicators (indirect nutrient indicators)
        # Chlorophyll Index (Green-Red)
        (pl.col("Nadir_Reflectance_Band4") / pl.col("Nadir_Reflectance_Band1") - 1).alias("Chlorophyll_Index_mcd43a4"),
        
        # Stress indicators (nutrient deficiency can cause stress)
        # Blue-NIR ratio (plant stress indicator)
        (pl.col("Nadir_Reflectance_Band3") / pl.col("Nadir_Reflectance_Band2")).alias("Blue_NIR_Ratio_mcd43a4")
    ])
    
    # Select and reorder relevant columns
    output_columns = ['PID']
    
    
    # Add derived indices
    output_columns.extend(['NDVI_mcd43a4', 'EVI_mcd43a4', 'SAVI_mcd43a4', 'GNDVI_mcd43a4', 'SR_mcd43a4', 'NDBR_mcd43a4', 'GRVI_mcd43a4', 
                          'Brightness_Index_mcd43a4', 'Red_Green_Ratio_mcd43a4', 'Chlorophyll_Index_mcd43a4', 'Blue_NIR_Ratio_mcd43a4'])
    
    return df.select(output_columns)

def main():
    """Main function to process MODIS MCD43A4 data."""
    
    # Check if input file is provided
    input_file = "../dataset/MODIS_MCD43A4_data.csv"  # Adjust path as needed
    
    # Create output directory
    output_dir = Path("../processed_data")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Read the input file
        print(f"Reading MODIS MCD43A4 data from: {input_file}")
        
        # Determine file type and read accordingly
        if input_file.endswith('.parquet'):
            df = pl.read_parquet(input_file)
        elif input_file.endswith('.csv'):
            df = pl.read_csv(input_file)
        else:
            raise ValueError("Unsupported file format. Please use .parquet or .csv files.")
        
        print(f"Input data shape: {df.shape}")
        print(f"Input columns: {df.columns}")
        
        # Validate required columns
        required_columns = ['Nadir_Reflectance_Band1', 'Nadir_Reflectance_Band2', 
                           'Nadir_Reflectance_Band3', 'Nadir_Reflectance_Band4', 'date']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Process the data
        print("Processing MODIS MCD43A4 data...")
        processed_df = process_modis_mcd43a4_data(df)
        
        # Generate output filename
        input_path = Path(input_file)
        output_file = output_dir / f"processed_modis_mcd43a4.parquet"
        
        # Save processed data
        print(f"Saving processed data to: {output_file}")
        processed_df.write_parquet(output_file)
        
        print(f"Processed data shape: {processed_df.shape}")
        print(f"Output columns: {processed_df.columns}")

        print("MODIS MCD43A4 processing completed successfully!")
        
    except Exception as e:
        print(f"Error processing MODIS MCD43A4 data: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()