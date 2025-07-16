import polars as pl
import numpy as np
from pathlib import Path
import sys

def process_modis_mod13q1_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Process MODIS MOD13Q1 (Vegetation Indices) data for soil nutrient prediction.
    
    Args:
        df (pl.DataFrame): Input DataFrame with MODIS MOD13Q1 columns:
                          'EVI', 'NDVI', 'RelativeAzimuth', 'SolarZenith', 'ViewZenith',
                          'date', 'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b07', 'PID'.

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
    
    # MODIS MOD13Q1 surface reflectance bands (scale factor 0.0001, valid range 0-10000)
    # Band 1: Red (620-670 nm)
    # Band 2: NIR (841-876 nm) 
    # Band 3: Blue (459-479 nm)
    # Band 7: SWIR (2105-2155 nm)
    reflectance_bands = ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b07']
    
    # Scale and clean reflectance bands
    refl_expressions = []
    for band in reflectance_bands:
        refl_expressions.append(
            (pl.col(band).cast(pl.Float64) * 0.0001)
            .map_elements(lambda x: None if x <= 0 or x > 1 else x, return_dtype=pl.Float64)
            .alias(band)
        )
    
    df = df.with_columns(refl_expressions)
    
    # Scale existing vegetation indices (EVI and NDVI from MOD13Q1 have scale factor 0.0001)
    df = df.with_columns([
        (pl.col("EVI").cast(pl.Float64) * 0.0001)
        .map_elements(lambda x: None if x < -0.2 or x > 1 else x, return_dtype=pl.Float64)
        .alias("EVI_scaled"),
        
        (pl.col("NDVI").cast(pl.Float64) * 0.0001)
        .map_elements(lambda x: None if x < -0.2 or x > 1 else x, return_dtype=pl.Float64)
        .alias("NDVI_scaled")
    ])
    
    # Derived Spectral Features for Soil Nutrient Analysis
    df = df.with_columns([
        # Alternative vegetation indices using surface reflectance
        # SAVI (Soil Adjusted Vegetation Index) - accounts for soil background
        (((pl.col("sur_refl_b02") - pl.col("sur_refl_b01")) / 
          (pl.col("sur_refl_b02") + pl.col("sur_refl_b01") + 0.5)) * 1.5).alias("SAVI_m13q1"),
        
        # MSAVI (Modified Soil Adjusted Vegetation Index)
        ((2 * pl.col("sur_refl_b02") + 1 - 
          ((2 * pl.col("sur_refl_b02") + 1).pow(2) - 
           8 * (pl.col("sur_refl_b02") - pl.col("sur_refl_b01"))).sqrt()) / 2).alias("MSAVI_m13q1"),
        
        # Simple Ratio (SR) - vegetation vigor
        (pl.col("sur_refl_b02") / pl.col("sur_refl_b01")).alias("SR_m13q1"),
        
        # Normalized Difference Blue-Red Index (soil/vegetation discrimination)
        ((pl.col("sur_refl_b03") - pl.col("sur_refl_b01")) / 
         (pl.col("sur_refl_b03") + pl.col("sur_refl_b01"))).alias("NDBR_m13q1"),
        
        # SWIR-based indices (sensitive to soil moisture and organic matter)
        # Normalized Difference SWIR-Red Index
        ((pl.col("sur_refl_b07") - pl.col("sur_refl_b01")) / 
         (pl.col("sur_refl_b07") + pl.col("sur_refl_b01"))).alias("NDSWIR_m13q1"),
        
        # Normalized Difference SWIR-NIR Index (soil moisture indicator)  
        ((pl.col("sur_refl_b07") - pl.col("sur_refl_b02")) / 
         (pl.col("sur_refl_b07") + pl.col("sur_refl_b02"))).alias("NDSWIR_NIR_m13q1"),
        
        # Soil-related indices
        # Brightness Index (soil brightness - organic matter proxy)
        ((pl.col("sur_refl_b01") + pl.col("sur_refl_b02") + 
          pl.col("sur_refl_b03") + pl.col("sur_refl_b07")) / 4).alias("Brightness_Index_m13q1"),
        
        # Red-Blue Ratio (soil color/iron oxide content)
        (pl.col("sur_refl_b01") / pl.col("sur_refl_b03")).alias("Red_Blue_Ratio_m13q1"),
        
        # SWIR-Blue Ratio (clay mineral detection)
        (pl.col("sur_refl_b07") / pl.col("sur_refl_b03")).alias("SWIR_Blue_Ratio_m13q1"),
        
        # Nutrient-sensitive indices
        # Chlorophyll Red Edge (nutrient stress indicator)
        (pl.col("sur_refl_b02") / pl.col("sur_refl_b01") - 1).alias("Chlorophyll_Red_Edge_m13q1"),
        
        # Nitrogen Reflectance Index (approximate, limited bands)
        ((pl.col("sur_refl_b02") - pl.col("sur_refl_b03")) / 
         (pl.col("sur_refl_b02") + pl.col("sur_refl_b03"))).alias("NRI_approx_m13q1"),
        
        # Plant Senescence Reflectance Index (nutrient stress)
        ((pl.col("sur_refl_b01") - pl.col("sur_refl_b03")) / pl.col("sur_refl_b02")).alias("PSRI_m13q1"),
        
        # Structural Independent Pigment Index (chlorophyll/nutrient content)
        ((pl.col("sur_refl_b01") - pl.col("sur_refl_b03")) / 
         (pl.col("sur_refl_b01") - pl.col("sur_refl_b07"))).alias("SIPI_m13q1"),
        
        # Moisture Stress Index (water/nutrient stress)
        (pl.col("sur_refl_b07") / pl.col("sur_refl_b02")).alias("MSI_m13q1")
    ])
    
    # Temporal Features
    df = df.with_columns([
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.ordinal_day().alias("day_of_year"),
        pl.col("date").dt.weekday().alias("day_of_week"),
        
        # Seasonal indicators
        ((pl.col("date").dt.ordinal_day() - 80) * 2 * np.pi / 365).sin().alias("season_sin"),
        ((pl.col("date").dt.ordinal_day() - 80) * 2 * np.pi / 365).cos().alias("season_cos")
    ])
    
    # Select and reorder relevant columns
    output_columns = ['PID', 'season_sin', 'season_cos']
    
    # Add scaled vegetation indices
    output_columns.extend(['EVI_scaled', 'NDVI_scaled'])
    
    # Add derived spectral indices
    output_columns.extend(['SAVI_m13q1', 'MSAVI_m13q1', 'SR_m13q1', 'NDBR_m13q1', 'NDSWIR_m13q1', 'NDSWIR_NIR_m13q1',
                          'Brightness_Index_m13q1', 'Red_Blue_Ratio_m13q1', 'SWIR_Blue_Ratio_m13q1',
                          'Chlorophyll_Red_Edge_m13q1', 'NRI_approx_m13q1', 'PSRI_m13q1', 'SIPI_m13q1', 'MSI_m13q1'])
    
    return df.select(output_columns)

def main():
    """Main function to process MODIS MOD13Q1 data."""
    
    # Check if input file is provided
    input_file = "../dataset/MODIS_MOD13Q1_data.csv"
    
    # Create output directory
    output_dir = Path("../processed_data")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Read the input file
        print(f"Reading MODIS MOD13Q1 data from: {input_file}")
        
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
        required_columns = ['EVI', 'NDVI', 'RelativeAzimuth', 'SolarZenith', 'ViewZenith', 
                           'date', 'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b07']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Process the data
        print("Processing MODIS MOD13Q1 data...")
        processed_df = process_modis_mod13q1_data(df)
        
        # Generate output filename
        input_path = Path(input_file)
        output_file = output_dir / f"processed_modis_mod13q1.parquet"
        
        # Save processed data
        print(f"Saving processed data to: {output_file}")
        processed_df.write_parquet(output_file)
        
        print(f"Processed data shape: {processed_df.shape}")
        print(f"Output columns: {processed_df.columns}")
        
        print("MODIS MOD13Q1 processing completed successfully!")
        
    except Exception as e:
        print(f"Error processing MODIS MOD13Q1 data: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()