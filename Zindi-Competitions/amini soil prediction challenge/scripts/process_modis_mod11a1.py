import polars as pl
import numpy as np
from pathlib import Path
import sys

def process_modis_mod11a1_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Args:
        df (pl.DataFrame): Input DataFrame with MODIS MOD11A1 columns:
                          'LST_Day_1km', 'LST_Night_1km', 'QC_Day', 'QC_Night', 
                          'Day_view_time', 'Night_view_time', 'Emis_31', 'Emis_32',
                          'Clear_day_cov', 'Clear_night_cov', 'date'.

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
    
    # Apply scaling factor for LST and convert from Kelvin to Celsius
    lst_cols = ['LST_Day_1km', 'LST_Night_1km']
    lst_expressions = []
    
    for col in lst_cols:
        if col in df.columns:
            # Scale by 0.02 and convert to Celsius
            celsius_col = f'{col}_C'
            lst_expressions.append(
                (pl.col(col).cast(pl.Float64) * 0.02 - 273.15)
                .map_elements(lambda x: None if x < -100 or x > 100 else x, return_dtype=pl.Float64)
                .alias(celsius_col)
            )
    
    df = df.with_columns(lst_expressions)
    
    # Process emissivity bands if available
    emis_expressions = []
    emis_cols = ['Emis_31', 'Emis_32']
    for col in emis_cols:
        if col in df.columns:
            emis_expressions.append(
                (pl.col(col).cast(pl.Float64) * 0.002 + 0.49)
                .map_elements(lambda x: None if x < 0.1 or x > 1.0 else x, return_dtype=pl.Float64)
                .alias(col + '_scaled')
            )
    
    if emis_expressions:
        df = df.with_columns(emis_expressions)
    
    # Process view time bands if available (scaling from hours*240 to hours)
    view_time_expressions = []
    view_time_cols = ['Day_view_time', 'Night_view_time']
    for col in view_time_cols:
        if col in df.columns:
            view_time_expressions.append(
                (pl.col(col).cast(pl.Float64) * 0.1)
                .map_elements(lambda x: None if x == 240 else x, return_dtype=pl.Float64)  # 240 is fill value
                .alias(col + '_hours')
            )
    
    if view_time_expressions:
        df = df.with_columns(view_time_expressions)
    
    # Derived LST Features - Step 1: Basic calculations
    if 'LST_Day_1km' in df.columns and 'LST_Night_1km' in df.columns:
        df = df.with_columns([
            (pl.col('LST_Day_1km_C') - pl.col('LST_Night_1km_C')).alias('DTR_C'),
            ((pl.col('LST_Day_1km_C') + pl.col('LST_Night_1km_C')) / 2).alias('Mean_LST_C')
        ])
        
        # Derived LST Features - Step 2: Features that depend on Step 1 results
        df = df.with_columns([
            # Thermal amplitude
            pl.when(pl.col('LST_Day_1km_C').is_not_null() & pl.col('LST_Night_1km_C').is_not_null())
            .then((pl.col('LST_Day_1km_C') - pl.col('LST_Night_1km_C')).abs())
            .alias('Thermal_Amplitude'),
            
            # Temperature asymmetry (positive when day heating dominates)
            pl.when(pl.col('Mean_LST_C').is_not_null())
            .then((pl.col('LST_Day_1km_C') - pl.col('Mean_LST_C')) - (pl.col('Mean_LST_C') - pl.col('LST_Night_1km_C')))
            .alias('Thermal_Asymmetry'),
            
            # Urban Heat Island potential indicator
            pl.when(pl.col('LST_Night_1km_C') > 20)  # Cities tend to stay warm at night
            .then(1).otherwise(0)
            .alias('Night_Heat_Flag')
        ])
    
    # Temporal Features
    df = df.with_columns([
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.ordinal_day().alias("day_of_year"),
        pl.col("date").dt.weekday().alias("day_of_week"),
        pl.col("date").dt.quarter().alias("quarter"),
        
        # Seasonal indicators
        pl.when(pl.col("date").dt.month().is_in([12, 1, 2])).then(pl.lit("Winter"))
        .when(pl.col("date").dt.month().is_in([3, 4, 5])).then(pl.lit("Spring"))
        .when(pl.col("date").dt.month().is_in([6, 7, 8])).then(pl.lit("Summer"))
        .otherwise(pl.lit("Fall")).alias("season")
    ])
    
    # Select and reorder relevant columns
    base_columns = ['PID', 'season']
    
    # LST columns
    # lst_columns = [col for col in df.columns if col.endswith('_C')]
    
    # Derived feature columns
    derived_columns = ['DTR_C', 'Mean_LST_C', 'Thermal_Amplitude', 'Thermal_Asymmetry', 'Night_Heat_Flag']
    derived_columns = [col for col in derived_columns if col in df.columns]
    
    # Quality and metadata columns
    qa_meta_cols = ['QC_Day', 'QC_Night', 'Clear_day_cov', 'Clear_night_cov']
    qa_meta_cols.extend([col for col in df.columns if col.endswith('_scaled') or col.endswith('_hours')])
    qa_meta_cols = [col for col in qa_meta_cols if col in df.columns]
    
    output_columns = base_columns + derived_columns + qa_meta_cols
    
    return df.select(output_columns)

def main():
    """Main function to process MODIS MOD11A1 data."""
    
    # Check if input file is provided
    input_file = "../dataset/MODIS_MOD11A1_data.csv"  # Adjust path as needed
    
    # Create output directory
    output_dir = Path("../processed_data")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Read the input file
        print(f"Reading MODIS MOD11A1 data from: {input_file}")
        
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
        print("Processing MODIS MOD11A1 data...")
        processed_df = process_modis_mod11a1_data(df)
        
        # Generate output filename
        input_path = Path(input_file)
        output_file = output_dir / f"processed_modis_mod11a1.parquet"
        
        # Save processed data
        print(f"Saving processed data to: {output_file}")
        processed_df.write_parquet(output_file)
        
        print(f"Processed data shape: {processed_df.shape}")
        print(f"Output columns: {processed_df.columns}")
        print("MODIS MOD11A1 processing completed successfully!")
        
    except Exception as e:
        print(f"Error processing MODIS MOD11A1 data: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()