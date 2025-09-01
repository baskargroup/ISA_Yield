import geopandas as gpd
import pandas as pd

year = '2024'

full_data = gpd.read_parquet(f'Yield_{year}.parquet')
full_data = full_data.iloc[:,1:]
full_data = full_data.drop_duplicates(subset='geometry', keep='first')
crop_class = pd.read_csv(f'Crop_Classification_{year}.csv')
# Renaming specific columns
crop_class = crop_class.rename(columns={'X_cent': 'x', 'Y_cent': 'y'})
# Using map (efficient, handles duplicates in df1)
crop_mapping = dict(zip(crop_class['Layer_ID'], crop_class['Crop']))
full_data['Crop'] = full_data['Layer_ID'].map(crop_mapping)
full_data.dropna(subset = ['Crop'],inplace=True)
# Filter df1 based on the conditions
full_data_filtered = full_data[
    # Keep rows where Crop is not Soybean or Yld_Vol_Dr is in range 0-150 for Soybean
    ((full_data['Crop'] != 'Soybean') | ((full_data['Yld_Vol_Dr'] >= 0) & (full_data['Yld_Vol_Dr'] <= 150))) &
    # Keep rows where Crop is not Corn or Yld_Vol_Dr is in range 0-550 for Corn
    ((full_data['Crop'] != 'Corn') | ((full_data['Yld_Vol_Dr'] >= 0) & (full_data['Yld_Vol_Dr'] <= 550)))
]
# full_data_filtered = full_data_filtered.set_index(['x','y'])
full_data_filtered.to_parquet(f'Yield_{2024}_filtered.parquet')