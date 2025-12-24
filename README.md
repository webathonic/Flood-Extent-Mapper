# Flood-Extent-Mapper
Automated flood extent mapping using Sentinel-1 SAR imagery

Overview
This tool automates the extraction of flood extents from Sentinel-1 SAR imagery, reducing processing time from 6 hours to 45 minutes per scene. Originally developed for near-real-time flood assessment during Nigeria's 2022 Lokoja floods.
Features

✅ Automated Sentinel-1 SAR data download and preprocessing
✅ Flood extent detection using thresholding and change detection
✅ Population and infrastructure exposure estimation
✅ Export to multiple formats (GeoTIFF, Shapefile, GeoJSON)
✅ Batch processing for time-series analysis

# Requirements
gdal>=3.0.0
rasterio>=1.2.0
geopandas>=0.10.0
numpy>=1.20.0
matplotlib>=3.3.0
sentinelsat>=1.0.0
# Clone repository
git clone https://github.com/yourusername/flood-extent-mapper.git
cd flood-extent-mapper

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

from flood_mapper import FloodExtentMapper

# Initialize mapper
mapper = FloodExtentMapper(
    aoi='path/to/area_of_interest.geojson',
    start_date='2022-10-01',
    end_date='2022-10-15'
)

# Download Sentinel-1 data
mapper.download_sentinel1()

# Process and detect floods
flood_extent = mapper.detect_flood_extent()

# Export results
mapper.export_results(output_path='outputs/', format='geotiff')

# Generate impact report
mapper.estimate_exposure(
    population_raster='path/to/worldpop.tif',
    buildings='path/to/buildings.geojson'
)
