"""
Flood Extent Mapper - Automated Flood Detection from Sentinel-1 SAR
Author: Adeola Anthonia Oyetunde
License: MIT
"""

import rasterio
import numpy as np
from rasterio.plot import show
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape

class FloodExtentMapper:
    """
    Automated flood extent detection using Sentinel-1 SAR imagery
    """
    
    def __init__(self, sar_image_path, dem_path=None):
        """
        Initialize flood mapper
        
        Parameters:
        -----------
        sar_image_path : str
            Path to Sentinel-1 SAR image (GeoTIFF)
        dem_path : str, optional
            Path to DEM for terrain masking
        """
        self.sar_path = sar_image_path
        self.dem_path = dem_path
        self.flood_extent = None
        
    def load_sar_data(self):
        """Load SAR imagery"""
        with rasterio.open(self.sar_path) as src:
            self.sar_data = src.read(1)
            self.profile = src.profile
            self.transform = src.transform
            self.crs = src.crs
        print(f"Loaded SAR data: {self.sar_data.shape}")
        return self.sar_data
    
    def preprocess(self, filter_size=5):
        """
        Apply speckle filtering to reduce noise
        
        Parameters:
        -----------
        filter_size : int
            Size of median filter kernel
        """
        print("Applying speckle filtering...")
        # Convert to dB scale
        self.sar_db = 10 * np.log10(self.sar_data + 1e-10)
        
        # Apply median filter to reduce speckle
        self.sar_filtered = median_filter(self.sar_db, size=filter_size)
        print("Preprocessing complete")
        return self.sar_filtered
    
    def detect_water(self, threshold=None):
        """
        Detect water bodies using thresholding
        
        Parameters:
        -----------
        threshold : float, optional
            Backscatter threshold in dB. If None, uses Otsu's method
        """
        print("Detecting water extent...")
        
        if threshold is None:
            # Automatic threshold using Otsu's method
            threshold = self._otsu_threshold(self.sar_filtered)
            print(f"Auto-calculated threshold: {threshold:.2f} dB")
        
        # Water has low backscatter values
        self.flood_mask = self.sar_filtered < threshold
        
        # Calculate flooded area
        pixel_area = abs(self.profile['transform'][0] * self.profile['transform'][4])  # m²
        flood_area_km2 = np.sum(self.flood_mask) * pixel_area / 1e6
        print(f"Detected flood extent: {flood_area_km2:.2f} km²")
        
        return self.flood_mask
    
    def _otsu_threshold(self, data):
        """
        Calculate optimal threshold using Otsu's method
        """
        # Flatten and remove invalid values
        valid_data = data[np.isfinite(data)].flatten()
        
        # Create histogram
        hist, bin_edges = np.histogram(valid_data, bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate weighted variance for each threshold
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]
        
        mean1 = np.cumsum(hist * bin_centers) / (weight1 + 1e-10)
        mean2 = (np.cumsum((hist * bin_centers)[::-1]) / (weight2 + 1e-10))[::-1]
        
        variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
        
        # Find threshold that maximizes between-class variance
        optimal_idx = np.argmax(variance)
        threshold = bin_centers[optimal_idx]
        
        return threshold
    
    def post_process(self, min_size=100):
        """
        Remove small isolated pixels (noise)
        
        Parameters:
        -----------
        min_size : int
            Minimum connected component size in pixels
        """
        from scipy.ndimage import label, sum as ndi_sum
        
        print("Removing small noise patches...")
        labeled, num_features = label(self.flood_mask)
        
        # Calculate size of each component
        component_sizes = ndi_sum(self.flood_mask, labeled, range(num_features + 1))
        
        # Keep only large components
        large_components = component_sizes >= min_size
        self.flood_mask = large_components[labeled]
        
        print(f"Kept {np.sum(large_components)} out of {num_features} features")
        return self.flood_mask
    
    def estimate_population_exposure(self, population_raster_path):
        """
        Estimate affected population using WorldPop or similar data
        
        Parameters:
        -----------
        population_raster_path : str
            Path to population density raster
        """
        print("Estimating population exposure...")
        
        with rasterio.open(population_raster_path) as pop_src:
            # Reproject flood mask to match population raster if needed
            # (Simplified - assumes same CRS and resolution)
            pop_data = pop_src.read(1)
            
            # Sum population in flooded areas
            affected_pop = np.sum(pop_data[self.flood_mask])
            print(f"Estimated affected population: {affected_pop:,.0f}")
            
        return affected_pop
    
    def export_shapefile(self, output_path):
        """
        Export flood extent as vector shapefile
        
        Parameters:
        -----------
        output_path : str
            Output path for shapefile
        """
        print(f"Exporting to {output_path}...")
        
        # Convert raster to vector polygons
        mask = self.flood_mask.astype('uint8')
        results = (
            {'properties': {'flood': 1}, 'geometry': s}
            for i, (s, v) in enumerate(
                shapes(mask, transform=self.transform))
            if v == 1
        )
        
        # Create GeoDataFrame
        geometries = list(results)
        gdf = gpd.GeoDataFrame.from_features(geometries, crs=self.crs)
        
        # Export
        gdf.to_file(output_path)
        print(f"Exported {len(gdf)} flood polygons")
        
        return gdf
    
    def visualize(self, output_path=None):
        """
        Create visualization of results
        
        Parameters:
        -----------
        output_path : str, optional
            If provided, saves figure to this path
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original SAR
        axes[0].imshow(self.sar_db, cmap='gray', vmin=-25, vmax=5)
        axes[0].set_title('SAR Backscatter (dB)', fontsize=14)
        axes[0].axis('off')
        
        # Filtered SAR
        axes[1].imshow(self.sar_filtered, cmap='gray', vmin=-25, vmax=5)
        axes[1].set_title('Speckle Filtered', fontsize=14)
        axes[1].axis('off')
        
        # Flood mask
        axes[2].imshow(self.flood_mask, cmap='Blues')
        axes[2].set_title('Detected Flood Extent', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        
        plt.show()


def main():
    """
    Example usage for Lokoja floods (2022)
    """
    # Initialize mapper
    mapper = FloodExtentMapper(
        sar_image_path='data/sentinel1_lokoja_20221012.tif'
    )
    
    # Load and preprocess
    mapper.load_sar_data()
    mapper.preprocess(filter_size=5)
    
    # Detect floods
    mapper.detect_water(threshold=-18)  # Typical water threshold
    mapper.post_process(min_size=100)
    
    # Export results
    mapper.export_shapefile('outputs/flood_extent_lokoja_2022.shp')
    
    # Visualize
    mapper.visualize(output_path='outputs/flood_visualization.png')
    
    # Optional: Estimate population impact
    # mapper.estimate_population_exposure('data/worldpop_nigeria.tif')


if __name__ == '__main__':
    main()
