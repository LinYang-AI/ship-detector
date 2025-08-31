import json
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union, transform
from shapely.affinity import affine_transform
import rasterio
from rasterio.features import shapes
from rasterio.transform import xy
import warnings

warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)


class GeoJSONExporter:
    """Export segmentation masks to GeoJSON with georeferencing."""
    
    def __init__(
        self,
        transform: Any = None,
        crs: Any = None,
        simplify_tolerance: float = 1.0,
        min_area: float = 10.0
    ):
        """
        Args:
            transform: Rasterio affine transform
            crs: Coordinate reference system
            simplify_tolerance: Douglas-Peucker simplification tolerance
            min_area: Minimum area threshold for polygons
        """
        self.transform = transform
        self.crs = crs
        self.simplify_tolerance = simplify_tolerance
        self.min_area = min_area
    
    def mask_to_polygons(
        self,
        mask: np.ndarray,
        connectivity: int = 8
    ) -> List[Polygon]:
        """Convert binary mask to list of polygons.
        
        Args:
            mask: Binary mask
            connectivity: Pixel connectivity (4 or 8)
        
        Returns:
            List of Shapely polygons
        """
        polygons = []
        
        # Ensure binary
        mask = (mask > 0.5).astype(np.uint8)
        
        # Find contours
        contours, hierarchy = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            # Skip small contours
            if cv2.contourArea(contour) < self.min_area:
                continue
            
            # Convert to polygon
            if len(contour) >= 3:
                # Reshape contour
                coords = contour.squeeze().tolist()
                
                # Ensure closed polygon
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                
                try:
                    poly = Polygon(coords)
                    
                    # Simplify if valid
                    if poly.is_valid:
                        if self.simplify_tolerance > 0:
                            poly = poly.simplify(
                                self.simplify_tolerance,
                                preserve_topology=True
                            )
                        polygons.append(poly)
                    else:
                        # Try to fix invalid polygon
                        poly = poly.buffer(0)
                        if poly.is_valid:
                            polygons.append(poly)
                
                except Exception as e:
                    print(f"Warning: Failed to create polygon: {e}")
                    continue
        
        return polygons
    
    def transform_polygons(
        self,
        polygons: List[Polygon],
        from_pixel: bool = True
    ) -> List[Polygon]:
        """Transform polygons between pixel and geographic coordinates.
        
        Args:
            polygons: List of polygons
            from_pixel: If True, transform from pixel to geographic
        
        Returns:
            Transformed polygons
        """
        if not self.transform:
            return polygons
        
        transformed = []
        
        for poly in polygons:
            if from_pixel:
                # Transform from pixel to geographic coordinates
                def pixel_to_geo(x, y):
                    return xy(self.transform, y, x)  # Note: row, col order
                
                transformed_poly = transform(pixel_to_geo, poly)
            else:
                # Transform from geographic to pixel coordinates
                def geo_to_pixel(lon, lat):
                    return ~self.transform * (lon, lat)
                
                transformed_poly = transform(geo_to_pixel, poly)
            
            transformed.append(transformed_poly)
        
        return transformed
    
    def create_feature(
        self,
        polygon: Polygon,
        properties: Optional[Dict] = None,
        feature_id: Optional[int] = None
    ) -> Dict:
        """Create a GeoJSON feature from a polygon.
        
        Args:
            polygon: Shapely polygon
            properties: Feature properties
            feature_id: Optional feature ID
        
        Returns:
            GeoJSON feature dictionary
        """
        feature = {
            "type": "Feature",
            "geometry": mapping(polygon),
            "properties": properties or {}
        }
        
        if feature_id is not None:
            feature["id"] = feature_id
        
        # Add computed properties
        feature["properties"].update({
            "area": polygon.area,
            "perimeter": polygon.length,
            "centroid": list(polygon.centroid.coords)[0],
            "bbox": polygon.bounds
        })
        
        return feature
    
    def export_mask(
        self,
        mask: np.ndarray,
        output_path: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Export mask to GeoJSON.
        
        Args:
            mask: Binary segmentation mask
            output_path: Optional path to save GeoJSON
            metadata: Additional metadata to include
        
        Returns:
            GeoJSON dictionary
        """
        # Convert mask to polygons
        polygons = self.mask_to_polygons(mask)
        
        # Transform to geographic coordinates
        if self.transform:
            polygons = self.transform_polygons(polygons, from_pixel=True)
        
        # Create features
        features = []
        for idx, poly in enumerate(polygons):
            properties = {
                "class": "ship",
                "confidence": 1.0,
                "id": idx
            }
            
            if metadata:
                properties.update(metadata)
            
            feature = self.create_feature(poly, properties, idx)
            features.append(feature)
        
        # Create FeatureCollection
        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "num_ships": len(features),
                "total_area": sum(f["properties"]["area"] for f in features),
                "generated_by": "ship-detection-pipeline"
            }
        }
        
        # Add CRS if available
        if self.crs:
            geojson["crs"] = {
                "type": "name",
                "properties": {
                    "name": str(self.crs)
                }
            }
        
        # Save if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(geojson, f, indent=2)
        
        return geojson
    
    def export_instances(
        self,
        instance_masks: List[np.ndarray],
        confidences: Optional[List[float]] = None,
        output_path: Optional[str] = None
    ) -> Dict:
        """Export multiple instance masks to GeoJSON.
        
        Args:
            instance_masks: List of binary masks for each instance
            confidences: Optional confidence scores
            output_path: Optional save path
        
        Returns:
            GeoJSON dictionary
        """
        features = []
        
        for idx, mask in enumerate(instance_masks):
            # Convert to polygons
            polygons = self.mask_to_polygons(mask)
            
            # Transform if needed
            if self.transform:
                polygons = self.transform_polygons(polygons, from_pixel=True)
            
            # Create features for this instance
            for poly in polygons:
                properties = {
                    "class": "ship",
                    "instance_id": idx,
                    "confidence": confidences[idx] if confidences else 1.0
                }
                
                feature = self.create_feature(
                    poly,
                    properties,
                    len(features)
                )
                features.append(feature)
        
        # Create FeatureCollection
        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "num_instances": len(instance_masks),
                "num_polygons": len(features)
            }
        }
        
        if self.crs:
            geojson["crs"] = {
                "type": "name",
                "properties": {"name": str(self.crs)}
            }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(geojson, f, indent=2)
        
        return geojson


def merge_geojson_tiles(
    tile_geojsons: List[Dict],
    dissolve_overlaps: bool = True,
    min_overlap_ratio: float = 0.5
) -> Dict:
    """Merge multiple GeoJSON tiles into single collection.
    
    Args:
        tile_geojsons: List of GeoJSON dictionaries
        dissolve_overlaps: Whether to merge overlapping polygons
        min_overlap_ratio: Minimum overlap ratio to merge
    
    Returns:
        Merged GeoJSON
    """
    all_features = []
    all_polygons = []
    
    # Collect all features
    for geojson in tile_geojsons:
        if 'features' in geojson:
            for feature in geojson['features']:
                geom = shape(feature['geometry'])
                all_polygons.append(geom)
                all_features.append(feature)
    
    if dissolve_overlaps and all_polygons:
        # Group overlapping polygons
        merged_polygons = []
        processed = set()
        
        for i, poly1 in enumerate(all_polygons):
            if i in processed:
                continue
            
            # Find overlapping polygons
            to_merge = [poly1]
            processed.add(i)
            
            for j, poly2 in enumerate(all_polygons[i+1:], i+1):
                if j in processed:
                    continue
                
                # Check overlap
                if poly1.intersects(poly2):
                    intersection = poly1.intersection(poly2)
                    overlap_ratio = intersection.area / min(poly1.area, poly2.area)
                    
                    if overlap_ratio >= min_overlap_ratio:
                        to_merge.append(poly2)
                        processed.add(j)
            
            # Merge overlapping polygons
            if len(to_merge) > 1:
                merged = unary_union(to_merge)
                if isinstance(merged, MultiPolygon):
                    merged_polygons.extend(list(merged.geoms))
                else:
                    merged_polygons.append(merged)
            else:
                merged_polygons.append(poly1)
        
        # Create new features from merged polygons
        features = []
        for idx, poly in enumerate(merged_polygons):
            feature = {
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": {
                    "class": "ship",
                    "id": idx,
                    "area": poly.area,
                    "perimeter": poly.length
                }
            }
            features.append(feature)
    else:
        features = all_features
    
    # Create merged GeoJSON
    merged_geojson = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "num_tiles_merged": len(tile_geojsons),
            "num_features": len(features)
        }
    }
    
    # Preserve CRS from first tile
    if tile_geojsons and 'crs' in tile_geojsons[0]:
        merged_geojson['crs'] = tile_geojsons[0]['crs']
    
    return merged_geojson


def validate_geojson(geojson: Dict) -> Tuple[bool, List[str]]:
    """Validate GeoJSON structure and geometries.
    
    Args:
        geojson: GeoJSON dictionary
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check structure
    if 'type' not in geojson:
        errors.append("Missing 'type' field")
    elif geojson['type'] != 'FeatureCollection':
        errors.append(f"Invalid type: {geojson['type']}")
    
    if 'features' not in geojson:
        errors.append("Missing 'features' field")
    else:
        # Validate each feature
        for idx, feature in enumerate(geojson['features']):
            if 'type' not in feature:
                errors.append(f"Feature {idx}: missing 'type'")
            elif feature['type'] != 'Feature':
                errors.append(f"Feature {idx}: invalid type")
            
            if 'geometry' not in feature:
                errors.append(f"Feature {idx}: missing 'geometry'")
            else:
                try:
                    # Validate geometry
                    geom = shape(feature['geometry'])
                    if not geom.is_valid:
                        errors.append(f"Feature {idx}: invalid geometry")
                except Exception as e:
                    errors.append(f"Feature {idx}: geometry error - {e}")
            
            if 'properties' not in feature:
                errors.append(f"Feature {idx}: missing 'properties'")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def geojson_to_mask(
    geojson: Dict,
    image_shape: Tuple[int, int],
    transform: Optional[Any] = None
) -> np.ndarray:
    """Convert GeoJSON back to binary mask.
    
    Args:
        geojson: GeoJSON dictionary
        image_shape: Shape of output mask
        transform: Optional transform for coordinate conversion
    
    Returns:
        Binary mask
    """
    from rasterio.features import rasterize
    
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    if 'features' not in geojson:
        return mask
    
    # Extract geometries
    geometries = []
    for feature in geojson['features']:
        if 'geometry' in feature:
            geom = shape(feature['geometry'])
            
            # Transform if needed
            if transform:
                # Convert from geographic to pixel coordinates
                inv_transform = ~transform
                
                def geo_to_pixel(x, y):
                    return inv_transform * (x, y)
                
                geom = transform(geo_to_pixel, geom)
            
            geometries.append((geom, 1))
    
    # Rasterize
    if geometries:
        mask = rasterize(
            geometries,
            out_shape=image_shape,
            dtype=np.uint8
        )
    
    return mask


def calculate_ship_statistics(geojson: Dict) -> Dict:
    """Calculate statistics from ship detections.
    
    Args:
        geojson: GeoJSON with ship polygons
    
    Returns:
        Statistics dictionary
    """
    stats = {
        'num_ships': 0,
        'total_area': 0,
        'mean_area': 0,
        'std_area': 0,
        'min_area': float('inf'),
        'max_area': 0,
        'size_distribution': {
            'small': 0,  # < 100 px
            'medium': 0,  # 100-500 px
            'large': 0    # > 500 px
        }
    }
    
    if 'features' not in geojson:
        return stats
    
    areas = []
    for feature in geojson['features']:
        if 'properties' in feature and 'area' in feature['properties']:
            area = feature['properties']['area']
            areas.append(area)
            
            # Update size distribution
            if area < 100:
                stats['size_distribution']['small'] += 1
            elif area < 500:
                stats['size_distribution']['medium'] += 1
            else:
                stats['size_distribution']['large'] += 1
    
    if areas:
        stats['num_ships'] = len(areas)
        stats['total_area'] = sum(areas)
        stats['mean_area'] = np.mean(areas)
        stats['std_area'] = np.std(areas)
        stats['min_area'] = min(areas)
        stats['max_area'] = max(areas)
    
    return stats


class CoordinateTransformer:
    """Handle coordinate transformations between different CRS."""
    
    def __init__(self, source_crs: str, target_crs: str = 'EPSG:4326'):
        """
        Args:
            source_crs: Source coordinate reference system
            target_crs: Target coordinate reference system
        """
        import pyproj
        
        self.source_crs = source_crs
        self.target_crs = target_crs
        self.transformer = pyproj.Transformer.from_crs(
            source_crs,
            target_crs,
            always_xy=True
        )
    
    def transform_geojson(self, geojson: Dict) -> Dict:
        """Transform GeoJSON to target CRS.
        
        Args:
            geojson: Input GeoJSON
        
        Returns:
            Transformed GeoJSON
        """
        transformed = geojson.copy()
        
        if 'features' in transformed:
            for feature in transformed['features']:
                if 'geometry' in feature:
                    # Transform geometry
                    geom = shape(feature['geometry'])
                    
                    # Apply transformation
                    transformed_geom = transform(
                        self.transformer.transform,
                        geom
                    )
                    
                    feature['geometry'] = mapping(transformed_geom)
        
        # Update CRS
        transformed['crs'] = {
            "type": "name",
            "properties": {"name": self.target_crs}
        }
        
        return transformed


if __name__ == "__main__":
    # Test GeoJSON export
    print("Testing GeoJSON export utilities...")
    
    # Create synthetic mask
    mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.rectangle(mask, (100, 100), (200, 200), 1, -1)
    cv2.circle(mask, (300, 300), 50, 1, -1)
    
    # Create exporter
    exporter = GeoJSONExporter(simplify_tolerance=2.0, min_area=100)
    
    # Export to GeoJSON
    geojson = exporter.export_mask(mask)
    
    print(f"Created {len(geojson['features'])} features")
    
    # Validate
    is_valid, errors = validate_geojson(geojson)
    if is_valid:
        print("GeoJSON is valid!")
    else:
        print(f"Validation errors: {errors}")
    
    # Calculate statistics
    stats = calculate_ship_statistics(geojson)
    print(f"Statistics: {stats}")
    
    # Test round-trip conversion
    mask_reconstructed = geojson_to_mask(geojson, mask.shape)
    diff = np.abs(mask - mask_reconstructed).sum()
    print(f"Round-trip difference: {diff} pixels")
    
    print("Test complete!")