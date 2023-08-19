# This script was used to construct the Indo-Gangetic Plain dataset with the use of already identified 189 brick kiln coordinates by us. 
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
import requests
import os
import sys

def get_grid_points(N, center_lat, center_lon, gap):
    delta = 0.0024*(N//2)
    gap = 0.0024
    top_lat = center_lat + delta
    bottom_lat = center_lat - delta
    left_lon = center_lon - delta
    right_lon = center_lon + delta

    top_row_coords = [(top_lat, lon) for lon in np.linspace(left_lon, right_lon, N)]
    bottom_row_coords = [(bottom_lat, lon) for lon in np.linspace(left_lon, right_lon, N)]

    mid_rows_coords = []
    for lat in np.linspace(top_lat - gap, bottom_lat + gap, N - 2):
        mid_rows_coords.extend([(lat, left_lon), (lat, right_lon)])
    
    all_coords = top_row_coords + mid_rows_coords + bottom_row_coords
    return all_coords

def get_satellite_image(lat, lon, zoom, size):
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": zoom,
        "size": f"{size}x{size}",
        "maptype": "satellite",
        "key": "YOUR GOOGLE MAPS API KEY",
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        return image
    else:
        print("Error fetching satellite image.")
        return None

def main(csv_path, output_dir):
    data = pd.read_csv(csv_path, header=None)
    os.makedirs(output_dir, exist_ok=True)

    for index, row in data.iterrows():
        N = 17  # Set your desired grid size
        center_lat = row[0]  # Assuming latitude is in the first column
        center_lon = row[1]  # Assuming longitude is in the second column
        gap = 0.0024  # Set your desired gap
        
        coordinates = get_grid_points(N, center_lat, center_lon, gap)
        zoom_level = 17
        image_size = 276
        
        for lat, lon in coordinates:
            image = get_satellite_image(lat, lon, zoom_level, image_size)
            if image:
                image_filename = f"{lat}_{lon}.png"
                image_path = os.path.join(output_dir, image_filename)
                image.save(image_path)
                print(f"Image saved: {image_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_to_csv> <output_directory>")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = sys.argv[2]
    main(csv_path, output_dir)
