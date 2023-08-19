# Run this script with the google maps api and the lat long csv file to download the images via google earth satellite.

import csv
import os
import requests

API_KEY = "YOUR GOOGLE MAPS API KEY"
ZOOM_LEVEL = 17
IMAGE_SIZE = "256x276"
OUTPUT_DIRECTORY = "output"

def generate_static_map_url(latitude, longitude):
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{latitude},{longitude}",
        "zoom": ZOOM_LEVEL,
        "size": IMAGE_SIZE,
        "key": API_KEY,
        'maptype': 'satellite'
    }
    return base_url + "?" + "&".join([f"{k}={v}" for k, v in params.items()])

def download_image(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Downloaded: {filename}")
    else:
        print(f"Failed to download: {filename}")

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

with open("PATH TO THE CSV", "r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        latitude, longitude, label = row
        image_url = generate_static_map_url(latitude, longitude)
        filename = f"{latitude}_{longitude}.png"
        output_path = os.path.join(OUTPUT_DIRECTORY, filename)
        download_image(image_url, output_path)