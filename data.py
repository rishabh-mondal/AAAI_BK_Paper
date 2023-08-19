import csv
import os
import requests
import sys

API_KEY = "YOUR GOOGLE MAPS API KEY"
ZOOM_LEVEL = 17
IMAGE_SIZE = "256x276"

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

def main(csv_path, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with open(csv_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            latitude, longitude, label = row
            image_url = generate_static_map_url(latitude, longitude)
            filename = f"{latitude}_{longitude}.png"
            output_path = os.path.join(output_directory, filename)
            download_image(image_url, output_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_to_csv> <output_directory>")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_directory = sys.argv[2]
    main(csv_path, output_directory)
