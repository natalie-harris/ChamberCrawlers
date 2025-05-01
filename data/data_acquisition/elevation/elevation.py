import requests
import json
import time
import numpy as np

def get_elevation(lat, lng):
    """Queries the Open Topo Data API for elevation at a single point."""
    url = f"https://api.opentopodata.org/v1/aster30m?locations={lat},{lng}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        if data and data['status'] == 'OK' and data['results']:
            return data['results'][0]['elevation']
        else:
            print(f"Error fetching elevation for {lat},{lng}: {data.get('status')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request error for {lat},{lng}: {e}")
        return None

def create_elevation_matrix(min_lat, max_lat, min_lng, max_lng, spacing=10):
    """
    Creates an elevation matrix for the given bounding box using Open Topo Data API.

    Args:
        min_lat (float): Minimum latitude of the bounding box.
        max_lat (float): Maximum latitude of the bounding box.
        min_lng (float): Minimum longitude of the bounding box.
        max_lng (float): Maximum longitude of the bounding box.
        spacing (float): Approximate spacing between points in meters.

    Returns:
        numpy.ndarray: A 2D numpy array representing the elevation matrix, or None if an error occurs.
    """
    elevation_matrix = []
    lat_points = np.arange(max_lat, min_lat - 0.0001, -spacing / 111132) # Approx. meters to degrees latitude
    lng_points = np.arange(min_lng, max_lng + 0.0001, spacing / (111320 * np.cos(np.radians((min_lat + max_lat) / 2)))) # Approx. meters to degrees longitude

    total_calls = len(lat_points) * len(lng_points)
    if total_calls > 1000:
        print(f"Warning: The requested grid size will result in {total_calls} API calls, exceeding the daily limit of 1000.")
        return None

    print(f"Fetching elevation data for a grid of {len(lat_points)} x {len(lng_points)} points...")

    call_count = 0
    for lat in lat_points:
        row_elevations = []
        for lng in lng_points:
            elevation = get_elevation(lat, lng)
            row_elevations.append(elevation)
            call_count += 1
            if call_count % 10 == 0:
                print(f"Processed {call_count}/{total_calls} calls...")
            time.sleep(1.05) # Wait slightly longer than 1 second to be safe
        elevation_matrix.append(row_elevations)

    return np.array(elevation_matrix)

if __name__ == "__main__":
    # Approximate bounding box for the Valley of the Kings (adjust as needed)
    min_latitude = 25.73753
    max_latitude = 25.74315
    min_longitude = 32.59838
    max_longitude = 32.6047

    elevation_data = create_elevation_matrix(min_latitude, max_latitude, min_longitude, max_longitude, spacing=30)

    if elevation_data is not None:
        # Save the elevation matrix to a JSON file
        data_to_save = {"elevation_matrix": elevation_data.tolist()}
        with open("KVElevation.json", "w") as f:
            json.dump(data_to_save, f, indent=2)
        print("Elevation data saved to KVElevation.json")
    else:
        print("Failed to retrieve elevation data.")