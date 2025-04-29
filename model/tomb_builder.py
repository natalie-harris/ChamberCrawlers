# Creates the agents

from mesa import Agent
import random
import math

# --- Target distribution based on real data ---
TARGET_DISTRIBUTION = {
    "0_25": 0.0186,
    "25_75": 0.0904,
    "75_150": 0.1977,
    "150_250": 0.2580,
    "250_400": 0.2048,
    "400_600": 0.0895,
    "600_800": 0.0399,
    "800_1000": 0.0736,
    "1000_1200": 0.0275
}

class TombBuilder(Agent):
    def __init__(self, unique_id, model, start_lon, start_lat):
        super().__init__(unique_id, model)
        self.lon = start_lon
        self.lat = start_lat
        self.has_built = False

    def step(self):
        if self.has_built:
            return

        # Random small movement
        move_lon = random.uniform(-0.001, 0.001)
        move_lat = random.uniform(-0.001, 0.001)
        new_lon = self.lon + move_lon
        new_lat = self.lat + move_lat

        if (self.model.lon_min <= new_lon <= self.model.lon_max and
            self.model.lat_min <= new_lat <= self.model.lat_max):

            elevation = self.model.get_elevation(new_lon, new_lat)

            if elevation is None:
                return

            if 160 <= elevation <= 200:

                # Hypothetical tomb list
                hypothetical_tombs = self.model.tombs + [{
                    "lon": new_lon,
                    "lat": new_lat,
                    "elevation": elevation,
                    "builder_id": self.unique_id
                }]

                # Check if hypothetical tomb set fits target distribution
                if fits_distribution(hypothetical_tombs):
                    self.model.tombs.append({
                        "lon": new_lon,
                        "lat": new_lat,
                        "elevation": elevation,
                        "builder_id": self.unique_id
                    })
                    self.has_built = True
                    print(f"✅ Agent {self.unique_id} built tomb at ({new_lon:.5f}, {new_lat:.5f}) elev={elevation:.2f}m")

            self.lon = new_lon
            self.lat = new_lat

# --- Helper function to check if placement fits distribution ---
def fits_distribution(tombs):
    # Categorize all pairwise distances
    bins = {
        "0_25": 0,
        "25_75": 0,
        "75_150": 0,
        "150_250": 0,
        "250_400": 0,
        "400_600": 0,
        "600_800": 0,
        "800_1000": 0,
        "1000_1200": 0
    }
    total_pairs = 0

    for i in range(len(tombs)):
        for j in range(i+1, len(tombs)):
            d = haversine(tombs[i]["lon"], tombs[i]["lat"], tombs[j]["lon"], tombs[j]["lat"])
            total_pairs += 1

            if d < 25:
                bins["0_25"] += 1
            elif d < 75:
                bins["25_75"] += 1
            elif d < 150:
                bins["75_150"] += 1
            elif d < 250:
                bins["150_250"] += 1
            elif d < 400:
                bins["250_400"] += 1
            elif d < 600:
                bins["400_600"] += 1
            elif d < 800:
                bins["600_800"] += 1
            elif d < 1000:
                bins["800_1000"] += 1
            elif d < 1200:
                bins["1000_1200"] += 1

    # Compare ratios to target
    if total_pairs == 0:
        return True  # No pairs yet, allow first tombs

    for key in bins:
        actual_ratio = bins[key] / total_pairs
        target_ratio = TARGET_DISTRIBUTION[key]

        # Allow some tolerance (+5%)
        if actual_ratio > (target_ratio + 0.05):
            return False

    return True

# --- Helper: Haversine distance ---
def haversine(lon1, lat1, lon2, lat2):
    R = 6371000  # radius of Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi/2)**2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c
