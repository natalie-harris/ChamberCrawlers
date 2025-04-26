import json
import pandas as pd

def convert_json_to_csv(json_file_path, csv_file_path):
    # Load the JSON data
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ---------------------------------------------------------------------------- #
    #                                   Tomb Data                                  #
    # ---------------------------------------------------------------------------- #
    flat_data = []
    for tomb in data:
        record = {
            "tomb_id": tomb.get("tomb_id"),
            "tomb_name": tomb.get("tomb_name"),
            "url": tomb.get("url"),
            "Owner": tomb.get("Owner"),
            "Decoration": ", ".join(tomb.get("Decoration", [])) if isinstance(tomb.get("Decoration"), list) else tomb.get("Decoration"),
            "Categories of Objects Recovered": ", ".join(tomb.get("Categories of Objects Recovered", [])) if isinstance(tomb.get("Categories of Objects Recovered", []), list) else tomb.get("Categories of Objects Recovered"),
        }

        # --------------------------------- Location --------------------------------- #
        # Location field (extract 'East Bank' or 'West Bank')
        location_list = tomb.get("Location", [])
        if isinstance(location_list, list):
            location_str = " | ".join(location_list)
        else:
            location_str = location_list

        # Default to None
        record["Location (Bank Side)"] = None

        # Try to detect West or East Bank
        if "West Bank" in location_str:
            record["Location (Bank Side)"] = "West Bank"
        elif "East Bank" in location_str:
            record["Location (Bank Side)"] = "East Bank"

        # --------------------------------- Elevation -------------------------------- #
        # Elevation fields
        elevation_list = tomb.get("Elevation:", [])
        if isinstance(elevation_list, list) and len(elevation_list) >= 3:
            try:
                # Elevation of -1.0 means elevation not provided.
                record["Elevation_main (m)"] = float(elevation_list[0].replace(",", ""))
                record["Northing (m)"] = float(elevation_list[1].replace(",", ""))
                record["Easting (m)"] = float(elevation_list[2].replace(",", ""))
                record["JOG Map Ref"] = (elevation_list[3]) 
                record["Modern governorate"] = (elevation_list[4]) 
                record["Ancient nome"] = (elevation_list[5]) 
            except ValueError:
                record["Elevation_main (m)"] = None
                record["Easting (m)"] = None
                record["Northing (m)"] = None
                record["JOG Map Ref"] = None
                record["Modern governorate"] = None
                record["Ancient nome"] = None
        else:
            record["Elevation_main (m)"] = None
            record["Easting (m)"] = None
            record["Northing (m)"] = None
            record["JOG Map Ref"] = None
            record["Modern governorate"] = None
            record["Ancient nome"] = None

        # ------------------------------- Measurements ------------------------------- #
        measurements = tomb.get("Measurements", [])
        if isinstance(measurements, list) and len(measurements) >= 6:
            try:
                record["Height (m)"] = float(measurements[0].replace(" m", "").replace(",", ""))
                record["Min. Width (m)"] = float(measurements[1].replace(" m", "").replace(",", ""))
                record["Max. Width (m)"] = float(measurements[2].replace(" m", "").replace(",", ""))
                record["Length (m)"] = float(measurements[3].replace(" m", "").replace(",", ""))
                record["Area (m²)"] = float(measurements[4].replace(" m²", "").replace(",", ""))
                record["Volume (m³)"] = float(measurements[5].replace(" m³", "").replace(",", ""))
            except ValueError:
                record["Height (m)"] = None
                record["Min. Width (m)"] = None
                record["Max. Width (m)"] = None
                record["Length (m)"] = None
                record["Area (m²)"] = None
                record["Volume (m³)"] = None
        else:
            record["Height (m)"] = None
            record["Min. Width (m)"] = None
            record["Max. Width (m)"] = None
            record["Length (m)"] = None
            record["Area (m²)"] = None
            record["Volume (m³)"] = None

        # ------------------------------- Addtl Tomb Info ------------------------------- #
        addtl_info = tomb.get("Additional Tomb Information", [])
        if isinstance(addtl_info, list) and len(addtl_info) >= 5:
            try:
                record["Owner Type"] = (addtl_info[0])
                record["Entrance Location"] = (addtl_info[1])
                record["Entrance Type"] = (addtl_info[2])
                record["Interior Layout"] = (addtl_info[3])
                record["Axis Type"] = (addtl_info[4])
            except ValueError:
                record["Owner Type"] = None
                record["Entrance Location"] = None
                record["Entrance Type"] = None
                record["Interior Layout"] = None
                record["Axis Type"] = None
        else:
            record["Owner Type"] = None
            record["Entrance Location"] = None
            record["Entrance Type"] = None
            record["Interior Layout"] = None
            record["Axis Type"] = None

        flat_data.append(record)

    # Create DataFrame
    df = pd.DataFrame(flat_data)

    # Save to CSV
    df.to_csv(csv_file_path, index=False, encoding="utf-8")
    print(f"CSV @ '{csv_file_path}'")

if __name__ == "__main__":
    input_json = "data/tombs_data.json"
    output_csv = "tombs_data2.csv"

    convert_json_to_csv(input_json, output_csv)
