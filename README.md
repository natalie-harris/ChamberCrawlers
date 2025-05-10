# ChamberCrawlers
Using Evolution-Inspired Models to Predict the Location of Hidden Tombs in the Eastern Valley of the Kings

## Usage
- Create and open a virtual environment
- You'll need to run `pip install rasterio numpy matplotlib mesa` to install the libraries 

## Why ABM
Unlike a standard statistical model (which predicts "yes/no" directly),
an ABM lets us simulate:

- Builders preferring hillsides or valleys
- Builders preferring certain elevations
- Builders preferring distances from previous tombs
- Builders preferring specific layouts (straight, bent corridors)
- Builders preferring "East Bank" or "West Bank"

This matches how tombs were actually planned historically.

# Lost Pharaoh Tombs
- Thutmose II of the 18th dynasty (found in 2022 in the Valley of the Queens)
- Ramses VIII of the 20th dynasty (still lost)

### Attributes
area: shape area (units of projection – see projections)
bounds: tuple of max and min coordinates on each axis for each shape
total_bounds: tuple of max and min coordinates on each axis for entire GeoSeries
geom_type: type of geometry.
is_valid: tests if coordinates make a shape that is reasonable geometric shape (according to this).

## Agent-Based Modeling for Archaeology - Chapter 7
https://github.com/SantaFeInstitute/ABMA/tree/master/ch7

## About the CSV
- Some values are marked as -1 or NULL meaning the information wasn't available
- Some columns are empty for the same reason
- I had to fill in some values with NULL or -1 to make the `json-to-csv.py` script work.

- I have changed the output file name and location for `json-to-csv.py` in case you choose to run it again so it does not overwrite my CSV.
- The CSV was in the `data` folder was hand-audited and added to. Both in column values and for row KV-32. 

- In regards to the dynasties, if it is a -1, it still occured within the 18-20th dynasties. 

- In regards to **Interior Layout**, I would group `corridors and chambers` with `corridors and chamber` 


