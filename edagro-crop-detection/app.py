import os


import folium as fl
import geopandas as gpd
import numpy as np
import streamlit as st
import xarray as xr
from dotenv import load_dotenv
from earthdaily import earthdatastore
from folium.plugins import Draw
from streamlit_folium import st_folium

load_dotenv()


MAP_LIMITS = [
    0.9949025472043449,
    43.39575202159983,
    1.54204613070862,
    43.81461515749905,
]
MIN_LON, MIN_LAT, MAX_LON, MAX_LAT = MAP_LIMITS
bbox = [-96.67205734973679, 40.96151481323847, -96.20975173030152, 41.39083465413976]


def get_pos(lat, lng):
    return lat, lng


m = fl.Map(
    zoom_start=16,
    location=[MIN_LAT, MIN_LON],
    max_bounds=True,
    min_lat=MIN_LAT,
    max_lat=MAX_LAT,
    min_lon=MIN_LON,
    max_lon=MAX_LON,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="ESRI",
)
Draw().add_to(m)

fl.CircleMarker([MAX_LAT, MIN_LON], tooltip="Upper Left Corner").add_to(m)
fl.CircleMarker([MIN_LAT, MIN_LON], tooltip="Lower Left Corner").add_to(m)
fl.CircleMarker([MIN_LAT, MAX_LON], tooltip="Lower Right Corner").add_to(m)
fl.CircleMarker([MAX_LAT, MAX_LON], tooltip="Upper Right Corner").add_to(m)


map = st_folium(m, height=800, width=800)

if map.get("last_active_drawing"):
    feature = map.get("last_active_drawing")
    st.write(feature)

    gdf = gpd.GeoDataFrame.from_features([feature], crs="EPSG:4326")
    bbox = gdf.bounds.iloc[0].to_list()
    st.write(bbox)

    days_interval = 5  # one information every x days (default=5)
    year = 2020

    eds = earthdatastore.Auth()
    items = eds.search(
        "earthdaily-simulated-cloudless-l2a-cog-edagro",
        bbox=bbox,
        datetime=[f"{year}-05-15", f"{year}-10-15"],  # it excludes 1st july
        prefer_alternate="download",
        query=dict(instruments={"contains": "vnir"}),
    )

    # get only one item every 5 days  (days_interval)
    items = [items[i] for i in np.arange(0, len(items), days_interval)]
    st.write(items)
    datacube_sr = earthdatastore.datacube(
        items,
        bbox=bbox,
        assets={
            "image_file_B": "blue",
            "image_file_G": "green",
            "image_file_Y": "yellow",
            "image_file_R": "red",
            "image_file_RE1": "redege1",
            "image_file_RE2": "redege2",
            "image_file_RE3": "redege3",
            "image_file_NIR": "nir",
        },
    )
    st.write(datacube_sr)

    stats = []
    for data_var in datacube_sr:
        ds_stats = earthdatastore.cube_utils.zonal_stats_numpy(
            datacube_sr[[data_var]], gdf
        )
        print(ds_stats)
        st.markdown(ds_stats, unsafe_allow_html=True)
        stats.append(ds_stats)

    st.write(xr.merge(stats))

    ds = utils.X_year(year, to_numpy=False, return_feature_index=False)
    