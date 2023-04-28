import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from datetime import datetime
import glob
import re
import yaml
import scipy

def preprocess(ds):
    """
    select only the lowest elevation so the open_mfdataset is less memory intensive
    """
    return ds.isel(lev=-1)

def interpolate(i, obs_data, model_mean_conc, annual_plot_data):
    """
    performs interpolation and changes to percentage
    takes in the index of the model, the observation data, the mean concentration across time and the data structure to store the results in
    returns the data structure to store the results in
    """
    model_mean_conc = model_mean_conc.mean("time") * 1000000000
    #interpolate
    model_interpolated_conc = model_mean_conc.interp(lat=obs_data["latitude"], lon=obs_data["longitude"], method='linear')
    #keep only the valid observation locations, currently after the interpolation we have many non-existent data location, and obtain a percentage
    annual_plot_data[i,:] = (np.diag(model_interpolated_conc.values) - np.array(obs_data['concentration'])) / np.array(obs_data['concentration']) * 100
    return annual_plot_data


def interpolate_seasonal(i, model_surface_conc, spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data, spring_obs, summer_obs, fall_obs, winter_obs):
    """
    performs the interpolation and changes to percentage but for select months
    takes in the index of the model, the surface observation data, the seasonal observation data, and the data structures to store the results in
    returns the data structures that store the results
    """
    # a better way would be to use datetime.season, however that throws an error
    #spring
    seasonal_surface_conc = model_surface_conc.groupby("time.season").mean("time") * 1000000000
    model_spring_conc = seasonal_surface_conc.sel(season="MAM")
    model_spring_interpolated_conc = model_spring_conc.interp(lat=spring_obs['latitude'], lon=spring_obs['longitude'], method='linear')
    spring_plot_data[i,:] = (np.diag(model_spring_interpolated_conc.values) - np.array(spring_obs['concentration'])) / np.array(spring_obs['concentration']) * 100
    #summer
    model_summer_conc = seasonal_surface_conc.sel(season="JJA")
    model_summer_interpolated_conc = model_summer_conc.interp(lat=summer_obs['latitude'], lon=summer_obs['longitude'], method='linear')
    summer_plot_data[i,:] = (np.diag(model_summer_interpolated_conc.values) - np.array(summer_obs['concentration'])) / np.array(summer_obs['concentration']) * 100
    #fall
    model_fall_conc = seasonal_surface_conc.sel(season="SON")
    model_fall_interpolated_conc = model_fall_conc.interp(lat=fall_obs['latitude'], lon=fall_obs['longitude'], method='linear')
    fall_plot_data[i,:] = (np.diag(model_fall_interpolated_conc.values) - np.array(fall_obs['concentration'])) / np.array(fall_obs['concentration']) * 100
    #winter
    model_winter_conc = seasonal_surface_conc.sel(season="DJF")
    model_winter_interpolated_conc = model_winter_conc.interp(lat=winter_obs['latitude'], lon=winter_obs['longitude'], method='linear')
    winter_plot_data[i,:] = (np.diag(model_winter_interpolated_conc.values) - np.array(winter_obs['concentration'])) / np.array(winter_obs['concentration']) * 100
    return spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data

def read_observation_data(year_start, year_end):
    """
    Reads in the observation data and return a dataframe containing the mean concentration of each station in the specified years
    The longitude values can be negative but never the latitude values
    """
    # filepaths
    filepath_us = Path("/space/hall5/sitestore/eccc/crd/ccrn/obs/slcf/AEROOBS/IMPROVE/update_improve_data/EC_ecf_improve_monthly_mean.txt")
    filepath_can = Path("/space/hall5/sitestore/eccc/crd/ccrn/obs/slcf/AEROOBS/aerodata_Cyndi/BC_CABMnew.csv")
    filepath_eu = Path("/space/hall5/sitestore/eccc/crd/ccrn/obs/slcf/AEROOBS/aerodata_Cyndi/EBAS_BC_monmean.csv")
    filepath_arctic = Path("/space/hall5/sitestore/eccc/crd/ccrn/obs/slcf/AEROOBS/ARCTIC/others/BC/Arctic_BC_others.txt")
    # read in files
    observations_us = pd.read_csv(filepath_us, sep=',')
    observations_can = pd.read_csv(filepath_can, sep=',')
    observations_eu = pd.read_csv(filepath_eu, sep=',')
    observations_arctic = pd.read_csv(filepath_arctic, sep=',')
    # create consistent columns
    observations_us.columns = ['year', 'month', 'longitude', 'latitude', 'altitude', 'concentration', 'POC',  'site_code',  'site_name']
    observations_can.columns = ['year', 'month', 'longitude', 'latitude', 'altitude', 'concentration']
    observations_eu.columns = ['year', 'month', 'longitude', 'latitude', 'altitude', 'concentration', 'site_name']
    observations_arctic.columns = ['year', 'month', 'longitude', 'latitude', 'altitude', 'concentration',  'site_name']
    # drop unnecessary columns
    observations_us = observations_us[['year', 'month', 'longitude', 'latitude', 'concentration']]
    observations_can = observations_can[['year', 'month', 'longitude', 'latitude', 'concentration']]
    observations_eu = observations_eu[['year', 'month', 'longitude', 'latitude', 'concentration']]
    observations_arctic = observations_arctic[['year', 'month', 'longitude', 'latitude', 'concentration']]
    observation_data = pd.concat([observations_us, observations_can, observations_eu, observations_arctic])
    #filter out for years
    year_adjusted_data = observation_data[observation_data["year"].isin([year_start, year_end])]
    #separate into seasons
    spring_observations = year_adjusted_data.loc[(year_adjusted_data['month'] == 3) | (year_adjusted_data['month'] == 4) | (year_adjusted_data['month'] == 5)]
    summer_observations = year_adjusted_data.loc[(year_adjusted_data['month'] == 6) | (year_adjusted_data['month'] == 7) | (year_adjusted_data['month'] == 8)]
    fall_observations = year_adjusted_data.loc[(year_adjusted_data['month'] == 9) | (year_adjusted_data['month'] == 10) | (year_adjusted_data['month'] == 11)]
    winter_observations = year_adjusted_data.loc[(year_adjusted_data['month'] == 12) | (year_adjusted_data['month'] == 1) | (year_adjusted_data['month'] == 2)]
    #group by site location to get annual concentration
    observation_sites = year_adjusted_data.groupby(['longitude', 'latitude'], as_index=False)["concentration"].mean()
    spring_observations = spring_observations.groupby(['longitude', 'latitude'], as_index=False)["concentration"].mean()
    summer_observations = summer_observations.groupby(['longitude', 'latitude'], as_index=False)["concentration"].mean()
    fall_observations = fall_observations.groupby(['longitude', 'latitude'], as_index=False)["concentration"].mean()
    winter_observations = winter_observations.groupby(['longitude', 'latitude'], as_index=False)["concentration"].mean()
    #convert longitude values
    observation_sites['longitude'] = observation_sites['longitude'] % 360
    spring_observations['longitude'] = spring_observations['longitude'] % 360
    summer_observations['longitude'] = summer_observations['longitude'] % 360
    fall_observations['longitude'] = fall_observations['longitude'] % 360
    winter_observations['longitude'] = winter_observations['longitude'] % 360
    return observation_sites, spring_observations, summer_observations, fall_observations, winter_observations

def read_models(models, year_start, year_end, obs, spring_obs, summer_obs, fall_obs, winter_obs, seasonal_flag):
    """
    Read in the model data and return annual_plot_data, a numpy ndarray that contains the data used for plotting the difference plots for each model
    Each model contains small differences in data presentation, requiring small adjustments in processing
    Models were written largely in sequence, and comments explaining functionality generally will not be repeated if the code itself is repeated
    """


    #setup variables
    obs_lat = obs['latitude']
    obs_lon = obs['longitude']
    np_obs_data = np.array(obs['concentration'])
    annual_plot_data = np.zeros((len(models), len(obs_lat)))
    spring_plot_data = np.zeros((len(models), len(spring_obs["latitude"])))
    summer_plot_data = np.zeros((len(models), len(summer_obs["latitude"])))
    fall_plot_data = np.zeros((len(models), len(fall_obs["latitude"])))
    winter_plot_data = np.zeros((len(models), len(winter_obs["latitude"])))

    #read model data
    for i, model in enumerate(models):
        if model == "CanAM5PAM":
            """
            reads canam data
            latitude ranges from -90 to 90, longitude ranges from 0 to 360
            """
            canampam_data = xr.open_dataset(f"/space/hall5/sitestore/eccc/crd/ccrn/legacy_tmp/AMAP/{model}/{model}_tp0_va_concbc_1990_2015.nc")
            #select the lowest elevation only, select the chosen years
            canampam_surface_conc = canampam_data['concbc'].isel(lev=-1).sel(time=slice(year_start, year_end))
            #interpolate
            annual_plot_data = interpolate(i, obs, canampam_surface_conc, annual_plot_data)
            if seasonal_flag:
                spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data = interpolate_seasonal(
                    i, canampam_surface_conc, spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data, spring_obs, summer_obs, fall_obs, winter_obs
                )

        if model == "CESM":
            """
            reads cesm data
            latitude ranges from 0 to 180, longitude ranges from 0 to 360
            """
            cesm_data = xr.open_dataset(f"/space/hall5/sitestore/eccc/crd/ccrn/legacy_tmp/AMAP/{model}/{model}_type0_concbc_{year_start}_{year_end}_3h.nc")
            cesm_surface_conc = cesm_data['concbc'].isel(lev=-1).sel(time=slice(year_start, year_end))
            annual_plot_data = interpolate(i, obs, cesm_surface_conc, annual_plot_data)
            if seasonal_flag:
                spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data = interpolate_seasonal(
                    i, cesm_surface_conc, spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data, spring_obs, summer_obs, fall_obs, winter_obs
                )

        if model == "DEHM":
            """
            reads dehm data
            latitude ranges from 0 to 180, longitude ranges from 0 to 360
            model elevation is reversed
            """
            dehm_data = xr.open_dataset(f"/space/hall5/sitestore/eccc/crd/ccrn/legacy_tmp/AMAP/{model}/{model}_type0_concbc_1990_2018.nc")
            dehm_surface_conc = dehm_data['concbc'].isel(lev=0).sel(time=slice(year_start, year_end))
            annual_plot_data = interpolate(i, obs, dehm_surface_conc, annual_plot_data)
            if seasonal_flag:
                spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data = interpolate_seasonal(
                    i, dehm_surface_conc, spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data, spring_obs, summer_obs, fall_obs, winter_obs
                )

        if model == "DESS-MAM7":
            #TODO: unable to drop years as it gives a dimension without coordinate
            """
            reads ciesm data
            latitude ranges from -90 to 90, longitude ranges from 0 to 360
            """
            dess_data = xr.open_dataset(f"/space/hall5/sitestore/eccc/crd/ccrn/legacy_tmp/AMAP/{model}/{model}_tp0_concbc.nc")
            dess_surface_conc = dess_data['concbc'].isel(lev=-1).sel(year=[int(year_start), int(year_end)])
            # dess_surface_conc = dess_surface_conc.rename({"month": "time"})
            # dess_surface_conc["time"] = dess_surface_conc["time"] * 28
            # dess_surface_conc["time"] = pd.to_datetime(dess_surface_conc["time"].values, unit="D", origin="2014-01-01")
            dess_mean_conc = dess_surface_conc.mean(["year", "month"]) * 1000000000
            dess_interpolated_conc = dess_mean_conc.interp(lat=obs_lat, lon=obs_lon, method='linear')
            annual_plot_data[i,:] = (np.diag(dess_interpolated_conc.values) - np_obs_data) / np_obs_data * 100
            # annual_plot_data = interpolate(i, obs, dess_surface_conc, annual_plot_data)
            # if seasonal_flag:
            #     spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data = interpolate_seasonal(
            #         i, dess_surface_conc, spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data, spring_obs, summer_obs, fall_obs, winter_obs
            #     )

        if model == "ECHAM-SALSA":
            """
            reads echam-salsa data
            latitude ranges from 0 to 90, longitude ranges from 0 to 360
            adjust chunk size for performance
            remove if fast runtime is desired, large file sizes
            """
            echam_start = xr.open_mfdataset(f"/space/hall5/sitestore/eccc/crd/ccrn/legacy_tmp/AMAP/{model}/{model}_type0_concbc_{year_start}*_NH.nc", preprocess=preprocess)
            echam_end = xr.open_mfdataset(f"/space/hall5/sitestore/eccc/crd/ccrn/legacy_tmp/AMAP/{model}/{model}_type0_concbc_{year_end}*_NH.nc", preprocess=preprocess)
            echam_data = xr.merge([echam_start, echam_end])
            echam_surface_conc = echam_data['concbc']
            annual_plot_data = interpolate(i, obs, echam_surface_conc, annual_plot_data)
            if seasonal_flag:
                spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data = interpolate_seasonal(
                    i, echam_surface_conc, spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data, spring_obs, summer_obs, fall_obs, winter_obs
                )

        # if model == "EMEP-MSCW":
        # TODO: make it work
        #     """
        #     reads emep-mscw data
        #     model elevation is reversed
        #     """
        #     emep_data = xr.open_dataset(f"/space/hall5/sitestore/eccc/crd/ccrn/legacy_tmp/AMAP/{model}/{model}_tp0_v04_concbc_3hour_{year_start}_{year_end}.nc")
        #     raw_emep_conc = emep_data['concbc']
        #     emep_mean_conc = raw_emep_conc.isel(lev=-1).sel(time=slice(year_start, year_end)).mean("time").drop("lev") * 1000000000
        #     emep_interpolated_conc = emep_mean_conc.interp(lat=obs_lat, lon=obs_lon, method='linear')
        #     annual_plot_data[i,:] = (np.diag(emep_interpolated_conc.values) - np_obs_data) / np_obs_data * 100

        if model == "FLEXPART":
            """
            reads flexpart data
            latitude ranges from -90 to 90, longitude ranges from -180 to 180
            data is stored in two separate files that must be merged
            data has no elevation, however it has additional wet and dry bc variables, lat and lon are negative as well
            """
            flex_data_start = xr.open_dataset(f"/space/hall5/sitestore/eccc/crd/ccrn/legacy_tmp/AMAP/{model}/{model}_BC_{year_start}_mon.nc")
            flex_data_end = xr.open_dataset(f"/space/hall5/sitestore/eccc/crd/ccrn/legacy_tmp/AMAP/{model}/{model}_BC_{year_end}_mon.nc")
            flex_data = xr.merge([flex_data_start, flex_data_end])
            #longitude must be rename to lon for the interpolation function
            flex_data = flex_data.rename(({'latitude': 'lat', 'longitude': 'lon'}))
            #longitude must be changed to the 0 to 360 format
            flex_data['lon'] = flex_data['lon'] % 360
            flex_surface_conc = flex_data['concbc']
            annual_plot_data = interpolate(i, obs, flex_surface_conc, annual_plot_data)
            if seasonal_flag:
                spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data = interpolate_seasonal(
                    i, flex_surface_conc, spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data, spring_obs, summer_obs, fall_obs, winter_obs
                )

        if model == "GEOS-CHEM":
            """
            reads geos-chem data
            latitude ranges from -90 to 90, longitude ranges from -180 to 180
            """
            geos_data = xr.open_dataset(f"/space/hall5/sitestore/eccc/crd/ccrn/legacy_tmp/AMAP/{model}/{model}_type0_concbc_{year_start}_{year_end}.nc")
            geos_data['lon'] = geos_data['lon'] % 360
            geos_surface_conc = geos_data['concbc'].isel(lev=0)
            geos_surface_conc["time"] = pd.to_datetime(geos_surface_conc["time"].values, unit="D", origin="2014-01-01")
            annual_plot_data = interpolate(i, obs, geos_surface_conc, annual_plot_data)
            if seasonal_flag:
                spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data = interpolate_seasonal(
                    i, geos_surface_conc, spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data, spring_obs, summer_obs, fall_obs, winter_obs
                )

        if model == "GISS-modelE-OMA":
            #TODO: same problem as dess
            """
            reads giss-data
            latitude ranges from -90 to 90, longitude ranges from -180 to 180
            time is given as years and months separately, and not in time series format
            """
            giss_data = xr.open_dataset(f"/space/hall5/sitestore/eccc/crd/ccrn/legacy_tmp/AMAP/{model}/{model}_type0_concbc_NCEP.nc")
            giss_data['lon'] = giss_data['lon'] % 360
            giss_surface_conc = giss_data['concbc'].isel(lvl=0).sel(year=[int(year_start), int(year_end)])
            giss_mean_conc = giss_surface_conc.mean(["year", "month"]) * 1000000000
            giss_interpolated_conc = giss_mean_conc.interp(lat=obs_lat, lon=obs_lon, method='linear')
            annual_plot_data[i,:] = (np.diag(giss_interpolated_conc.values) - np_obs_data) / np_obs_data * 100
            # annual_plot_data = interpolate(i, obs, giss_surface_conc, annual_plot_data)
            # if seasonal_flag:
            #     spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data = interpolate_seasonal(
            #         i, giss_surface_conc, spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data, spring_obs, summer_obs, fall_obs, winter_obs
            #     )

        if model == "MATCH":
            """
            reads match data
            latitude ranges from 0 to 180, longitude ranges from 0 to 360
            long runtime
            """
            match_data_start = xr.open_dataset(f"/space/hall5/sitestore/eccc/crd/ccrn/legacy_tmp/AMAP/{model}/{model}_Type0_concbc_{year_start}.nc")
            match_data_end = xr.open_dataset(f"/space/hall5/sitestore/eccc/crd/ccrn/legacy_tmp/AMAP/{model}/{model}_Type0_concbc_{year_end}.nc")
            match_data = xr.merge([match_data_start, match_data_end])
            match_surface_conc = match_data['concbc'].isel(lev=0)
            annual_plot_data = interpolate(i, obs, match_surface_conc, annual_plot_data)
            if seasonal_flag:
                spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data = interpolate_seasonal(
                    i, match_surface_conc, spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data, spring_obs, summer_obs, fall_obs, winter_obs
                )

        # if model == "MATCH-SALSA":
        # TODO: nonfunctional as it uses too much memory
        #     """
        #     reads match data
        #     latitude ranges from 0 to 90, longitude ranges from 0 to 360
        #     time is expressed as day as %Y%m%d.%f rather than days since 
        #     metadata in the netcdf file mentions the mapping is rotated pole, related to the projection?
        #     """
        #     match_salsa_data_start = xr.open_dataset(f"/space/hall5/sitestore/eccc/crd/ccrn/legacy_tmp/AMAP/{model}/{model}_type0_concbc_{year_start}.nc", chunks=100)
        #     match_salsa_data_end = xr.open_dataset(f"/space/hall5/sitestore/eccc/crd/ccrn/legacy_tmp/AMAP/{model}/{model}_type0_concbc_{year_end}.nc", chunks=100)
        #     match_salsa_data = xr.merge([match_salsa_data_start, match_salsa_data_end])
        #     raw_match_salsa_conc = match_salsa_data['concbc']
        #     match_salsa_mean_conc = raw_match_salsa_conc.isel(lev=0).mean("time") * 1000000000
        #     match_salsa_interpolated_conc = match_salsa_mean_conc.interp(lat=obs_lat, lon=obs_lon, method='linear')
        #     annual_plot_data[i,:] = (np.diag(match_salsa_interpolated_conc.values) - np_obs_data) / np_obs_data * 100

        if model == "MRI-ESM":
            """
            reads mri data
            latitude ranges from 0 to 180, longitude ranges from 0 to 360
            time is expressed as minutes since 2014
            """
            mri_data = xr.open_dataset(f"/space/hall5/sitestore/eccc/crd/ccrn/legacy_tmp/AMAP/{model}/{model}_type0_concbc_{year_start}-{year_end}.nc")
            mri_surface_conc = mri_data['concbc'].isel(lev=0).sel(time=slice(year_start, year_end))
            annual_plot_data = interpolate(i, obs, mri_surface_conc, annual_plot_data)
            if seasonal_flag:
                spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data = interpolate_seasonal(
                    i, mri_surface_conc, spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data, spring_obs, summer_obs, fall_obs, winter_obs
                )

        if model == "NorESM":
            """
            read noresm data
            latitude ranges from -90 to 90, longitude ranges from 0 to 360
            """
            noresm_data_start = xr.open_dataset(f"/space/hall5/sitestore/eccc/crd/ccrn/legacy_tmp/AMAP/{model}/{model}_type0_BC_{year_start}.nc")
            noresm_data_end = xr.open_dataset(f"/space/hall5/sitestore/eccc/crd/ccrn/legacy_tmp/AMAP/{model}/{model}_type0_BC_{year_end}.nc")
            noresm_data = xr.merge([noresm_data_start, noresm_data_end])
            noresm_surface_conc = noresm_data['BC'].isel(lev=-1)
            annual_plot_data = interpolate(i, obs, noresm_surface_conc, annual_plot_data)
            if seasonal_flag:
                spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data = interpolate_seasonal(
                    i, noresm_surface_conc, spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data, spring_obs, summer_obs, fall_obs, winter_obs
                )

        if model == "OsloCTM":
            #TODO: Date is only close to 2014/2015, since the metadata provided does not specify
            """
            reads oslo data
            latitude ranges from -90 to 90, longitude ranges from 0 to 360
            read in all data, not just january 2014 and december 2015
            """
            oslo_data_start = xr.open_dataset(f"/space/hall5/sitestore/eccc/crd/ccrn/legacy_tmp/AMAP/{model}/{model}_type0_concbc_monthly_{year_start}.nc")
            oslo_data_end = xr.open_dataset(f"/space/hall5/sitestore/eccc/crd/ccrn/legacy_tmp/AMAP/{model}/{model}_type0_concbc_monthly_{year_end}.nc")
            oslo_data = xr.merge([oslo_data_start, oslo_data_end])
            oslo_surface_conc = oslo_data['concbc'].isel(lev=0)
            oslo_surface_conc["time"] = pd.to_datetime(oslo_surface_conc["time"].values, unit="D", origin="2000-01-01")
            annual_plot_data = interpolate(i, obs, oslo_surface_conc, annual_plot_data)
            if seasonal_flag:
                spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data = interpolate_seasonal(
                    i, oslo_surface_conc, spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data, spring_obs, summer_obs, fall_obs, winter_obs
                )

        if model == "UKESM1":
            """
            reads ukesm data
            latitude ranges from -90 to 90, longitude ranges from 0 to 360
            """
            ukesm_data = xr.open_dataset(f"/space/hall5/sitestore/eccc/crd/ccrn/legacy_tmp/AMAP/{model}/{model}_type0_total_monthly_black_carbon_particulate_matter_2014_2015.nc")
            ukesm_data = ukesm_data.rename(({'latitude': 'lat', 'longitude': 'lon'}))
            raw_ukesm_conc = ukesm_data['total_mass_concentration_of_black_carbon_dry_aerosol_particles_in_air']
            ukesm_surface_conc = raw_ukesm_conc.isel(model_level_number=0)
            annual_plot_data = interpolate(i, obs, ukesm_surface_conc, annual_plot_data)
            if seasonal_flag:
                spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data = interpolate_seasonal(
                    i, ukesm_surface_conc, spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data, spring_obs, summer_obs, fall_obs, winter_obs
                )

        if model == "WRF-CHEM":
            #TODO: currently is not compatible with seasonal plots
            """
            reads wrf-chem data
            the dimensions of concbc for latitude and longitude, named south-north and west-east, respectively, are incorrect
            and must be replaced by the the variables of the netcdf file, xlat and xlong
            this is done by creating a new dataArray with the correct values
            xlat ranges from 0 to 90, xlong ranges from 0 to 360
            """
            wrf_data = xr.open_dataset(f"/space/hall5/sitestore/eccc/crd/ccrn/legacy_tmp/AMAP/{model}/{model}_type0_concbc.nc")
            raw_wrf_conc = wrf_data['concbc']
            wrf_da = xr.DataArray(data=raw_wrf_conc.isel(bottom_top=0).mean(dim="Time")*1000000000, 
                                  dims=["lat", "lon"], 
                                  coords=dict(lat=(["lat"], wrf_data['XLAT']), lon=(["lon"], wrf_data["XLONG"]))
                                  )
            wrf_interpolated_conc = wrf_da.interp(lat=obs_lat, lon=obs_lon, method='linear')
            annual_plot_data[i,:] = (np.diag(wrf_interpolated_conc.values) - np_obs_data) / np_obs_data * 100


    return annual_plot_data, spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data



def plot(obs_data, models, annual_plot_data):
    plot_output = '/space/hall5/sitestore/eccc/crd/ccrn/users/rlx001/figs/png_figs/'
    plot_output_eps = '/space/hall5/sitestore/eccc/crd/ccrn/users/rlx001/figs/eps_figs/'
    num_rows = 5
    num_cols = 3
    #figsize controls the vertical padding
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15,8))

    #colorbar customization
    observation_scale = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]
    model_scale = [-100,-50, -40, -30, -20, -10, 10, 20, 30, 40, 50, 100]
    colormap_obs=plt.get_cmap('YlOrRd')
    colormap_model = plt.get_cmap('seismic')
    obs_colors = colormap_obs(np.linspace(0, 1, len(observation_scale)+1))
    model_colors = colormap_model(np.linspace(0, 1, len(model_scale)+1))
    obs_cmap, obs_norm = mcolors.from_levels_and_colors(observation_scale, obs_colors, extend="both")
    model_cmap, model_norm = mcolors.from_levels_and_colors(model_scale, model_colors, extend="both")

    n = 0
    for nr in range(num_rows):
        for nc in range(num_cols):
            ax = axs[nr, nc]
            ax.set_extent([-180, 60, 0, 90], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor='lightgrey')
            ax.add_feature(cfeature.OCEAN, facecolor='lightgrey')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.2)
            # observation plot
            if not nr and not nc:
                ax.set_title("BC Observations",fontsize=9)
                pcm_obs = ax.scatter(y=obs_data['latitude'], x=obs_data['longitude'], c=obs_data['concentration'], cmap=obs_cmap, norm=obs_norm, s=1, vmin=0, vmax=2, transform=ccrs.PlateCarree())
                continue
            # remove excess plots
            if n+1 > len(models):
                fig.delaxes(ax)
                continue
            #difference plots
            ax.set_title(models[n],fontsize=9)
            pcm_diff = ax.scatter(x=obs_data['longitude'], y=obs_data['latitude'], c=annual_plot_data[n], cmap=model_cmap, norm=model_norm, s=1, vmin=-100, vmax=100, transform=ccrs.PlateCarree())
            n +=1
    

    cbaxes = fig.add_axes([0.02, 0.1, 0.01, 0.75])
    cbaxes2 = fig.add_axes([0.92, 0.1, 0.01, 0.75])
    fig.colorbar(pcm_obs, orientation='vertical',ticks=observation_scale, cax=cbaxes,pad=0.2, extend="both")
    fig.colorbar(pcm_diff, orientation='vertical',ticks=model_scale, cax=cbaxes2,pad=0.2, extend="both")
    #wspace controls the horizontal padding
    fig.subplots_adjust(wspace=0.0002)

    plt.savefig(plot_output+"bc_annual_conc.png")
    plt.savefig(plot_output_eps+"bc_annual_conc.eps")

def seasonal_plot(models, spring_obs, summer_obs, fall_obs, winter_obs, spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data):
    plot_output = '/space/hall5/sitestore/eccc/crd/ccrn/users/rlx001/figs/png_figs/'
    num_rows = 5
    num_cols = 3
    #figsize controls the vertical padding
    spring_fig, spring_axs = plt.subplots(nrows=num_rows, ncols=num_cols, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15,8))
    #colorbar customization
    observation_scale = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]
    model_scale = [-100,-50, -40, -30, -20, -10, 10, 20, 30, 40, 50, 100]
    colormap_obs=plt.get_cmap('YlOrRd')
    colormap_model = plt.get_cmap('seismic')
    obs_colors = colormap_obs(np.linspace(0, 1, len(observation_scale)+1))
    model_colors = colormap_model(np.linspace(0, 1, len(model_scale)+1))
    obs_cmap, obs_norm = mcolors.from_levels_and_colors(observation_scale, obs_colors, extend="both")
    model_cmap, model_norm = mcolors.from_levels_and_colors(model_scale, model_colors, extend="both")

    n = 0
    for nr in range(num_rows):
        for nc in range(num_cols):
            spring_ax = spring_axs[nr, nc]
            spring_ax.set_extent([-180, 60, 0, 90], crs=ccrs.PlateCarree())
            spring_ax.add_feature(cfeature.LAND, facecolor='lightgrey')
            spring_ax.add_feature(cfeature.OCEAN, facecolor='lightgrey')
            spring_ax.add_feature(cfeature.COASTLINE, linewidth=0.2)
            # observation plot
            if not nr and not nc:
                spring_ax.set_title("BC Spring Observations",fontsize=9)
                pcm_spring_obs = spring_ax.scatter(y=spring_obs['latitude'], x=spring_obs['longitude'], c=spring_obs['concentration'], cmap=obs_cmap, norm=obs_norm, s=1, vmin=0, vmax=2, transform=ccrs.PlateCarree())
                continue
            # remove excess plots
            if n+1 > len(models):
                spring_fig.delaxes(spring_ax)
                continue
            #difference plots
            spring_ax.set_title(f"{models[n]} Spring",fontsize=9)
            pcm_spring_diff = spring_ax.scatter(y=spring_obs['latitude'], x=spring_obs['longitude'], c=spring_plot_data[n], cmap=model_cmap, norm=model_norm, s=1, vmin=-100, vmax=100, transform=ccrs.PlateCarree())
            n +=1

    cbaxes = spring_fig.add_axes([0.02, 0.1, 0.01, 0.75])
    cbaxes2 = spring_fig.add_axes([0.92, 0.1, 0.01, 0.75])
    spring_fig.colorbar(pcm_spring_obs, orientation='vertical',ticks=observation_scale, cax=cbaxes,pad=0.2, extend="both")
    spring_fig.colorbar(pcm_spring_diff, orientation='vertical',ticks=model_scale, cax=cbaxes2,pad=0.2, extend="both")
    #wspace controls the horizontal padding
    spring_fig.subplots_adjust(wspace=0.0002)
    plt.savefig(plot_output+"bc_spring_conc.png")

    #figsize controls the vertical padding
    summer_fig, summer_axs = plt.subplots(nrows=num_rows, ncols=num_cols, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15,8))

    n = 0
    for nr in range(num_rows):
        for nc in range(num_cols):
            summer_ax = summer_axs[nr, nc]
            summer_ax.set_extent([-180, 60, 0, 90], crs=ccrs.PlateCarree())
            summer_ax.add_feature(cfeature.LAND, facecolor='lightgrey')
            summer_ax.add_feature(cfeature.OCEAN, facecolor='lightgrey')
            summer_ax.add_feature(cfeature.COASTLINE, linewidth=0.2)
            # observation plot
            if not nr and not nc:
                summer_ax.set_title("BC Summer Observations",fontsize=9)
                pcm_summer_obs = summer_ax.scatter(y=summer_obs['latitude'], x=summer_obs['longitude'], c=summer_obs['concentration'], cmap=obs_cmap, norm=obs_norm, s=1, vmin=0, vmax=2, transform=ccrs.PlateCarree())
                continue
            # remove excess plots
            if n+1 > len(models):
                summer_fig.delaxes(summer_ax)
                continue
            #difference plots
            summer_ax.set_title(f"{models[n]} Summer",fontsize=9)
            pcm_summer_diff = summer_ax.scatter(y=summer_obs['latitude'], x=summer_obs['longitude'], c=summer_plot_data[n], cmap=model_cmap, norm=model_norm, s=1, vmin=-100, vmax=100, transform=ccrs.PlateCarree())
            n +=1

    cbaxes = summer_fig.add_axes([0.02, 0.1, 0.01, 0.75])
    cbaxes2 = summer_fig.add_axes([0.92, 0.1, 0.01, 0.75])
    summer_fig.colorbar(pcm_summer_obs, orientation='vertical',ticks=observation_scale, cax=cbaxes,pad=0.2, extend="both")
    summer_fig.colorbar(pcm_summer_diff, orientation='vertical',ticks=model_scale, cax=cbaxes2,pad=0.2, extend="both")
    #wspace controls the horizontal padding
    summer_fig.subplots_adjust(wspace=0.0002)
    plt.savefig(plot_output+"bc_summer_conc.png")

    #figsize controls the vertical padding
    fall_fig, fall_axs = plt.subplots(nrows=num_rows, ncols=num_cols, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15,8))

    n = 0
    for nr in range(num_rows):
        for nc in range(num_cols):
            fall_ax = fall_axs[nr, nc]
            fall_ax.set_extent([-180, 60, 0, 90], crs=ccrs.PlateCarree())
            fall_ax.add_feature(cfeature.LAND, facecolor='lightgrey')
            fall_ax.add_feature(cfeature.OCEAN, facecolor='lightgrey')
            fall_ax.add_feature(cfeature.COASTLINE, linewidth=0.2)
            # observation plot
            if not nr and not nc:
                fall_ax.set_title("BC Fall Observations",fontsize=9)
                pcm_fall_obs = fall_ax.scatter(y=fall_obs['latitude'], x=fall_obs['longitude'], c=fall_obs['concentration'], cmap=obs_cmap, norm=obs_norm, s=1, vmin=0, vmax=2, transform=ccrs.PlateCarree())
                continue
            # remove excess plots
            if n+1 > len(models):
                fall_fig.delaxes(fall_ax)
                continue
            #difference plots
            fall_ax.set_title(f"{models[n]} Fall",fontsize=9)
            pcm_fall_diff = fall_ax.scatter(y=fall_obs['latitude'], x=fall_obs['longitude'], c=fall_plot_data[n], cmap=model_cmap, norm=model_norm, s=1, vmin=-100, vmax=100, transform=ccrs.PlateCarree())
            n +=1

    cbaxes = fall_fig.add_axes([0.02, 0.1, 0.01, 0.75])
    cbaxes2 = fall_fig.add_axes([0.92, 0.1, 0.01, 0.75])
    fall_fig.colorbar(pcm_fall_obs, orientation='vertical',ticks=observation_scale, cax=cbaxes,pad=0.2, extend="both")
    fall_fig.colorbar(pcm_fall_diff, orientation='vertical',ticks=model_scale, cax=cbaxes2,pad=0.2, extend="both")
    #wspace controls the horizontal padding
    fall_fig.subplots_adjust(wspace=0.0002)
    plt.savefig(plot_output+"bc_fall_conc.png")

    #figsize controls the vertical padding
    winter_fig, winter_axs = plt.subplots(nrows=num_rows, ncols=num_cols, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15,8))

    n = 0
    for nr in range(num_rows):
        for nc in range(num_cols):
            winter_ax = winter_axs[nr, nc]
            winter_ax.set_extent([-180, 60, 0, 90], crs=ccrs.PlateCarree())
            winter_ax.add_feature(cfeature.LAND, facecolor='lightgrey')
            winter_ax.add_feature(cfeature.OCEAN, facecolor='lightgrey')
            winter_ax.add_feature(cfeature.COASTLINE, linewidth=0.2)
            # observation plot
            if not nr and not nc:
                winter_ax.set_title("BC Winter Observations",fontsize=9)
                pcm_winter_obs = winter_ax.scatter(y=winter_obs['latitude'], x=winter_obs['longitude'], c=winter_obs['concentration'], cmap=obs_cmap, norm=obs_norm, s=1, vmin=0, vmax=2, transform=ccrs.PlateCarree())
                continue
            # remove excess plots
            if n+1 > len(models):
                winter_fig.delaxes(winter_ax)
                continue
            #difference plots
            winter_ax.set_title(f"{models[n]} Winter",fontsize=9)
            pcm_winter_diff = winter_ax.scatter(y=winter_obs['latitude'], x=winter_obs['longitude'], c=winter_plot_data[n], cmap=model_cmap, norm=model_norm, s=1, vmin=-100, vmax=100, transform=ccrs.PlateCarree())
            n +=1

    cbaxes = winter_fig.add_axes([0.02, 0.1, 0.01, 0.75])
    cbaxes2 = winter_fig.add_axes([0.92, 0.1, 0.01, 0.75])
    winter_fig.colorbar(pcm_winter_obs, orientation='vertical',ticks=observation_scale, cax=cbaxes,pad=0.2, extend="both")
    winter_fig.colorbar(pcm_winter_diff, orientation='vertical',ticks=model_scale, cax=cbaxes2,pad=0.2, extend="both")
    #wspace controls the horizontal padding
    winter_fig.subplots_adjust(wspace=0.0002)
    plt.savefig(plot_output+"bc_winter_conc.png")


if __name__ == "__main__":
    start = datetime.now()
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    year_start = str(config['setup_variables']['year_start'])
    year_end = str(config['setup_variables']['year_end'])
    models = config['setup_variables']['models']
    seasonal_flag = config['setup_variables']['seasonal']
    
    obs, spring_obs, summer_obs, fall_obs, winter_obs = read_observation_data(year_start, year_end)
    observations_end = datetime.now()
    print(f'reading in observations runtime: {observations_end-start}')
    annual_plot_data, spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data = read_models(models, year_start, year_end, obs, spring_obs, summer_obs, fall_obs, winter_obs, seasonal_flag)
    models_end = datetime.now()
    print(f'reading in models runtime: {models_end-observations_end}')
    plot(obs, models, annual_plot_data)
    if seasonal_flag:
        seasonal_plot(models, spring_obs, summer_obs, fall_obs, winter_obs, spring_plot_data, summer_plot_data, fall_plot_data, winter_plot_data)
    end = datetime.now()
    print(f'plotting runtime: {end-models_end}')
    print(f'total runtime: {end-start}')
    # Current supported seasonal models: [CanAM5PAM, CESM, DEHM, ECHAM-SALSA, FLEXPART, GEOS-CHEM, MATCH, MRI-ESM, NorESM, OsloCTM, UKESM1]
    # Current supported annual models (superset of seasonal models): [CanAM5PAM, CESM, DEHM, DESS-MAM7, ECHAM-SALSA, FLEXPART, GEOS-CHEM, MATCH, MRI-ESM, NorESM, OsloCTM, UKESM1, WRF-CHEM]
