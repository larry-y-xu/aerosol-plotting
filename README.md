# aerosol-plotting
**Reads in and compares the annual and seasonal differences in black carbon concentration between observation and model data

Output is in the form of difference plot image files, both annual and seasonal

The program, bc_plots.py consists of three main components. 
* The first, the function read_observation_data, reads in the observation data from stations in Canada, the United States, Europe and the Arctic. 
The year range of the data is filtered according to the variables in `config.yaml`, the mean is taken by grouping together latitude and longitude
to obtain the site concentrations of black carbon.
* The second, the function read_models reads in the model data which is treated in a similar way to the observation data to ensure they are compatible
for plotting purposes. Each of the model files have to be handled on a case to case basis due to the differences in the way they are stored in their
underlying NetCDF file format.
* To match up the gridded model data with the point based observation stations, we use bilinear interpolation provided by numpy.
* The third functions, plot and plot_seasonal use a combination of cartopy and matplotlib to plot the data

**Running the program**
1.Adjust `config.yaml`
2.Change the filepaths for the observation data, and for each of the individual models
3.Install requirement.txt with pip
4.Run the program
