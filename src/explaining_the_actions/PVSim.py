from meteostat import Point, Hourly
from pvlib.location import Location
from datetime import datetime
import pandas as pd
import numpy as np
import pvlib

from pvlib.irradiance import get_total_irradiance
from tzfpy import get_tz
from PVutils import pvefficiency_adr

import matplotlib.pyplot as plt

class PVSim:
    """
    Class to represent a PVSim object.

    Attributes
    ----------
    id : int
        The id of the PVSim object.
    name : str
        The name of the PVSim object.

    Methods
    -------
    * __init__(self, id: int, name: str)
        Constructor for the PVSim class.
    * __repr__(self)
        Returns a string representation of the PVSim object.
    * __str__(self)
        Returns a string representation of the PVSim object.
    * __eq__(self, other)
        Returns True if the PVSim objects are equal, False otherwise.
    * __hash__(self)
        Returns a hash value for the PVSim object.
    * get_irradiance_data(self, start end freq)
        Returns a pandas dataframe with the columns about the irradiance data.
    * get_weather_data(self, start end freq)
        Returns a pandas dataframe with the columns about the weather data.
    * model_solar(self, pv_size, pv_efficiency, pv_azimuth, pv_tilt, pv_type)
        Returns a pandas dataframe with the columns about the solar model.
    * simulate(self,pv_size,start,end,freq,model,consider_cloud_cover,tilt,orient)
        Returns a pandas dataframe with the columns about the simulation and
        all the results of the simulation.
    """
    def __init__(self,
                id: int = 0,
                name: str = "test",
                TZ: str = None,
                lat: float = 0.0,
                lon: float = 0.0,
                alt: float = 0.0):
        self._id = id
        self._name = name
        if TZ is None:
            self._TZ = get_tz(lon, lat)
        else:
            self._TZ = TZ
        self._lat = lat
        self._lon = lon
        self._alt = alt
        #infer timezone from lat/lon
        self._location = Location(lat,
                    lon,
                    tz=self._TZ,
                    altitude=alt,
                    name=name)
        
        self._pv_size = 0
        self.results = pd.DataFrame()

    def __repr__(self):
        return f"PVSim(id={self.id}, name={self.name})"
    
    def __str__(self):
        return f"PVSim(id={self.id}, name={self.name})"
    
    def __eq__(self, other):
        return self.id == other.id and self.name == other.name
    
    def __hash__(self):
        return hash((self.id, self.name))
    
    #__________________________________________________________________________
    # Properties
    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name):
        self._name = name

    @property
    def lat(self):
        return self._lat
    
    @property
    def lon(self):
        return self._lon
    
    @property
    def alt(self):
        return self._alt
    
    @property
    def TZ(self):
        return self._TZ
    
    @TZ.setter
    def TZ(self, TZ):
        if TZ is None:
            self._TZ = get_tz(self.lat, self.lon)
        else:
            self._TZ = TZ
    
    @property
    def lat_lon_alt(self):
        return (self._lat, self._lon, self._alt)

    @lat_lon_alt.setter
    def lat_lon_alt(self, lat_lon_alt):
        self._lat = lat_lon_alt[0]
        self._lon = lat_lon_alt[1]
        self._alt = lat_lon_alt[2]
        #recalculate timezone
        self.TZ = get_tz(lat_lon_alt[0], lat_lon_alt[1])
        #recalculate location
        self._location = Location(self.lat,
                    self.lon,
                    tz=self.TZ,
                    altitude=self.alt,
                    name=self.name)
    
    @property
    def pv_size(self):
        return self._pv_size
    
    @pv_size.setter
    def pv_size(self, pv_size):
        self._pv_size = pv_size
    
    @property
    def location(self):
        return self._location

    #__________________________________________________________________________
    # Methods
    def get_irradiance_data(self,
                            start: datetime = None,
                            end: datetime = None,
                            freq: str = "15min",
                            model: str = 'ineichen'):
        """

        INPUT:
        Function takes metadata dictionary as an input and includes the following keys:
            'latitude'      ... float,
            'longitude'     ... float,
            'altitude'      ... float,
            'start_date'    ... datetime,
            'end_date'      ... datetime,
            'freq'          ... str,

        OUTPUT:
            ghi ... global horizontal irradiance
            dni ... direct normal irradiance
            dhi ... diffuse horizontal irradiance
        """
        times = pd.date_range(start=start,
                                end=end,
                                freq=freq,
                                tz=self.TZ)
        # ineichen with climatology table by default
        cs = self.location.get_clearsky(times, model=model)[:]
        cs = cs.iloc[:len(cs)-4]
        # change index to pd.DatetimeIndex
        cs.index = pd.DatetimeIndex(cs.index)
        # drop tz aware
        # cs.index = cs.index.tz_localize(None)
        self.results = pd.DataFrame({'ghi': cs['ghi'],
                        'dhi': cs['dhi'],
                        'dni': cs['dni']
                        })
        return self.results

    def get_weather_data(self,
                        start: datetime = None,
                        end: datetime = None,
                        freq: str = "15min"):
        """
        INPUT:
        Function takes metadata dictionary as an input and includes the following keys:
            'latitude'      ... float,
            'longitude'     ... float,
            'altitude'      ... float,
            'start_date'    ... datetime,
            'end_date'      ... datetime,
            'freq'          ... str,
        OUTPUT:
            weather_data ... pandas dataframe with 
            weather data that includesthe following columns:
                temp ... The air temperature in °C
                dwpt ... The dew point in °C
                rhum ... The relative humidity in percent (%)
                prcp ... The one hour precipitation total in mm
                snow ... The snow depth in mm
                wdir ... The average wind direction in degrees (°)
                wspd ... The average wind speed in km/h
                wpgt ... The peak wind gust in km/h
                pres ... The average sea-level air pressure in hPa
                tsun ... The one hour sunshine total in minutes (m)
                coco ... The weather condition code
        """
        location = Point(self.lat, self.lon, self.alt)
        weather_data = Hourly(location, 
                                start,
                                end,
                                self.TZ)
        weather_data = weather_data.fetch()
        weather_data = weather_data.iloc[:-1]
        weather_data = weather_data.resample(freq) \
                                    .mean() \
                                    .interpolate(method='linear')
        # append weather data to results
        self.results = pd.concat([self.results, weather_data], axis=1)

        return self.results

    def model_solar(self,
                    pv_size: float = 0,
                    consider_cloud_cover: bool = False,
                    tilt: int = 25,
                    orient: int = 160):
        """
        INPUT:
        Function takes metadata dictionary as an input and includes the following keys:
            'pv_size'               ... float,
            'consider_cloud_cover'  ... bool,
            'tilt'                  ... float,
            'orient'                ... float,
        OUTPUT:
            results ... pandas dataframe with
            results that includesthe following columns:
                ghi         ... global horizontal irradiance
                dni         ... direct normal irradiance
                dhi         ... diffuse horizontal irradiance
                temp        ... The air temperature in °C
                dwpt        ... The dew point in °C
                rhum        ... The relative humidity in percent (%)
                prcp        ... The one hour precipitation total in mm
                snow        ... The snow depth in mm
                wdir        ... The average wind direction in degrees (°)
                wspd        ... The average wind speed in km/h
                wpgt        ... The peak wind gust in km/h
                pres        ... The average sea-level air pressure in hPa
                tsun        ... The one hour sunshine total in minutes (m)
                coco        ... The weather condition code
                poa_global  ... Total in-plane irradiance
                temp_pv     ... temperature of the pv module
                eta_rel     ... relative efficiency of the pv module
                p_mp        ... output power of the pv array
        """
        TILT = tilt
        ORIENT = orient
        # and the irradiance level needed to achieve this output:
        G_STC = 1050.   # (W/m2)

        self.results.index = self.results.index - pd.Timedelta(minutes=7)
        

        solpos = self.location.get_solarposition(self.results.index)

        total_irrad = get_total_irradiance(TILT,
                                            ORIENT,
                                            solpos.apparent_zenith,
                                            solpos.azimuth,
                                            self.results.dni,
                                            self.results.ghi,
                                            self.results.dhi)

        self.results['poa_global'] = total_irrad.poa_global
        self.results['temp_pv'] = pvlib.temperature.faiman(self.results.poa_global,
                                                            self.results.temp,
                                                            self.results.wspd)
        # Borrow the ADR model parameters from the other example:
        # https://pvlib-python.readthedocs.io/en/stable/gallery/adr-pvarray/plot_fit_to_matrix.html
        # IEC 61853-1 standard defines a standard matrix of conditions for measurements
        adr_params = {'k_a': 0.99924,
                    'k_d': -5.49097,
                    'tc_d': 0.01918,
                    'k_rs': 0.06999,
                    'k_rsh': 0.26144
                    }

        self.results['eta_rel'] = pvefficiency_adr(self.results['poa_global'],
                                            self.results['temp_pv'],
                                            **adr_params)
        
        # parameter that is used to mask out the data 
        # when the weather condition code is worse than Overcast
        self.results["coco_mask"] = self.results["coco"].apply(
                                    lambda x: 1 if x < 2.5 
                                            else (np.random.uniform(0., 0.2) if x < 4.5 
                                                                            else 0))
        # self.results["coco_mask"] = self.results["coco_mask_min"] + self.results["coco_mask_diff"].apply(lambda x: np.random.uniform(0., 0.3))
        if consider_cloud_cover:
            #  pv_size  * scaling
            #           * relative_efficiency of the pannels
            #           * (poa_global / G_STC) - the irradiance level needed to achieve this output
            #           * percentage of minutes of sunshine per hour 
            #           * weather condition codes / hard cutoff at 3 - clowdy -  https://dev.meteostat.net/formats.html#weather-condition-codes
            
            # an numpy array of length self.results with values beteen 0.6 and 0.97 randomly distributed
            # this is used to scale the output power of the pv array
            # this is used to simulate the effect of cloud cover
            # mainly to introduce noise into the data
            scaling = np.random.uniform(0.45, 1.0, len(self.results))

            self.results['p_mp'] = pv_size * scaling \
                                * self.results['eta_rel'] \
                                * (self.results['poa_global'] / G_STC) \
                                * (self.results['tsun'] / 60) \
                                * self.results["coco_mask"]
        else:
            self.results['p_mp'] = pv_size * self.results['eta_rel'] \
                                * (self.results['poa_global'] / G_STC)
        return self.results

    def simulate(self,
                pv_size: float = 10.0,
                start: datetime = None,
                end: datetime = None,
                freq: str = "15min",
                model: str = "ineichen", # "ineichen", "haurwitz", "simplified_solis"
                consider_cloud_cover: float = False,
                tilt: int = 30,
                orient: int = 160):
        """
        INPUT:
        Function takes metadata dictionary as an input and includes the following keys:
            'pv_size' ... float,
            'start' ... datetime,
            'end' ... datetime,
            'freq' ... str,
            'model' ... str,
            'consider_cloud_cover' ... bool,
            'tilt' ... float,
            'orient' ... float,
        OUTPUT:
            results ... dataframe with results that includesthe following columns:
                temp ... The air temperature in °C
                .
                .
                .
                temp_pv ... temperature of the pv module
                p_mp ... output power of the pv array
        """
        self.get_irradiance_data(start, end, freq, model)
        self.get_weather_data(start, end, freq)
        self.model_solar(pv_size, consider_cloud_cover, tilt, orient)
        return self.results
    
    def get_average_daily_profile(self, p_mp: pd.DataFrame = None):
        """
        INPUT:
            p_mp ... output power of the pv array
        OUTPUT:
            average_daily_production ... dataframe with the average daily profile
        """
        # 2021-12-31 23:53:00+01:00 get the date and hour

        # get the average daily profile
        p_mp.index = pd.to_datetime(p_mp.index)
        average_daily_production = p_mp.groupby([p_mp.index.hour]).mean()
        return average_daily_production

if __name__ == '__main__':
    
    pv = PVSim(id=1, name="test", lat=48.1512, lon=16.9955, alt=400, TZ="Europe/Vienna")
    import time
    start_time = time.time()
    year = 2022
    start = datetime(year=year,month=1,day=1,hour=0,minute=0,second=0)
    end = datetime(year=year+1,month=1,day=1,hour=1,minute=0,second=0)
    results = pv.simulate(14000, start, end, '15min', model="ineichen", consider_cloud_cover=True)
    end_time = time.time()
    print("time elapsed: ", end_time - start_time)






    # print("Total: ", pv.results['p_mp'].sum()/1000000)
    # print("Winter:", pv.results["p_mp"][32000:].sum()/1000000 + pv.results["p_mp"][:5000].sum()/1000000)
    # # # get the sum for the summer time (june to august) and divide by 1000000 to get the value in MWh
    # print("Summer:", pv.results['p_mp'][16800:25500].sum()/1000000)
    # # get the sum for the interseason time  and divide by 1000000 to get the value in MWh
    # print("interseasion:", pv.results['p_mp'][25500:32000].sum()/1000000 + pv.results["p_mp"][5000:16800].sum()/1000000)

    # # plot single day
    # pv.results['p_mp']['2022-07-10':'2022-07-15'].plot()
    # plt.show()

    # pv.results['p_mp'][:8000].plot(label="winter", color="blue")
    # pv.results['p_mp'][32000:].plot(label="winter", color="blue")
    # pv.results['p_mp'][16800:25500].plot(label="summer", color="red")
    # pv.results['p_mp'][25500:32000].plot(label="interseason", color="orange")
    # pv.results['p_mp'][8000:16800].plot(label="interseason", color="orange")
    # plt.show()



    # print("max:",  pv.results["p_mp"].max()/1000)
