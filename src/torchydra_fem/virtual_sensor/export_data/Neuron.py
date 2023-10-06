import collections
import numpy as np
import os
import json
import h5py
import xarray as xr
from .generic_sensors import status

#------------------------------------
# @ USER DEFINED IMPORTS START
import functools
from .generic_sensors import channel_psd
# @ USER DEFINED IMPORTS END
#------------------------------------

class Neuron():

    """Data model representing the data from Morphosense Neuron system
    """

    #------------
    # Constructor
    #------------
    def __init__(self,name=None):
#------------------------------------
# @ USER DEFINED DESCRIPTION START
# @ USER DEFINED DESCRIPTION END
#------------------------------------

        #------------------------------------------------------------
        # Optional parameters are set to None by default.
        # User shall manually create adequate constructor for the
        # optional properties before using them in the code.
        #------------------------------------------------------------
        # If a property is None or if it includes any None,
        # the full property is not exported with "pro_rep",
        # i.e saved in .json or .hdf5 files.
        #------------------------------------------------------------
        self._time=np.zeros(shape=(1), dtype=float)
        self._time_psd= None
        self._Frequency_psd= None
        self._institution='Morphosense'
        self._source='openfast simulations'
        self._status_Neuron_Sensors=status.status()
        self._status_Neuron_Sensors.description = 'Neuron sensors Status description for each hour representing the next step in the process (Fetch/Allocate/waiting_merge/Done |Error).'
        self._NeurNode1ALxt_psd= None
        self._NeurNode1ALxt_psd_max_env= None
        self._NeurNode1ALxt_psd_min_env= None
        self._NeurNode2ALxt_psd= None
        self._NeurNode2ALxt_psd_max_env= None
        self._NeurNode2ALxt_psd_min_env= None
        self._NeurNode3ALxt_psd= None
        self._NeurNode3ALxt_psd_max_env= None
        self._NeurNode3ALxt_psd_min_env= None
        self._NeurNode4ALxt_psd= None
        self._NeurNode4ALxt_psd_max_env= None
        self._NeurNode4ALxt_psd_min_env= None
        self._NeurNode5ALxt_psd= None
        self._NeurNode5ALxt_psd_max_env= None
        self._NeurNode5ALxt_psd_min_env= None
        self._NeurNode6ALxt_psd= None
        self._NeurNode6ALxt_psd_max_env= None
        self._NeurNode6ALxt_psd_min_env= None
        self._NeurNode1ALyt_psd= None
        self._NeurNode1ALyt_psd_max_env= None
        self._NeurNode1ALyt_psd_min_env= None
        self._NeurNode2ALyt_psd= None
        self._NeurNode2ALyt_psd_max_env= None
        self._NeurNode2ALyt_psd_min_env= None
        self._NeurNode3ALyt_psd= None
        self._NeurNode3ALyt_psd_max_env= None
        self._NeurNode3ALyt_psd_min_env= None
        self._NeurNode4ALyt_psd= None
        self._NeurNode4ALyt_psd_max_env= None
        self._NeurNode4ALyt_psd_min_env= None
        self._NeurNode5ALyt_psd= None
        self._NeurNode5ALyt_psd_max_env= None
        self._NeurNode5ALyt_psd_min_env= None
        self._NeurNode6ALyt_psd= None
        self._NeurNode6ALyt_psd_max_env= None
        self._NeurNode6ALyt_psd_min_env= None
        self._NeurNode1ALzt_psd= None
        self._NeurNode1ALzt_psd_max_env= None
        self._NeurNode1ALzt_psd_min_env= None
        self._NeurNode2ALzt_psd= None
        self._NeurNode2ALzt_psd_max_env= None
        self._NeurNode2ALzt_psd_min_env= None
        self._NeurNode3ALzt_psd= None
        self._NeurNode3ALzt_psd_max_env= None
        self._NeurNode3ALzt_psd_min_env= None
        self._NeurNode4ALzt_psd= None
        self._NeurNode4ALzt_psd_max_env= None
        self._NeurNode4ALzt_psd_min_env= None
        self._NeurNode5ALzt_psd= None
        self._NeurNode5ALzt_psd_max_env= None
        self._NeurNode5ALzt_psd_min_env= None
        self._NeurNode6ALzt_psd= None
        self._NeurNode6ALzt_psd_max_env= None
        self._NeurNode6ALzt_psd_min_env= None
        self._name='none'
        self._description=''
        if not(name == None): # pragma: no cover
            self._name = name

#------------------------------------
# @ USER DEFINED PROPERTIES START
# @ USER DEFINED PROPERTIES END
#------------------------------------

#------------------------------------
# @ USER DEFINED METHODS START
    def rsetattr(self,obj, attr, val):
        pre, _, post = attr.rpartition('.')
        return setattr(self.rgetattr(obj, pre) if pre else obj, post, val)

    def rgetattr(self,obj, attr, *args):
        def _getattr(obj, attr):
            return getattr(obj, attr, *args)
        return functools.reduce(_getattr, [obj] + attr.split('.'))

    def allocate_ann_psd_inference(self, channel_name, unit_dictionnary, values) :
        """allocate_ann_psd_inference _summary_

        allocate the PSD values of the ANN inference on the channel_name

        :param channel_name: _description_
        :type channel_name: string
        :param values: _description_
        :type values: numpy array
        """
        values = values.astype('float64')
        self.rsetattr(self,f".{channel_name}", channel_psd.channel_psd())
        self.rsetattr(self,f"{channel_name}.short_name",f"{channel_name}")
        self.rsetattr(self,f"{channel_name}.values", values)
        self.rsetattr(self,f"{channel_name}.unit", unit_dictionnary[channel_name])

    def save_nc(self,date : str, save_path : str, ann_name :str = "ann_Neuron_Tower") -> str:
        xr_ds = self.create_xr_Dataset()
        url_base = os.path.join(f"{save_path}/{date.replace('-','/')}/ANN/{ann_name}")
        os.makedirs(url_base, exist_ok=True)
        xr_ds.to_netcdf(os.path.join(url_base,f'{ann_name}_{date}.nc'),engine='netcdf4')
        return url_base
# @ USER DEFINED METHODS END
#------------------------------------

    #------------
    # Get functions
    #------------
    @ property
    def time(self): # pragma: no cover
        """:obj:`.numpy.ndarray` of :obj:`float`:Unix Timestamp seconds since 1. January 1970 00:00:00 UTC - Representing date and hour of recording dim(*) [s]
        """
        return self._time
    #------------
    @ property
    def time_psd(self): # pragma: no cover
        """:obj:`.numpy.ndarray` of :obj:`float`:Unix Timestamp seconds since 1. January 1970 00:00:00 UTC - Representing date and hour of recording for PSD specifically dim(*) [s]
        """
        return self._time_psd
    #------------
    @ property
    def Frequency_psd(self): # pragma: no cover
        """:obj:`.numpy.ndarray` of :obj:`float`:Frequency for Power Spectral Density dim(*) [Hz]
        """
        return self._Frequency_psd
    #------------
    @ property
    def institution(self): # pragma: no cover
        """str: none
        """
        return self._institution
    #------------
    @ property
    def source(self): # pragma: no cover
        """str: none
        """
        return self._source
    #------------
    @ property
    def status_Neuron_Sensors(self): # pragma: no cover
        """:obj:`~.status.status`: Neuron sensors Status description for each hour representing the next step in the process (Fetch/Allocate/waiting_merge/Done |Error).
        """
        return self._status_Neuron_Sensors
    #------------
    @ property
    def NeurNode1ALxt_psd(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 1; Directed along the local xt-axis 
        """
        return self._NeurNode1ALxt_psd
    #------------
    @ property
    def NeurNode1ALxt_psd_max_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 1; Directed along the local xt-axis 
        """
        return self._NeurNode1ALxt_psd_max_env
    #------------
    @ property
    def NeurNode1ALxt_psd_min_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 1; Directed along the local xt-axis 
        """
        return self._NeurNode1ALxt_psd_min_env
    #------------
    @ property
    def NeurNode2ALxt_psd(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 2; Directed along the local xt-axis 
        """
        return self._NeurNode2ALxt_psd
    #------------
    @ property
    def NeurNode2ALxt_psd_max_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 2; Directed along the local xt-axis 
        """
        return self._NeurNode2ALxt_psd_max_env
    #------------
    @ property
    def NeurNode2ALxt_psd_min_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 2; Directed along the local xt-axis 
        """
        return self._NeurNode2ALxt_psd_min_env
    #------------
    @ property
    def NeurNode3ALxt_psd(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 3; Directed along the local xt-axis 
        """
        return self._NeurNode3ALxt_psd
    #------------
    @ property
    def NeurNode3ALxt_psd_max_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 3; Directed along the local xt-axis 
        """
        return self._NeurNode3ALxt_psd_max_env
    #------------
    @ property
    def NeurNode3ALxt_psd_min_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 3; Directed along the local xt-axis 
        """
        return self._NeurNode3ALxt_psd_min_env
    #------------
    @ property
    def NeurNode4ALxt_psd(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 4; Directed along the local xt-axis 
        """
        return self._NeurNode4ALxt_psd
    #------------
    @ property
    def NeurNode4ALxt_psd_max_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 4; Directed along the local xt-axis 
        """
        return self._NeurNode4ALxt_psd_max_env
    #------------
    @ property
    def NeurNode4ALxt_psd_min_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 4; Directed along the local xt-axis 
        """
        return self._NeurNode4ALxt_psd_min_env
    #------------
    @ property
    def NeurNode5ALxt_psd(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 5; Directed along the local xt-axis 
        """
        return self._NeurNode5ALxt_psd
    #------------
    @ property
    def NeurNode5ALxt_psd_max_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 5; Directed along the local xt-axis 
        """
        return self._NeurNode5ALxt_psd_max_env
    #------------
    @ property
    def NeurNode5ALxt_psd_min_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 5; Directed along the local xt-axis 
        """
        return self._NeurNode5ALxt_psd_min_env
    #------------
    @ property
    def NeurNode6ALxt_psd(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 6; Directed along the local xt-axis 
        """
        return self._NeurNode6ALxt_psd
    #------------
    @ property
    def NeurNode6ALxt_psd_max_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 6; Directed along the local xt-axis 
        """
        return self._NeurNode6ALxt_psd_max_env
    #------------
    @ property
    def NeurNode6ALxt_psd_min_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 6; Directed along the local xt-axis 
        """
        return self._NeurNode6ALxt_psd_min_env
    #------------
    @ property
    def NeurNode1ALyt_psd(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 1; Directed along the local yt-axis 
        """
        return self._NeurNode1ALyt_psd
    #------------
    @ property
    def NeurNode1ALyt_psd_max_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 1; Directed along the local yt-axis 
        """
        return self._NeurNode1ALyt_psd_max_env
    #------------
    @ property
    def NeurNode1ALyt_psd_min_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 1; Directed along the local yt-axis 
        """
        return self._NeurNode1ALyt_psd_min_env
    #------------
    @ property
    def NeurNode2ALyt_psd(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 2; Directed along the local yt-axis 
        """
        return self._NeurNode2ALyt_psd
    #------------
    @ property
    def NeurNode2ALyt_psd_max_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 2; Directed along the local yt-axis 
        """
        return self._NeurNode2ALyt_psd_max_env
    #------------
    @ property
    def NeurNode2ALyt_psd_min_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 2; Directed along the local yt-axis 
        """
        return self._NeurNode2ALyt_psd_min_env
    #------------
    @ property
    def NeurNode3ALyt_psd(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 3; Directed along the local yt-axis 
        """
        return self._NeurNode3ALyt_psd
    #------------
    @ property
    def NeurNode3ALyt_psd_max_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 3; Directed along the local yt-axis 
        """
        return self._NeurNode3ALyt_psd_max_env
    #------------
    @ property
    def NeurNode3ALyt_psd_min_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 3; Directed along the local yt-axis 
        """
        return self._NeurNode3ALyt_psd_min_env
    #------------
    @ property
    def NeurNode4ALyt_psd(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 4; Directed along the local yt-axis 
        """
        return self._NeurNode4ALyt_psd
    #------------
    @ property
    def NeurNode4ALyt_psd_max_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 4; Directed along the local yt-axis 
        """
        return self._NeurNode4ALyt_psd_max_env
    #------------
    @ property
    def NeurNode4ALyt_psd_min_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 4; Directed along the local yt-axis 
        """
        return self._NeurNode4ALyt_psd_min_env
    #------------
    @ property
    def NeurNode5ALyt_psd(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 5; Directed along the local yt-axis 
        """
        return self._NeurNode5ALyt_psd
    #------------
    @ property
    def NeurNode5ALyt_psd_max_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 5; Directed along the local yt-axis 
        """
        return self._NeurNode5ALyt_psd_max_env
    #------------
    @ property
    def NeurNode5ALyt_psd_min_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 5; Directed along the local yt-axis 
        """
        return self._NeurNode5ALyt_psd_min_env
    #------------
    @ property
    def NeurNode6ALyt_psd(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 6; Directed along the local yt-axis 
        """
        return self._NeurNode6ALyt_psd
    #------------
    @ property
    def NeurNode6ALyt_psd_max_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 6; Directed along the local yt-axis 
        """
        return self._NeurNode6ALyt_psd_max_env
    #------------
    @ property
    def NeurNode6ALyt_psd_min_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 6; Directed along the local yt-axis 
        """
        return self._NeurNode6ALyt_psd_min_env
    #------------
    @ property
    def NeurNode1ALzt_psd(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 1; Directed along the local zt-axis 
        """
        return self._NeurNode1ALzt_psd
    #------------
    @ property
    def NeurNode1ALzt_psd_max_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 1; Directed along the local zt-axis 
        """
        return self._NeurNode1ALzt_psd_max_env
    #------------
    @ property
    def NeurNode1ALzt_psd_min_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 1; Directed along the local zt-axis 
        """
        return self._NeurNode1ALzt_psd_min_env
    #------------
    @ property
    def NeurNode2ALzt_psd(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 2; Directed along the local zt-axis 
        """
        return self._NeurNode2ALzt_psd
    #------------
    @ property
    def NeurNode2ALzt_psd_max_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 2; Directed along the local zt-axis 
        """
        return self._NeurNode2ALzt_psd_max_env
    #------------
    @ property
    def NeurNode2ALzt_psd_min_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 2; Directed along the local zt-axis 
        """
        return self._NeurNode2ALzt_psd_min_env
    #------------
    @ property
    def NeurNode3ALzt_psd(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 3; Directed along the local zt-axis 
        """
        return self._NeurNode3ALzt_psd
    #------------
    @ property
    def NeurNode3ALzt_psd_max_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 3; Directed along the local zt-axis 
        """
        return self._NeurNode3ALzt_psd_max_env
    #------------
    @ property
    def NeurNode3ALzt_psd_min_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 3; Directed along the local zt-axis 
        """
        return self._NeurNode3ALzt_psd_min_env
    #------------
    @ property
    def NeurNode4ALzt_psd(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 4; Directed along the local zt-axis 
        """
        return self._NeurNode4ALzt_psd
    #------------
    @ property
    def NeurNode4ALzt_psd_max_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 4; Directed along the local zt-axis 
        """
        return self._NeurNode4ALzt_psd_max_env
    #------------
    @ property
    def NeurNode4ALzt_psd_min_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 4; Directed along the local zt-axis 
        """
        return self._NeurNode4ALzt_psd_min_env
    #------------
    @ property
    def NeurNode5ALzt_psd(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 5; Directed along the local zt-axis 
        """
        return self._NeurNode5ALzt_psd
    #------------
    @ property
    def NeurNode5ALzt_psd_max_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 5; Directed along the local zt-axis 
        """
        return self._NeurNode5ALzt_psd_max_env
    #------------
    @ property
    def NeurNode5ALzt_psd_min_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 5; Directed along the local zt-axis 
        """
        return self._NeurNode5ALzt_psd_min_env
    #------------
    @ property
    def NeurNode6ALzt_psd(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 6; Directed along the local zt-axis 
        """
        return self._NeurNode6ALzt_psd
    #------------
    @ property
    def NeurNode6ALzt_psd_max_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 6; Directed along the local zt-axis 
        """
        return self._NeurNode6ALzt_psd_max_env
    #------------
    @ property
    def NeurNode6ALzt_psd_min_env(self): # pragma: no cover
        """:obj:`~.status.status`: Estimate power spectral density using Welch’s method of tower gage 6; Directed along the local zt-axis 
        """
        return self._NeurNode6ALzt_psd_min_env
    #------------
    @ property
    def name(self): # pragma: no cover
        """str: name of the instance object
        """
        return self._name
    #------------
    @ property
    def description(self): # pragma: no cover
        """str: description of the instance object
        """
        return self._description
    #------------
    #------------
    # Set functions
    #------------
    @ time.setter
    def time(self,val): # pragma: no cover
        self._time=val
    #------------
    @ time_psd.setter
    def time_psd(self,val): # pragma: no cover
        self._time_psd=val
    #------------
    @ Frequency_psd.setter
    def Frequency_psd(self,val): # pragma: no cover
        self._Frequency_psd=val
    #------------
    @ institution.setter
    def institution(self,val): # pragma: no cover
        self._institution=str(val)
    #------------
    @ source.setter
    def source(self,val): # pragma: no cover
        self._source=str(val)
    #------------
    @ status_Neuron_Sensors.setter
    def status_Neuron_Sensors(self,val): # pragma: no cover
        self._status_Neuron_Sensors=val
    #------------
    @ NeurNode1ALxt_psd.setter
    def NeurNode1ALxt_psd(self,val): # pragma: no cover
        self._NeurNode1ALxt_psd=val
    #------------
    @ NeurNode1ALxt_psd_max_env.setter
    def NeurNode1ALxt_psd_max_env(self,val): # pragma: no cover
        self._NeurNode1ALxt_psd_max_env=val
    #------------
    @ NeurNode1ALxt_psd_min_env.setter
    def NeurNode1ALxt_psd_min_env(self,val): # pragma: no cover
        self._NeurNode1ALxt_psd_min_env=val
    #------------
    @ NeurNode2ALxt_psd.setter
    def NeurNode2ALxt_psd(self,val): # pragma: no cover
        self._NeurNode2ALxt_psd=val
    #------------
    @ NeurNode2ALxt_psd_max_env.setter
    def NeurNode2ALxt_psd_max_env(self,val): # pragma: no cover
        self._NeurNode2ALxt_psd_max_env=val
    #------------
    @ NeurNode2ALxt_psd_min_env.setter
    def NeurNode2ALxt_psd_min_env(self,val): # pragma: no cover
        self._NeurNode2ALxt_psd_min_env=val
    #------------
    @ NeurNode3ALxt_psd.setter
    def NeurNode3ALxt_psd(self,val): # pragma: no cover
        self._NeurNode3ALxt_psd=val
    #------------
    @ NeurNode3ALxt_psd_max_env.setter
    def NeurNode3ALxt_psd_max_env(self,val): # pragma: no cover
        self._NeurNode3ALxt_psd_max_env=val
    #------------
    @ NeurNode3ALxt_psd_min_env.setter
    def NeurNode3ALxt_psd_min_env(self,val): # pragma: no cover
        self._NeurNode3ALxt_psd_min_env=val
    #------------
    @ NeurNode4ALxt_psd.setter
    def NeurNode4ALxt_psd(self,val): # pragma: no cover
        self._NeurNode4ALxt_psd=val
    #------------
    @ NeurNode4ALxt_psd_max_env.setter
    def NeurNode4ALxt_psd_max_env(self,val): # pragma: no cover
        self._NeurNode4ALxt_psd_max_env=val
    #------------
    @ NeurNode4ALxt_psd_min_env.setter
    def NeurNode4ALxt_psd_min_env(self,val): # pragma: no cover
        self._NeurNode4ALxt_psd_min_env=val
    #------------
    @ NeurNode5ALxt_psd.setter
    def NeurNode5ALxt_psd(self,val): # pragma: no cover
        self._NeurNode5ALxt_psd=val
    #------------
    @ NeurNode5ALxt_psd_max_env.setter
    def NeurNode5ALxt_psd_max_env(self,val): # pragma: no cover
        self._NeurNode5ALxt_psd_max_env=val
    #------------
    @ NeurNode5ALxt_psd_min_env.setter
    def NeurNode5ALxt_psd_min_env(self,val): # pragma: no cover
        self._NeurNode5ALxt_psd_min_env=val
    #------------
    @ NeurNode6ALxt_psd.setter
    def NeurNode6ALxt_psd(self,val): # pragma: no cover
        self._NeurNode6ALxt_psd=val
    #------------
    @ NeurNode6ALxt_psd_max_env.setter
    def NeurNode6ALxt_psd_max_env(self,val): # pragma: no cover
        self._NeurNode6ALxt_psd_max_env=val
    #------------
    @ NeurNode6ALxt_psd_min_env.setter
    def NeurNode6ALxt_psd_min_env(self,val): # pragma: no cover
        self._NeurNode6ALxt_psd_min_env=val
    #------------
    @ NeurNode1ALyt_psd.setter
    def NeurNode1ALyt_psd(self,val): # pragma: no cover
        self._NeurNode1ALyt_psd=val
    #------------
    @ NeurNode1ALyt_psd_max_env.setter
    def NeurNode1ALyt_psd_max_env(self,val): # pragma: no cover
        self._NeurNode1ALyt_psd_max_env=val
    #------------
    @ NeurNode1ALyt_psd_min_env.setter
    def NeurNode1ALyt_psd_min_env(self,val): # pragma: no cover
        self._NeurNode1ALyt_psd_min_env=val
    #------------
    @ NeurNode2ALyt_psd.setter
    def NeurNode2ALyt_psd(self,val): # pragma: no cover
        self._NeurNode2ALyt_psd=val
    #------------
    @ NeurNode2ALyt_psd_max_env.setter
    def NeurNode2ALyt_psd_max_env(self,val): # pragma: no cover
        self._NeurNode2ALyt_psd_max_env=val
    #------------
    @ NeurNode2ALyt_psd_min_env.setter
    def NeurNode2ALyt_psd_min_env(self,val): # pragma: no cover
        self._NeurNode2ALyt_psd_min_env=val
    #------------
    @ NeurNode3ALyt_psd.setter
    def NeurNode3ALyt_psd(self,val): # pragma: no cover
        self._NeurNode3ALyt_psd=val
    #------------
    @ NeurNode3ALyt_psd_max_env.setter
    def NeurNode3ALyt_psd_max_env(self,val): # pragma: no cover
        self._NeurNode3ALyt_psd_max_env=val
    #------------
    @ NeurNode3ALyt_psd_min_env.setter
    def NeurNode3ALyt_psd_min_env(self,val): # pragma: no cover
        self._NeurNode3ALyt_psd_min_env=val
    #------------
    @ NeurNode4ALyt_psd.setter
    def NeurNode4ALyt_psd(self,val): # pragma: no cover
        self._NeurNode4ALyt_psd=val
    #------------
    @ NeurNode4ALyt_psd_max_env.setter
    def NeurNode4ALyt_psd_max_env(self,val): # pragma: no cover
        self._NeurNode4ALyt_psd_max_env=val
    #------------
    @ NeurNode4ALyt_psd_min_env.setter
    def NeurNode4ALyt_psd_min_env(self,val): # pragma: no cover
        self._NeurNode4ALyt_psd_min_env=val
    #------------
    @ NeurNode5ALyt_psd.setter
    def NeurNode5ALyt_psd(self,val): # pragma: no cover
        self._NeurNode5ALyt_psd=val
    #------------
    @ NeurNode5ALyt_psd_max_env.setter
    def NeurNode5ALyt_psd_max_env(self,val): # pragma: no cover
        self._NeurNode5ALyt_psd_max_env=val
    #------------
    @ NeurNode5ALyt_psd_min_env.setter
    def NeurNode5ALyt_psd_min_env(self,val): # pragma: no cover
        self._NeurNode5ALyt_psd_min_env=val
    #------------
    @ NeurNode6ALyt_psd.setter
    def NeurNode6ALyt_psd(self,val): # pragma: no cover
        self._NeurNode6ALyt_psd=val
    #------------
    @ NeurNode6ALyt_psd_max_env.setter
    def NeurNode6ALyt_psd_max_env(self,val): # pragma: no cover
        self._NeurNode6ALyt_psd_max_env=val
    #------------
    @ NeurNode6ALyt_psd_min_env.setter
    def NeurNode6ALyt_psd_min_env(self,val): # pragma: no cover
        self._NeurNode6ALyt_psd_min_env=val
    #------------
    @ NeurNode1ALzt_psd.setter
    def NeurNode1ALzt_psd(self,val): # pragma: no cover
        self._NeurNode1ALzt_psd=val
    #------------
    @ NeurNode1ALzt_psd_max_env.setter
    def NeurNode1ALzt_psd_max_env(self,val): # pragma: no cover
        self._NeurNode1ALzt_psd_max_env=val
    #------------
    @ NeurNode1ALzt_psd_min_env.setter
    def NeurNode1ALzt_psd_min_env(self,val): # pragma: no cover
        self._NeurNode1ALzt_psd_min_env=val
    #------------
    @ NeurNode2ALzt_psd.setter
    def NeurNode2ALzt_psd(self,val): # pragma: no cover
        self._NeurNode2ALzt_psd=val
    #------------
    @ NeurNode2ALzt_psd_max_env.setter
    def NeurNode2ALzt_psd_max_env(self,val): # pragma: no cover
        self._NeurNode2ALzt_psd_max_env=val
    #------------
    @ NeurNode2ALzt_psd_min_env.setter
    def NeurNode2ALzt_psd_min_env(self,val): # pragma: no cover
        self._NeurNode2ALzt_psd_min_env=val
    #------------
    @ NeurNode3ALzt_psd.setter
    def NeurNode3ALzt_psd(self,val): # pragma: no cover
        self._NeurNode3ALzt_psd=val
    #------------
    @ NeurNode3ALzt_psd_max_env.setter
    def NeurNode3ALzt_psd_max_env(self,val): # pragma: no cover
        self._NeurNode3ALzt_psd_max_env=val
    #------------
    @ NeurNode3ALzt_psd_min_env.setter
    def NeurNode3ALzt_psd_min_env(self,val): # pragma: no cover
        self._NeurNode3ALzt_psd_min_env=val
    #------------
    @ NeurNode4ALzt_psd.setter
    def NeurNode4ALzt_psd(self,val): # pragma: no cover
        self._NeurNode4ALzt_psd=val
    #------------
    @ NeurNode4ALzt_psd_max_env.setter
    def NeurNode4ALzt_psd_max_env(self,val): # pragma: no cover
        self._NeurNode4ALzt_psd_max_env=val
    #------------
    @ NeurNode4ALzt_psd_min_env.setter
    def NeurNode4ALzt_psd_min_env(self,val): # pragma: no cover
        self._NeurNode4ALzt_psd_min_env=val
    #------------
    @ NeurNode5ALzt_psd.setter
    def NeurNode5ALzt_psd(self,val): # pragma: no cover
        self._NeurNode5ALzt_psd=val
    #------------
    @ NeurNode5ALzt_psd_max_env.setter
    def NeurNode5ALzt_psd_max_env(self,val): # pragma: no cover
        self._NeurNode5ALzt_psd_max_env=val
    #------------
    @ NeurNode5ALzt_psd_min_env.setter
    def NeurNode5ALzt_psd_min_env(self,val): # pragma: no cover
        self._NeurNode5ALzt_psd_min_env=val
    #------------
    @ NeurNode6ALzt_psd.setter
    def NeurNode6ALzt_psd(self,val): # pragma: no cover
        self._NeurNode6ALzt_psd=val
    #------------
    @ NeurNode6ALzt_psd_max_env.setter
    def NeurNode6ALzt_psd_max_env(self,val): # pragma: no cover
        self._NeurNode6ALzt_psd_max_env=val
    #------------
    @ NeurNode6ALzt_psd_min_env.setter
    def NeurNode6ALzt_psd_min_env(self,val): # pragma: no cover
        self._NeurNode6ALzt_psd_min_env=val
    #------------
    @ name.setter
    def name(self,val): # pragma: no cover
        self._name=str(val)
    #------------
    @ description.setter
    def description(self,val): # pragma: no cover
        self._description=str(val)
    #------------
    #-------------------------
    # Representation functions
    #-------------------------
    def type_rep(self): # pragma: no cover
        """Generate a representation of the object type

        Returns:
            :obj:`collections.OrderedDict`: dictionnary that contains the representation of the object type
        """

        rep = collections.OrderedDict()
        rep["__type__"] = "Neuron"
        rep["name"] = self.name
        rep["description"] = self.description
        return rep
    def prop_rep(self, short = False, deep = True):

        """Generate a representation of the object properties

        Args:
            short (:obj:`bool`,optional): if True, properties are represented by their type only. If False, the values of the properties are included.
            deep (:obj:`bool`,optional): if True, the properties of each property will be included.

        Returns:
            :obj:`collections.OrderedDict`: dictionnary that contains a representation of the object properties
        """

        rep = collections.OrderedDict()
        rep["__type__"] = "Neuron"
        rep["name"] = self.name
        rep["description"] = self.description
        if self.is_set("time"):
            if (short):
                rep["time"] = str(self.time.shape)
            else:
                try:
                    rep["time"] = self.time.astype(float).round(2).tolist()
                except:
                    rep["time"] = self.time
        if self.is_set("time_psd"):
            if (short):
                rep["time_psd"] = str(self.time_psd.shape)
            else:
                try:
                    rep["time_psd"] = self.time_psd.astype(float).round(2).tolist()
                except:
                    rep["time_psd"] = self.time_psd
        if self.is_set("Frequency_psd"):
            if (short):
                rep["Frequency_psd"] = str(self.Frequency_psd.shape)
            else:
                try:
                    rep["Frequency_psd"] = self.Frequency_psd.astype(float).round(2).tolist()
                except:
                    rep["Frequency_psd"] = self.Frequency_psd
        if self.is_set("institution"):
            rep["institution"] = self.institution
        if self.is_set("source"):
            rep["source"] = self.source
        if self.is_set("status_Neuron_Sensors"):
            if (short and not(deep)):
                rep["status_Neuron_Sensors"] = self.status_Neuron_Sensors.type_rep()
            else:
                rep["status_Neuron_Sensors"] = self.status_Neuron_Sensors.prop_rep(short, deep)
        if self.is_set("NeurNode1ALxt_psd"):
            if (short and not(deep)):
                rep["NeurNode1ALxt_psd"] = self.NeurNode1ALxt_psd.type_rep()
            else:
                rep["NeurNode1ALxt_psd"] = self.NeurNode1ALxt_psd.prop_rep(short, deep)
        if self.is_set("NeurNode1ALxt_psd_max_env"):
            if (short and not(deep)):
                rep["NeurNode1ALxt_psd_max_env"] = self.NeurNode1ALxt_psd_max_env.type_rep()
            else:
                rep["NeurNode1ALxt_psd_max_env"] = self.NeurNode1ALxt_psd_max_env.prop_rep(short, deep)
        if self.is_set("NeurNode1ALxt_psd_min_env"):
            if (short and not(deep)):
                rep["NeurNode1ALxt_psd_min_env"] = self.NeurNode1ALxt_psd_min_env.type_rep()
            else:
                rep["NeurNode1ALxt_psd_min_env"] = self.NeurNode1ALxt_psd_min_env.prop_rep(short, deep)
        if self.is_set("NeurNode2ALxt_psd"):
            if (short and not(deep)):
                rep["NeurNode2ALxt_psd"] = self.NeurNode2ALxt_psd.type_rep()
            else:
                rep["NeurNode2ALxt_psd"] = self.NeurNode2ALxt_psd.prop_rep(short, deep)
        if self.is_set("NeurNode2ALxt_psd_max_env"):
            if (short and not(deep)):
                rep["NeurNode2ALxt_psd_max_env"] = self.NeurNode2ALxt_psd_max_env.type_rep()
            else:
                rep["NeurNode2ALxt_psd_max_env"] = self.NeurNode2ALxt_psd_max_env.prop_rep(short, deep)
        if self.is_set("NeurNode2ALxt_psd_min_env"):
            if (short and not(deep)):
                rep["NeurNode2ALxt_psd_min_env"] = self.NeurNode2ALxt_psd_min_env.type_rep()
            else:
                rep["NeurNode2ALxt_psd_min_env"] = self.NeurNode2ALxt_psd_min_env.prop_rep(short, deep)
        if self.is_set("NeurNode3ALxt_psd"):
            if (short and not(deep)):
                rep["NeurNode3ALxt_psd"] = self.NeurNode3ALxt_psd.type_rep()
            else:
                rep["NeurNode3ALxt_psd"] = self.NeurNode3ALxt_psd.prop_rep(short, deep)
        if self.is_set("NeurNode3ALxt_psd_max_env"):
            if (short and not(deep)):
                rep["NeurNode3ALxt_psd_max_env"] = self.NeurNode3ALxt_psd_max_env.type_rep()
            else:
                rep["NeurNode3ALxt_psd_max_env"] = self.NeurNode3ALxt_psd_max_env.prop_rep(short, deep)
        if self.is_set("NeurNode3ALxt_psd_min_env"):
            if (short and not(deep)):
                rep["NeurNode3ALxt_psd_min_env"] = self.NeurNode3ALxt_psd_min_env.type_rep()
            else:
                rep["NeurNode3ALxt_psd_min_env"] = self.NeurNode3ALxt_psd_min_env.prop_rep(short, deep)
        if self.is_set("NeurNode4ALxt_psd"):
            if (short and not(deep)):
                rep["NeurNode4ALxt_psd"] = self.NeurNode4ALxt_psd.type_rep()
            else:
                rep["NeurNode4ALxt_psd"] = self.NeurNode4ALxt_psd.prop_rep(short, deep)
        if self.is_set("NeurNode4ALxt_psd_max_env"):
            if (short and not(deep)):
                rep["NeurNode4ALxt_psd_max_env"] = self.NeurNode4ALxt_psd_max_env.type_rep()
            else:
                rep["NeurNode4ALxt_psd_max_env"] = self.NeurNode4ALxt_psd_max_env.prop_rep(short, deep)
        if self.is_set("NeurNode4ALxt_psd_min_env"):
            if (short and not(deep)):
                rep["NeurNode4ALxt_psd_min_env"] = self.NeurNode4ALxt_psd_min_env.type_rep()
            else:
                rep["NeurNode4ALxt_psd_min_env"] = self.NeurNode4ALxt_psd_min_env.prop_rep(short, deep)
        if self.is_set("NeurNode5ALxt_psd"):
            if (short and not(deep)):
                rep["NeurNode5ALxt_psd"] = self.NeurNode5ALxt_psd.type_rep()
            else:
                rep["NeurNode5ALxt_psd"] = self.NeurNode5ALxt_psd.prop_rep(short, deep)
        if self.is_set("NeurNode5ALxt_psd_max_env"):
            if (short and not(deep)):
                rep["NeurNode5ALxt_psd_max_env"] = self.NeurNode5ALxt_psd_max_env.type_rep()
            else:
                rep["NeurNode5ALxt_psd_max_env"] = self.NeurNode5ALxt_psd_max_env.prop_rep(short, deep)
        if self.is_set("NeurNode5ALxt_psd_min_env"):
            if (short and not(deep)):
                rep["NeurNode5ALxt_psd_min_env"] = self.NeurNode5ALxt_psd_min_env.type_rep()
            else:
                rep["NeurNode5ALxt_psd_min_env"] = self.NeurNode5ALxt_psd_min_env.prop_rep(short, deep)
        if self.is_set("NeurNode6ALxt_psd"):
            if (short and not(deep)):
                rep["NeurNode6ALxt_psd"] = self.NeurNode6ALxt_psd.type_rep()
            else:
                rep["NeurNode6ALxt_psd"] = self.NeurNode6ALxt_psd.prop_rep(short, deep)
        if self.is_set("NeurNode6ALxt_psd_max_env"):
            if (short and not(deep)):
                rep["NeurNode6ALxt_psd_max_env"] = self.NeurNode6ALxt_psd_max_env.type_rep()
            else:
                rep["NeurNode6ALxt_psd_max_env"] = self.NeurNode6ALxt_psd_max_env.prop_rep(short, deep)
        if self.is_set("NeurNode6ALxt_psd_min_env"):
            if (short and not(deep)):
                rep["NeurNode6ALxt_psd_min_env"] = self.NeurNode6ALxt_psd_min_env.type_rep()
            else:
                rep["NeurNode6ALxt_psd_min_env"] = self.NeurNode6ALxt_psd_min_env.prop_rep(short, deep)
        if self.is_set("NeurNode1ALyt_psd"):
            if (short and not(deep)):
                rep["NeurNode1ALyt_psd"] = self.NeurNode1ALyt_psd.type_rep()
            else:
                rep["NeurNode1ALyt_psd"] = self.NeurNode1ALyt_psd.prop_rep(short, deep)
        if self.is_set("NeurNode1ALyt_psd_max_env"):
            if (short and not(deep)):
                rep["NeurNode1ALyt_psd_max_env"] = self.NeurNode1ALyt_psd_max_env.type_rep()
            else:
                rep["NeurNode1ALyt_psd_max_env"] = self.NeurNode1ALyt_psd_max_env.prop_rep(short, deep)
        if self.is_set("NeurNode1ALyt_psd_min_env"):
            if (short and not(deep)):
                rep["NeurNode1ALyt_psd_min_env"] = self.NeurNode1ALyt_psd_min_env.type_rep()
            else:
                rep["NeurNode1ALyt_psd_min_env"] = self.NeurNode1ALyt_psd_min_env.prop_rep(short, deep)
        if self.is_set("NeurNode2ALyt_psd"):
            if (short and not(deep)):
                rep["NeurNode2ALyt_psd"] = self.NeurNode2ALyt_psd.type_rep()
            else:
                rep["NeurNode2ALyt_psd"] = self.NeurNode2ALyt_psd.prop_rep(short, deep)
        if self.is_set("NeurNode2ALyt_psd_max_env"):
            if (short and not(deep)):
                rep["NeurNode2ALyt_psd_max_env"] = self.NeurNode2ALyt_psd_max_env.type_rep()
            else:
                rep["NeurNode2ALyt_psd_max_env"] = self.NeurNode2ALyt_psd_max_env.prop_rep(short, deep)
        if self.is_set("NeurNode2ALyt_psd_min_env"):
            if (short and not(deep)):
                rep["NeurNode2ALyt_psd_min_env"] = self.NeurNode2ALyt_psd_min_env.type_rep()
            else:
                rep["NeurNode2ALyt_psd_min_env"] = self.NeurNode2ALyt_psd_min_env.prop_rep(short, deep)
        if self.is_set("NeurNode3ALyt_psd"):
            if (short and not(deep)):
                rep["NeurNode3ALyt_psd"] = self.NeurNode3ALyt_psd.type_rep()
            else:
                rep["NeurNode3ALyt_psd"] = self.NeurNode3ALyt_psd.prop_rep(short, deep)
        if self.is_set("NeurNode3ALyt_psd_max_env"):
            if (short and not(deep)):
                rep["NeurNode3ALyt_psd_max_env"] = self.NeurNode3ALyt_psd_max_env.type_rep()
            else:
                rep["NeurNode3ALyt_psd_max_env"] = self.NeurNode3ALyt_psd_max_env.prop_rep(short, deep)
        if self.is_set("NeurNode3ALyt_psd_min_env"):
            if (short and not(deep)):
                rep["NeurNode3ALyt_psd_min_env"] = self.NeurNode3ALyt_psd_min_env.type_rep()
            else:
                rep["NeurNode3ALyt_psd_min_env"] = self.NeurNode3ALyt_psd_min_env.prop_rep(short, deep)
        if self.is_set("NeurNode4ALyt_psd"):
            if (short and not(deep)):
                rep["NeurNode4ALyt_psd"] = self.NeurNode4ALyt_psd.type_rep()
            else:
                rep["NeurNode4ALyt_psd"] = self.NeurNode4ALyt_psd.prop_rep(short, deep)
        if self.is_set("NeurNode4ALyt_psd_max_env"):
            if (short and not(deep)):
                rep["NeurNode4ALyt_psd_max_env"] = self.NeurNode4ALyt_psd_max_env.type_rep()
            else:
                rep["NeurNode4ALyt_psd_max_env"] = self.NeurNode4ALyt_psd_max_env.prop_rep(short, deep)
        if self.is_set("NeurNode4ALyt_psd_min_env"):
            if (short and not(deep)):
                rep["NeurNode4ALyt_psd_min_env"] = self.NeurNode4ALyt_psd_min_env.type_rep()
            else:
                rep["NeurNode4ALyt_psd_min_env"] = self.NeurNode4ALyt_psd_min_env.prop_rep(short, deep)
        if self.is_set("NeurNode5ALyt_psd"):
            if (short and not(deep)):
                rep["NeurNode5ALyt_psd"] = self.NeurNode5ALyt_psd.type_rep()
            else:
                rep["NeurNode5ALyt_psd"] = self.NeurNode5ALyt_psd.prop_rep(short, deep)
        if self.is_set("NeurNode5ALyt_psd_max_env"):
            if (short and not(deep)):
                rep["NeurNode5ALyt_psd_max_env"] = self.NeurNode5ALyt_psd_max_env.type_rep()
            else:
                rep["NeurNode5ALyt_psd_max_env"] = self.NeurNode5ALyt_psd_max_env.prop_rep(short, deep)
        if self.is_set("NeurNode5ALyt_psd_min_env"):
            if (short and not(deep)):
                rep["NeurNode5ALyt_psd_min_env"] = self.NeurNode5ALyt_psd_min_env.type_rep()
            else:
                rep["NeurNode5ALyt_psd_min_env"] = self.NeurNode5ALyt_psd_min_env.prop_rep(short, deep)
        if self.is_set("NeurNode6ALyt_psd"):
            if (short and not(deep)):
                rep["NeurNode6ALyt_psd"] = self.NeurNode6ALyt_psd.type_rep()
            else:
                rep["NeurNode6ALyt_psd"] = self.NeurNode6ALyt_psd.prop_rep(short, deep)
        if self.is_set("NeurNode6ALyt_psd_max_env"):
            if (short and not(deep)):
                rep["NeurNode6ALyt_psd_max_env"] = self.NeurNode6ALyt_psd_max_env.type_rep()
            else:
                rep["NeurNode6ALyt_psd_max_env"] = self.NeurNode6ALyt_psd_max_env.prop_rep(short, deep)
        if self.is_set("NeurNode6ALyt_psd_min_env"):
            if (short and not(deep)):
                rep["NeurNode6ALyt_psd_min_env"] = self.NeurNode6ALyt_psd_min_env.type_rep()
            else:
                rep["NeurNode6ALyt_psd_min_env"] = self.NeurNode6ALyt_psd_min_env.prop_rep(short, deep)
        if self.is_set("NeurNode1ALzt_psd"):
            if (short and not(deep)):
                rep["NeurNode1ALzt_psd"] = self.NeurNode1ALzt_psd.type_rep()
            else:
                rep["NeurNode1ALzt_psd"] = self.NeurNode1ALzt_psd.prop_rep(short, deep)
        if self.is_set("NeurNode1ALzt_psd_max_env"):
            if (short and not(deep)):
                rep["NeurNode1ALzt_psd_max_env"] = self.NeurNode1ALzt_psd_max_env.type_rep()
            else:
                rep["NeurNode1ALzt_psd_max_env"] = self.NeurNode1ALzt_psd_max_env.prop_rep(short, deep)
        if self.is_set("NeurNode1ALzt_psd_min_env"):
            if (short and not(deep)):
                rep["NeurNode1ALzt_psd_min_env"] = self.NeurNode1ALzt_psd_min_env.type_rep()
            else:
                rep["NeurNode1ALzt_psd_min_env"] = self.NeurNode1ALzt_psd_min_env.prop_rep(short, deep)
        if self.is_set("NeurNode2ALzt_psd"):
            if (short and not(deep)):
                rep["NeurNode2ALzt_psd"] = self.NeurNode2ALzt_psd.type_rep()
            else:
                rep["NeurNode2ALzt_psd"] = self.NeurNode2ALzt_psd.prop_rep(short, deep)
        if self.is_set("NeurNode2ALzt_psd_max_env"):
            if (short and not(deep)):
                rep["NeurNode2ALzt_psd_max_env"] = self.NeurNode2ALzt_psd_max_env.type_rep()
            else:
                rep["NeurNode2ALzt_psd_max_env"] = self.NeurNode2ALzt_psd_max_env.prop_rep(short, deep)
        if self.is_set("NeurNode2ALzt_psd_min_env"):
            if (short and not(deep)):
                rep["NeurNode2ALzt_psd_min_env"] = self.NeurNode2ALzt_psd_min_env.type_rep()
            else:
                rep["NeurNode2ALzt_psd_min_env"] = self.NeurNode2ALzt_psd_min_env.prop_rep(short, deep)
        if self.is_set("NeurNode3ALzt_psd"):
            if (short and not(deep)):
                rep["NeurNode3ALzt_psd"] = self.NeurNode3ALzt_psd.type_rep()
            else:
                rep["NeurNode3ALzt_psd"] = self.NeurNode3ALzt_psd.prop_rep(short, deep)
        if self.is_set("NeurNode3ALzt_psd_max_env"):
            if (short and not(deep)):
                rep["NeurNode3ALzt_psd_max_env"] = self.NeurNode3ALzt_psd_max_env.type_rep()
            else:
                rep["NeurNode3ALzt_psd_max_env"] = self.NeurNode3ALzt_psd_max_env.prop_rep(short, deep)
        if self.is_set("NeurNode3ALzt_psd_min_env"):
            if (short and not(deep)):
                rep["NeurNode3ALzt_psd_min_env"] = self.NeurNode3ALzt_psd_min_env.type_rep()
            else:
                rep["NeurNode3ALzt_psd_min_env"] = self.NeurNode3ALzt_psd_min_env.prop_rep(short, deep)
        if self.is_set("NeurNode4ALzt_psd"):
            if (short and not(deep)):
                rep["NeurNode4ALzt_psd"] = self.NeurNode4ALzt_psd.type_rep()
            else:
                rep["NeurNode4ALzt_psd"] = self.NeurNode4ALzt_psd.prop_rep(short, deep)
        if self.is_set("NeurNode4ALzt_psd_max_env"):
            if (short and not(deep)):
                rep["NeurNode4ALzt_psd_max_env"] = self.NeurNode4ALzt_psd_max_env.type_rep()
            else:
                rep["NeurNode4ALzt_psd_max_env"] = self.NeurNode4ALzt_psd_max_env.prop_rep(short, deep)
        if self.is_set("NeurNode4ALzt_psd_min_env"):
            if (short and not(deep)):
                rep["NeurNode4ALzt_psd_min_env"] = self.NeurNode4ALzt_psd_min_env.type_rep()
            else:
                rep["NeurNode4ALzt_psd_min_env"] = self.NeurNode4ALzt_psd_min_env.prop_rep(short, deep)
        if self.is_set("NeurNode5ALzt_psd"):
            if (short and not(deep)):
                rep["NeurNode5ALzt_psd"] = self.NeurNode5ALzt_psd.type_rep()
            else:
                rep["NeurNode5ALzt_psd"] = self.NeurNode5ALzt_psd.prop_rep(short, deep)
        if self.is_set("NeurNode5ALzt_psd_max_env"):
            if (short and not(deep)):
                rep["NeurNode5ALzt_psd_max_env"] = self.NeurNode5ALzt_psd_max_env.type_rep()
            else:
                rep["NeurNode5ALzt_psd_max_env"] = self.NeurNode5ALzt_psd_max_env.prop_rep(short, deep)
        if self.is_set("NeurNode5ALzt_psd_min_env"):
            if (short and not(deep)):
                rep["NeurNode5ALzt_psd_min_env"] = self.NeurNode5ALzt_psd_min_env.type_rep()
            else:
                rep["NeurNode5ALzt_psd_min_env"] = self.NeurNode5ALzt_psd_min_env.prop_rep(short, deep)
        if self.is_set("NeurNode6ALzt_psd"):
            if (short and not(deep)):
                rep["NeurNode6ALzt_psd"] = self.NeurNode6ALzt_psd.type_rep()
            else:
                rep["NeurNode6ALzt_psd"] = self.NeurNode6ALzt_psd.prop_rep(short, deep)
        if self.is_set("NeurNode6ALzt_psd_max_env"):
            if (short and not(deep)):
                rep["NeurNode6ALzt_psd_max_env"] = self.NeurNode6ALzt_psd_max_env.type_rep()
            else:
                rep["NeurNode6ALzt_psd_max_env"] = self.NeurNode6ALzt_psd_max_env.prop_rep(short, deep)
        if self.is_set("NeurNode6ALzt_psd_min_env"):
            if (short and not(deep)):
                rep["NeurNode6ALzt_psd_min_env"] = self.NeurNode6ALzt_psd_min_env.type_rep()
            else:
                rep["NeurNode6ALzt_psd_min_env"] = self.NeurNode6ALzt_psd_min_env.prop_rep(short, deep)
        if self.is_set("name"):
            rep["name"] = self.name
        if self.is_set("description"):
            rep["description"] = self.description
        return rep
    #-------------------------
    # Save functions
    #-------------------------
    def json_rep(self, short=False, deep=True):

        """Generate a JSON representation of the object properties

        Args:
            short (:obj:`bool`,optional): if True, properties are represented by their type only. If False, the values of the properties are included.
            deep (:obj:`bool`,optional): if True, the properties of each property will be included.

        Returns:
            :obj:`str`: string that contains a JSON representation of the object properties
        """

        return ( json.dumps(self.prop_rep(short=short, deep=deep),indent=4, separators=(",",": ")))
    def saveJSON(self, fileName=None):
        """Save the instance of the object to JSON format file

        Args:
            fileName (:obj:`str`, optional): Name of the JSON file, included extension. Defaults is None. If None, the name of the JSON file will be self.name.json. It can also contain an absolute or relative path.

        """

        if fileName==None:
            fileName=self.name + ".json"
        f=open(fileName, "w")
        f.write(self.json_rep())
        f.write("\n")
        f.close()

    def saveHDF5(self,filePath=None):
        """Save the instance of the object to HDF5 format file
        Args:
            filePath (:obj:`str`, optional): Name of the HDF5 file, included extension. Defaults is None. If None, the name of the HDF5 file will be "self.name".h5. It can also contain an absolute or relative path.
        """
        # Define filepath
        if (filePath == None):
            if hasattr(self, "name"):
                filePath = self.name + ".h5"
            else:
                raise Exception("name is required for saving")
        # Open hdf5 file
        h = h5py.File(filePath,"w")
        group = h.create_group(self.name)
        # Save the object to the group
        self.saveToHDF5Handle(group)
        # Close file
        h.close()
        pass
    def saveToHDF5Handle(self, handle):
        """Save the properties of the object to the hdf5 handle.
        Args:
            handle (:obj:`h5py.Group`): Handle used to store the object properties
        """
        if self.is_set("time") :
            handle["time"] = np.array(self.time,dtype=float)
        if self.is_set("time_psd") :
            handle["time_psd"] = np.array(self.time_psd,dtype=float)
        if self.is_set("Frequency_psd") :
            handle["Frequency_psd"] = np.array(self.Frequency_psd,dtype=float)
        if self.is_set("institution") :
            ar = []
            ar.append(self.institution.encode("ascii"))
            handle["institution"] = np.asarray(ar)
        if self.is_set("source") :
            ar = []
            ar.append(self.source.encode("ascii"))
            handle["source"] = np.asarray(ar)
        if self.is_set("status_Neuron_Sensors") :
            subgroup = handle.create_group("status_Neuron_Sensors")
            self.status_Neuron_Sensors.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode1ALxt_psd") :
            subgroup = handle.create_group("NeurNode1ALxt_psd")
            self.NeurNode1ALxt_psd.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode1ALxt_psd_max_env") :
            subgroup = handle.create_group("NeurNode1ALxt_psd_max_env")
            self.NeurNode1ALxt_psd_max_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode1ALxt_psd_min_env") :
            subgroup = handle.create_group("NeurNode1ALxt_psd_min_env")
            self.NeurNode1ALxt_psd_min_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode2ALxt_psd") :
            subgroup = handle.create_group("NeurNode2ALxt_psd")
            self.NeurNode2ALxt_psd.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode2ALxt_psd_max_env") :
            subgroup = handle.create_group("NeurNode2ALxt_psd_max_env")
            self.NeurNode2ALxt_psd_max_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode2ALxt_psd_min_env") :
            subgroup = handle.create_group("NeurNode2ALxt_psd_min_env")
            self.NeurNode2ALxt_psd_min_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode3ALxt_psd") :
            subgroup = handle.create_group("NeurNode3ALxt_psd")
            self.NeurNode3ALxt_psd.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode3ALxt_psd_max_env") :
            subgroup = handle.create_group("NeurNode3ALxt_psd_max_env")
            self.NeurNode3ALxt_psd_max_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode3ALxt_psd_min_env") :
            subgroup = handle.create_group("NeurNode3ALxt_psd_min_env")
            self.NeurNode3ALxt_psd_min_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode4ALxt_psd") :
            subgroup = handle.create_group("NeurNode4ALxt_psd")
            self.NeurNode4ALxt_psd.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode4ALxt_psd_max_env") :
            subgroup = handle.create_group("NeurNode4ALxt_psd_max_env")
            self.NeurNode4ALxt_psd_max_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode4ALxt_psd_min_env") :
            subgroup = handle.create_group("NeurNode4ALxt_psd_min_env")
            self.NeurNode4ALxt_psd_min_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode5ALxt_psd") :
            subgroup = handle.create_group("NeurNode5ALxt_psd")
            self.NeurNode5ALxt_psd.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode5ALxt_psd_max_env") :
            subgroup = handle.create_group("NeurNode5ALxt_psd_max_env")
            self.NeurNode5ALxt_psd_max_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode5ALxt_psd_min_env") :
            subgroup = handle.create_group("NeurNode5ALxt_psd_min_env")
            self.NeurNode5ALxt_psd_min_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode6ALxt_psd") :
            subgroup = handle.create_group("NeurNode6ALxt_psd")
            self.NeurNode6ALxt_psd.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode6ALxt_psd_max_env") :
            subgroup = handle.create_group("NeurNode6ALxt_psd_max_env")
            self.NeurNode6ALxt_psd_max_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode6ALxt_psd_min_env") :
            subgroup = handle.create_group("NeurNode6ALxt_psd_min_env")
            self.NeurNode6ALxt_psd_min_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode1ALyt_psd") :
            subgroup = handle.create_group("NeurNode1ALyt_psd")
            self.NeurNode1ALyt_psd.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode1ALyt_psd_max_env") :
            subgroup = handle.create_group("NeurNode1ALyt_psd_max_env")
            self.NeurNode1ALyt_psd_max_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode1ALyt_psd_min_env") :
            subgroup = handle.create_group("NeurNode1ALyt_psd_min_env")
            self.NeurNode1ALyt_psd_min_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode2ALyt_psd") :
            subgroup = handle.create_group("NeurNode2ALyt_psd")
            self.NeurNode2ALyt_psd.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode2ALyt_psd_max_env") :
            subgroup = handle.create_group("NeurNode2ALyt_psd_max_env")
            self.NeurNode2ALyt_psd_max_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode2ALyt_psd_min_env") :
            subgroup = handle.create_group("NeurNode2ALyt_psd_min_env")
            self.NeurNode2ALyt_psd_min_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode3ALyt_psd") :
            subgroup = handle.create_group("NeurNode3ALyt_psd")
            self.NeurNode3ALyt_psd.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode3ALyt_psd_max_env") :
            subgroup = handle.create_group("NeurNode3ALyt_psd_max_env")
            self.NeurNode3ALyt_psd_max_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode3ALyt_psd_min_env") :
            subgroup = handle.create_group("NeurNode3ALyt_psd_min_env")
            self.NeurNode3ALyt_psd_min_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode4ALyt_psd") :
            subgroup = handle.create_group("NeurNode4ALyt_psd")
            self.NeurNode4ALyt_psd.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode4ALyt_psd_max_env") :
            subgroup = handle.create_group("NeurNode4ALyt_psd_max_env")
            self.NeurNode4ALyt_psd_max_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode4ALyt_psd_min_env") :
            subgroup = handle.create_group("NeurNode4ALyt_psd_min_env")
            self.NeurNode4ALyt_psd_min_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode5ALyt_psd") :
            subgroup = handle.create_group("NeurNode5ALyt_psd")
            self.NeurNode5ALyt_psd.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode5ALyt_psd_max_env") :
            subgroup = handle.create_group("NeurNode5ALyt_psd_max_env")
            self.NeurNode5ALyt_psd_max_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode5ALyt_psd_min_env") :
            subgroup = handle.create_group("NeurNode5ALyt_psd_min_env")
            self.NeurNode5ALyt_psd_min_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode6ALyt_psd") :
            subgroup = handle.create_group("NeurNode6ALyt_psd")
            self.NeurNode6ALyt_psd.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode6ALyt_psd_max_env") :
            subgroup = handle.create_group("NeurNode6ALyt_psd_max_env")
            self.NeurNode6ALyt_psd_max_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode6ALyt_psd_min_env") :
            subgroup = handle.create_group("NeurNode6ALyt_psd_min_env")
            self.NeurNode6ALyt_psd_min_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode1ALzt_psd") :
            subgroup = handle.create_group("NeurNode1ALzt_psd")
            self.NeurNode1ALzt_psd.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode1ALzt_psd_max_env") :
            subgroup = handle.create_group("NeurNode1ALzt_psd_max_env")
            self.NeurNode1ALzt_psd_max_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode1ALzt_psd_min_env") :
            subgroup = handle.create_group("NeurNode1ALzt_psd_min_env")
            self.NeurNode1ALzt_psd_min_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode2ALzt_psd") :
            subgroup = handle.create_group("NeurNode2ALzt_psd")
            self.NeurNode2ALzt_psd.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode2ALzt_psd_max_env") :
            subgroup = handle.create_group("NeurNode2ALzt_psd_max_env")
            self.NeurNode2ALzt_psd_max_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode2ALzt_psd_min_env") :
            subgroup = handle.create_group("NeurNode2ALzt_psd_min_env")
            self.NeurNode2ALzt_psd_min_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode3ALzt_psd") :
            subgroup = handle.create_group("NeurNode3ALzt_psd")
            self.NeurNode3ALzt_psd.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode3ALzt_psd_max_env") :
            subgroup = handle.create_group("NeurNode3ALzt_psd_max_env")
            self.NeurNode3ALzt_psd_max_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode3ALzt_psd_min_env") :
            subgroup = handle.create_group("NeurNode3ALzt_psd_min_env")
            self.NeurNode3ALzt_psd_min_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode4ALzt_psd") :
            subgroup = handle.create_group("NeurNode4ALzt_psd")
            self.NeurNode4ALzt_psd.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode4ALzt_psd_max_env") :
            subgroup = handle.create_group("NeurNode4ALzt_psd_max_env")
            self.NeurNode4ALzt_psd_max_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode4ALzt_psd_min_env") :
            subgroup = handle.create_group("NeurNode4ALzt_psd_min_env")
            self.NeurNode4ALzt_psd_min_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode5ALzt_psd") :
            subgroup = handle.create_group("NeurNode5ALzt_psd")
            self.NeurNode5ALzt_psd.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode5ALzt_psd_max_env") :
            subgroup = handle.create_group("NeurNode5ALzt_psd_max_env")
            self.NeurNode5ALzt_psd_max_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode5ALzt_psd_min_env") :
            subgroup = handle.create_group("NeurNode5ALzt_psd_min_env")
            self.NeurNode5ALzt_psd_min_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode6ALzt_psd") :
            subgroup = handle.create_group("NeurNode6ALzt_psd")
            self.NeurNode6ALzt_psd.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode6ALzt_psd_max_env") :
            subgroup = handle.create_group("NeurNode6ALzt_psd_max_env")
            self.NeurNode6ALzt_psd_max_env.saveToHDF5Handle(subgroup)
        if self.is_set("NeurNode6ALzt_psd_min_env") :
            subgroup = handle.create_group("NeurNode6ALzt_psd_min_env")
            self.NeurNode6ALzt_psd_min_env.saveToHDF5Handle(subgroup)
        if self.is_set("name") :
            ar = []
            ar.append(self.name.encode("ascii"))
            handle["name"] = np.asarray(ar)
        if self.is_set("description") :
            ar = []
            ar.append(self.description.encode("ascii"))
            handle["description"] = np.asarray(ar)
    def create_xr_Dataset(self, time_format = "%Y-%m-%d %H:%M:%S", user_dims_dict={'status_Neuron_Sensors': 'time', 'NeurNode1ALxt_psd': ['time_psd', 'Frequency_psd'], 'NeurNode1ALxt_psd_max_env': ['time_psd', 'Frequency_psd'], 'NeurNode1ALxt_psd_min_env': ['time_psd', 'Frequency_psd'], 'NeurNode2ALxt_psd': ['time_psd', 'Frequency_psd'], 'NeurNode2ALxt_psd_max_env': ['time_psd', 'Frequency_psd'], 'NeurNode2ALxt_psd_min_env': ['time_psd', 'Frequency_psd'], 'NeurNode3ALxt_psd': ['time_psd', 'Frequency_psd'], 'NeurNode3ALxt_psd_max_env': ['time_psd', 'Frequency_psd'], 'NeurNode3ALxt_psd_min_env': ['time_psd', 'Frequency_psd'], 'NeurNode4ALxt_psd': ['time_psd', 'Frequency_psd'], 'NeurNode4ALxt_psd_max_env': ['time_psd', 'Frequency_psd'], 'NeurNode4ALxt_psd_min_env': ['time_psd', 'Frequency_psd'], 'NeurNode5ALxt_psd': ['time_psd', 'Frequency_psd'], 'NeurNode5ALxt_psd_max_env': ['time_psd', 'Frequency_psd'], 'NeurNode5ALxt_psd_min_env': ['time_psd', 'Frequency_psd'], 'NeurNode6ALxt_psd': ['time_psd', 'Frequency_psd'], 'NeurNode6ALxt_psd_max_env': ['time_psd', 'Frequency_psd'], 'NeurNode6ALxt_psd_min_env': ['time_psd', 'Frequency_psd'], 'NeurNode1ALyt_psd': ['time_psd', 'Frequency_psd'], 'NeurNode1ALyt_psd_max_env': ['time_psd', 'Frequency_psd'], 'NeurNode1ALyt_psd_min_env': ['time_psd', 'Frequency_psd'], 'NeurNode2ALyt_psd': ['time_psd', 'Frequency_psd'], 'NeurNode2ALyt_psd_max_env': ['time_psd', 'Frequency_psd'], 'NeurNode2ALyt_psd_min_env': ['time_psd', 'Frequency_psd'], 'NeurNode3ALyt_psd': ['time_psd', 'Frequency_psd'], 'NeurNode3ALyt_psd_max_env': ['time_psd', 'Frequency_psd'], 'NeurNode3ALyt_psd_min_env': ['time_psd', 'Frequency_psd'], 'NeurNode4ALyt_psd': ['time_psd', 'Frequency_psd'], 'NeurNode4ALyt_psd_max_env': ['time_psd', 'Frequency_psd'], 'NeurNode4ALyt_psd_min_env': ['time_psd', 'Frequency_psd'], 'NeurNode5ALyt_psd': ['time_psd', 'Frequency_psd'], 'NeurNode5ALyt_psd_max_env': ['time_psd', 'Frequency_psd'], 'NeurNode5ALyt_psd_min_env': ['time_psd', 'Frequency_psd'], 'NeurNode6ALyt_psd': ['time_psd', 'Frequency_psd'], 'NeurNode6ALyt_psd_max_env': ['time_psd', 'Frequency_psd'], 'NeurNode6ALyt_psd_min_env': ['time_psd', 'Frequency_psd'], 'NeurNode1ALzt_psd': ['time_psd', 'Frequency_psd'], 'NeurNode1ALzt_psd_max_env': ['time_psd', 'Frequency_psd'], 'NeurNode1ALzt_psd_min_env': ['time_psd', 'Frequency_psd'], 'NeurNode2ALzt_psd': ['time_psd', 'Frequency_psd'], 'NeurNode2ALzt_psd_max_env': ['time_psd', 'Frequency_psd'], 'NeurNode2ALzt_psd_min_env': ['time_psd', 'Frequency_psd'], 'NeurNode3ALzt_psd': ['time_psd', 'Frequency_psd'], 'NeurNode3ALzt_psd_max_env': ['time_psd', 'Frequency_psd'], 'NeurNode3ALzt_psd_min_env': ['time_psd', 'Frequency_psd'], 'NeurNode4ALzt_psd': ['time_psd', 'Frequency_psd'], 'NeurNode4ALzt_psd_max_env': ['time_psd', 'Frequency_psd'], 'NeurNode4ALzt_psd_min_env': ['time_psd', 'Frequency_psd'], 'NeurNode5ALzt_psd': ['time_psd', 'Frequency_psd'], 'NeurNode5ALzt_psd_max_env': ['time_psd', 'Frequency_psd'], 'NeurNode5ALzt_psd_min_env': ['time_psd', 'Frequency_psd'], 'NeurNode6ALzt_psd': ['time_psd', 'Frequency_psd'], 'NeurNode6ALzt_psd_max_env': ['time_psd', 'Frequency_psd'], 'NeurNode6ALzt_psd_min_env': ['time_psd', 'Frequency_psd']}):
        xr_ds_l=[]
        xr_ds=[]
        import pandas
        time= pandas.to_datetime(self.time, unit = "s", origin="unix")
        if self.is_set("status_Neuron_Sensors") :
            if len(self.time)==len(self.status_Neuron_Sensors.values) and not any(pandas.isnull(self.time)):
                xr_ds5 = xr.Dataset(
                    {self.status_Neuron_Sensors.short_name : self.status_Neuron_Sensors.create_xr_DataArray(user_dims=user_dims_dict["status_Neuron_Sensors"])},
                    coords={"time" :  time},
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds5)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode1ALxt_psd") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode1ALxt_psd.values) :
                xr_ds6 = xr.Dataset(
                    {self.NeurNode1ALxt_psd.short_name : self.NeurNode1ALxt_psd.create_xr_DataArray(user_dims=user_dims_dict["NeurNode1ALxt_psd"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds6)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode1ALxt_psd_max_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode1ALxt_psd_max_env.values) :
                xr_ds7 = xr.Dataset(
                    {self.NeurNode1ALxt_psd_max_env.short_name : self.NeurNode1ALxt_psd_max_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode1ALxt_psd_max_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds7)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode1ALxt_psd_min_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode1ALxt_psd_min_env.values) :
                xr_ds8 = xr.Dataset(
                    {self.NeurNode1ALxt_psd_min_env.short_name : self.NeurNode1ALxt_psd_min_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode1ALxt_psd_min_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds8)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode2ALxt_psd") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode2ALxt_psd.values) :
                xr_ds9 = xr.Dataset(
                    {self.NeurNode2ALxt_psd.short_name : self.NeurNode2ALxt_psd.create_xr_DataArray(user_dims=user_dims_dict["NeurNode2ALxt_psd"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds9)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode2ALxt_psd_max_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode2ALxt_psd_max_env.values) :
                xr_ds10 = xr.Dataset(
                    {self.NeurNode2ALxt_psd_max_env.short_name : self.NeurNode2ALxt_psd_max_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode2ALxt_psd_max_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds10)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode2ALxt_psd_min_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode2ALxt_psd_min_env.values) :
                xr_ds11 = xr.Dataset(
                    {self.NeurNode2ALxt_psd_min_env.short_name : self.NeurNode2ALxt_psd_min_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode2ALxt_psd_min_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds11)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode3ALxt_psd") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode3ALxt_psd.values) :
                xr_ds12 = xr.Dataset(
                    {self.NeurNode3ALxt_psd.short_name : self.NeurNode3ALxt_psd.create_xr_DataArray(user_dims=user_dims_dict["NeurNode3ALxt_psd"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds12)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode3ALxt_psd_max_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode3ALxt_psd_max_env.values) :
                xr_ds13 = xr.Dataset(
                    {self.NeurNode3ALxt_psd_max_env.short_name : self.NeurNode3ALxt_psd_max_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode3ALxt_psd_max_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds13)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode3ALxt_psd_min_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode3ALxt_psd_min_env.values) :
                xr_ds14 = xr.Dataset(
                    {self.NeurNode3ALxt_psd_min_env.short_name : self.NeurNode3ALxt_psd_min_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode3ALxt_psd_min_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds14)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode4ALxt_psd") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode4ALxt_psd.values) :
                xr_ds15 = xr.Dataset(
                    {self.NeurNode4ALxt_psd.short_name : self.NeurNode4ALxt_psd.create_xr_DataArray(user_dims=user_dims_dict["NeurNode4ALxt_psd"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds15)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode4ALxt_psd_max_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode4ALxt_psd_max_env.values) :
                xr_ds16 = xr.Dataset(
                    {self.NeurNode4ALxt_psd_max_env.short_name : self.NeurNode4ALxt_psd_max_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode4ALxt_psd_max_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds16)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode4ALxt_psd_min_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode4ALxt_psd_min_env.values) :
                xr_ds17 = xr.Dataset(
                    {self.NeurNode4ALxt_psd_min_env.short_name : self.NeurNode4ALxt_psd_min_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode4ALxt_psd_min_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds17)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode5ALxt_psd") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode5ALxt_psd.values) :
                xr_ds18 = xr.Dataset(
                    {self.NeurNode5ALxt_psd.short_name : self.NeurNode5ALxt_psd.create_xr_DataArray(user_dims=user_dims_dict["NeurNode5ALxt_psd"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds18)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode5ALxt_psd_max_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode5ALxt_psd_max_env.values) :
                xr_ds19 = xr.Dataset(
                    {self.NeurNode5ALxt_psd_max_env.short_name : self.NeurNode5ALxt_psd_max_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode5ALxt_psd_max_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds19)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode5ALxt_psd_min_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode5ALxt_psd_min_env.values) :
                xr_ds20 = xr.Dataset(
                    {self.NeurNode5ALxt_psd_min_env.short_name : self.NeurNode5ALxt_psd_min_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode5ALxt_psd_min_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds20)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode6ALxt_psd") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode6ALxt_psd.values) :
                xr_ds21 = xr.Dataset(
                    {self.NeurNode6ALxt_psd.short_name : self.NeurNode6ALxt_psd.create_xr_DataArray(user_dims=user_dims_dict["NeurNode6ALxt_psd"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds21)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode6ALxt_psd_max_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode6ALxt_psd_max_env.values) :
                xr_ds22 = xr.Dataset(
                    {self.NeurNode6ALxt_psd_max_env.short_name : self.NeurNode6ALxt_psd_max_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode6ALxt_psd_max_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds22)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode6ALxt_psd_min_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode6ALxt_psd_min_env.values) :
                xr_ds23 = xr.Dataset(
                    {self.NeurNode6ALxt_psd_min_env.short_name : self.NeurNode6ALxt_psd_min_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode6ALxt_psd_min_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds23)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode1ALyt_psd") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode1ALyt_psd.values) :
                xr_ds24 = xr.Dataset(
                    {self.NeurNode1ALyt_psd.short_name : self.NeurNode1ALyt_psd.create_xr_DataArray(user_dims=user_dims_dict["NeurNode1ALyt_psd"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds24)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode1ALyt_psd_max_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode1ALyt_psd_max_env.values) :
                xr_ds25 = xr.Dataset(
                    {self.NeurNode1ALyt_psd_max_env.short_name : self.NeurNode1ALyt_psd_max_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode1ALyt_psd_max_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds25)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode1ALyt_psd_min_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode1ALyt_psd_min_env.values) :
                xr_ds26 = xr.Dataset(
                    {self.NeurNode1ALyt_psd_min_env.short_name : self.NeurNode1ALyt_psd_min_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode1ALyt_psd_min_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds26)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode2ALyt_psd") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode2ALyt_psd.values) :
                xr_ds27 = xr.Dataset(
                    {self.NeurNode2ALyt_psd.short_name : self.NeurNode2ALyt_psd.create_xr_DataArray(user_dims=user_dims_dict["NeurNode2ALyt_psd"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds27)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode2ALyt_psd_max_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode2ALyt_psd_max_env.values) :
                xr_ds28 = xr.Dataset(
                    {self.NeurNode2ALyt_psd_max_env.short_name : self.NeurNode2ALyt_psd_max_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode2ALyt_psd_max_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds28)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode2ALyt_psd_min_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode2ALyt_psd_min_env.values) :
                xr_ds29 = xr.Dataset(
                    {self.NeurNode2ALyt_psd_min_env.short_name : self.NeurNode2ALyt_psd_min_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode2ALyt_psd_min_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds29)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode3ALyt_psd") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode3ALyt_psd.values) :
                xr_ds30 = xr.Dataset(
                    {self.NeurNode3ALyt_psd.short_name : self.NeurNode3ALyt_psd.create_xr_DataArray(user_dims=user_dims_dict["NeurNode3ALyt_psd"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds30)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode3ALyt_psd_max_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode3ALyt_psd_max_env.values) :
                xr_ds31 = xr.Dataset(
                    {self.NeurNode3ALyt_psd_max_env.short_name : self.NeurNode3ALyt_psd_max_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode3ALyt_psd_max_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds31)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode3ALyt_psd_min_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode3ALyt_psd_min_env.values) :
                xr_ds32 = xr.Dataset(
                    {self.NeurNode3ALyt_psd_min_env.short_name : self.NeurNode3ALyt_psd_min_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode3ALyt_psd_min_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds32)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode4ALyt_psd") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode4ALyt_psd.values) :
                xr_ds33 = xr.Dataset(
                    {self.NeurNode4ALyt_psd.short_name : self.NeurNode4ALyt_psd.create_xr_DataArray(user_dims=user_dims_dict["NeurNode4ALyt_psd"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds33)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode4ALyt_psd_max_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode4ALyt_psd_max_env.values) :
                xr_ds34 = xr.Dataset(
                    {self.NeurNode4ALyt_psd_max_env.short_name : self.NeurNode4ALyt_psd_max_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode4ALyt_psd_max_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds34)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode4ALyt_psd_min_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode4ALyt_psd_min_env.values) :
                xr_ds35 = xr.Dataset(
                    {self.NeurNode4ALyt_psd_min_env.short_name : self.NeurNode4ALyt_psd_min_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode4ALyt_psd_min_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds35)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode5ALyt_psd") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode5ALyt_psd.values) :
                xr_ds36 = xr.Dataset(
                    {self.NeurNode5ALyt_psd.short_name : self.NeurNode5ALyt_psd.create_xr_DataArray(user_dims=user_dims_dict["NeurNode5ALyt_psd"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds36)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode5ALyt_psd_max_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode5ALyt_psd_max_env.values) :
                xr_ds37 = xr.Dataset(
                    {self.NeurNode5ALyt_psd_max_env.short_name : self.NeurNode5ALyt_psd_max_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode5ALyt_psd_max_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds37)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode5ALyt_psd_min_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode5ALyt_psd_min_env.values) :
                xr_ds38 = xr.Dataset(
                    {self.NeurNode5ALyt_psd_min_env.short_name : self.NeurNode5ALyt_psd_min_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode5ALyt_psd_min_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds38)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode6ALyt_psd") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode6ALyt_psd.values) :
                xr_ds39 = xr.Dataset(
                    {self.NeurNode6ALyt_psd.short_name : self.NeurNode6ALyt_psd.create_xr_DataArray(user_dims=user_dims_dict["NeurNode6ALyt_psd"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds39)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode6ALyt_psd_max_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode6ALyt_psd_max_env.values) :
                xr_ds40 = xr.Dataset(
                    {self.NeurNode6ALyt_psd_max_env.short_name : self.NeurNode6ALyt_psd_max_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode6ALyt_psd_max_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds40)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode6ALyt_psd_min_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode6ALyt_psd_min_env.values) :
                xr_ds41 = xr.Dataset(
                    {self.NeurNode6ALyt_psd_min_env.short_name : self.NeurNode6ALyt_psd_min_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode6ALyt_psd_min_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds41)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode1ALzt_psd") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode1ALzt_psd.values) :
                xr_ds42 = xr.Dataset(
                    {self.NeurNode1ALzt_psd.short_name : self.NeurNode1ALzt_psd.create_xr_DataArray(user_dims=user_dims_dict["NeurNode1ALzt_psd"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds42)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode1ALzt_psd_max_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode1ALzt_psd_max_env.values) :
                xr_ds43 = xr.Dataset(
                    {self.NeurNode1ALzt_psd_max_env.short_name : self.NeurNode1ALzt_psd_max_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode1ALzt_psd_max_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds43)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode1ALzt_psd_min_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode1ALzt_psd_min_env.values) :
                xr_ds44 = xr.Dataset(
                    {self.NeurNode1ALzt_psd_min_env.short_name : self.NeurNode1ALzt_psd_min_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode1ALzt_psd_min_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds44)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode2ALzt_psd") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode2ALzt_psd.values) :
                xr_ds45 = xr.Dataset(
                    {self.NeurNode2ALzt_psd.short_name : self.NeurNode2ALzt_psd.create_xr_DataArray(user_dims=user_dims_dict["NeurNode2ALzt_psd"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds45)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode2ALzt_psd_max_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode2ALzt_psd_max_env.values) :
                xr_ds46 = xr.Dataset(
                    {self.NeurNode2ALzt_psd_max_env.short_name : self.NeurNode2ALzt_psd_max_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode2ALzt_psd_max_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds46)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode2ALzt_psd_min_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode2ALzt_psd_min_env.values) :
                xr_ds47 = xr.Dataset(
                    {self.NeurNode2ALzt_psd_min_env.short_name : self.NeurNode2ALzt_psd_min_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode2ALzt_psd_min_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds47)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode3ALzt_psd") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode3ALzt_psd.values) :
                xr_ds48 = xr.Dataset(
                    {self.NeurNode3ALzt_psd.short_name : self.NeurNode3ALzt_psd.create_xr_DataArray(user_dims=user_dims_dict["NeurNode3ALzt_psd"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds48)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode3ALzt_psd_max_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode3ALzt_psd_max_env.values) :
                xr_ds49 = xr.Dataset(
                    {self.NeurNode3ALzt_psd_max_env.short_name : self.NeurNode3ALzt_psd_max_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode3ALzt_psd_max_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds49)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode3ALzt_psd_min_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode3ALzt_psd_min_env.values) :
                xr_ds50 = xr.Dataset(
                    {self.NeurNode3ALzt_psd_min_env.short_name : self.NeurNode3ALzt_psd_min_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode3ALzt_psd_min_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds50)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode4ALzt_psd") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode4ALzt_psd.values) :
                xr_ds51 = xr.Dataset(
                    {self.NeurNode4ALzt_psd.short_name : self.NeurNode4ALzt_psd.create_xr_DataArray(user_dims=user_dims_dict["NeurNode4ALzt_psd"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds51)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode4ALzt_psd_max_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode4ALzt_psd_max_env.values) :
                xr_ds52 = xr.Dataset(
                    {self.NeurNode4ALzt_psd_max_env.short_name : self.NeurNode4ALzt_psd_max_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode4ALzt_psd_max_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds52)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode4ALzt_psd_min_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode4ALzt_psd_min_env.values) :
                xr_ds53 = xr.Dataset(
                    {self.NeurNode4ALzt_psd_min_env.short_name : self.NeurNode4ALzt_psd_min_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode4ALzt_psd_min_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds53)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode5ALzt_psd") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode5ALzt_psd.values) :
                xr_ds54 = xr.Dataset(
                    {self.NeurNode5ALzt_psd.short_name : self.NeurNode5ALzt_psd.create_xr_DataArray(user_dims=user_dims_dict["NeurNode5ALzt_psd"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds54)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode5ALzt_psd_max_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode5ALzt_psd_max_env.values) :
                xr_ds55 = xr.Dataset(
                    {self.NeurNode5ALzt_psd_max_env.short_name : self.NeurNode5ALzt_psd_max_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode5ALzt_psd_max_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds55)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode5ALzt_psd_min_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode5ALzt_psd_min_env.values) :
                xr_ds56 = xr.Dataset(
                    {self.NeurNode5ALzt_psd_min_env.short_name : self.NeurNode5ALzt_psd_min_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode5ALzt_psd_min_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds56)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode6ALzt_psd") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode6ALzt_psd.values) :
                xr_ds57 = xr.Dataset(
                    {self.NeurNode6ALzt_psd.short_name : self.NeurNode6ALzt_psd.create_xr_DataArray(user_dims=user_dims_dict["NeurNode6ALzt_psd"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds57)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode6ALzt_psd_max_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode6ALzt_psd_max_env.values) :
                xr_ds58 = xr.Dataset(
                    {self.NeurNode6ALzt_psd_max_env.short_name : self.NeurNode6ALzt_psd_max_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode6ALzt_psd_max_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds58)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if self.is_set("NeurNode6ALzt_psd_min_env") :
            if all(len(x) == len(self.Frequency_psd) for x in self.NeurNode6ALzt_psd_min_env.values) :
                xr_ds59 = xr.Dataset(
                    {self.NeurNode6ALzt_psd_min_env.short_name : self.NeurNode6ALzt_psd_min_env.create_xr_DataArray(user_dims=user_dims_dict["NeurNode6ALzt_psd_min_env"])},
                    coords={"time_psd" : pandas.to_datetime(self.time_psd, unit = "s", origin="unix"), "Frequency_psd" :  self.Frequency_psd}, 
                    attrs={
                        "institution" : self.institution,
                        "source" : self.source,
                        "description" : self.description},
                    )
                xr_ds_l.append(xr_ds59)
                xr_ds=xr.merge(xr_ds_l,combine_attrs= "drop_conflicts")
        if xr_ds:
            if 'time' in xr_ds.dims:
                xr_ds.time.attrs['description'] = 'Unix Timestamp seconds since 1. January 1970 00:00:00 UTC - Representing date and hour of recording'
                xr_ds.time.attrs['unit'] = '[s]'
            if 'time_psd' in xr_ds.dims:
                xr_ds.time_psd.attrs['description'] = 'Unix Timestamp seconds since 1. January 1970 00:00:00 UTC - Representing date and hour of recording for PSD specifically'
                xr_ds.time_psd.attrs['unit'] = '[s]'
            if 'Frequency_psd' in xr_ds.dims:
                xr_ds.Frequency_psd.attrs['description'] = 'Frequency for Power Spectral Density'
                xr_ds.Frequency_psd.attrs['unit'] = '[Hz]'
        return xr_ds

    #-------------------------
    # Load functions
    #-------------------------
    def loadJSON(self,name = None, filePath = None):
        """Load an instance of the object from JSON format file

        Args:
            name (:obj:`str`, optional): Name of the object to load.
            filePath (:obj:`str`, optional): Path of the JSON file to load. If None, the function looks for a file with name "name".json.

        The JSON file must contain a datastructure representing an instance of this object's class, as generated by e.g. the function :func:`~saveJSON`.

        """

        if not(name == None):
            self.name = name
        if (name == None) and not(filePath == None):
            self.name = ".".join(filePath.split(os.path.sep)[-1].split(".")[0:-1])
        if (filePath == None):
            if hasattr(self, "name"):
                filePath = self.name + ".json"
            else:
                raise Exception("object needs name for loading.")
        if not(os.path.isfile(filePath)):
            raise Exception("file %s not found."%filePath)
        self._loadedItems = []
        f = open(filePath,"r")
        data = f.read()
        f.close()
        dd = json.loads(data)
        self.loadFromJSONDict(dd)
    def loadFromJSONDict(self, data):
        """Load an instance of the object from a dictionnary representing a JSON structure)

        Args:
            name (:obj:`collections.OrderedDict`): Dictionnary containing a JSON structure, as produced by e.g. :func:`json.loads`

        """

        varName = "time"
        try :
            setattr(self,varName, np.array(data[varName]))
        except :
            pass
        varName = "time_psd"
        try :
            setattr(self,varName, np.array(data[varName]))
        except :
            pass
        varName = "Frequency_psd"
        try :
            setattr(self,varName, np.array(data[varName]))
        except :
            pass
        varName = "institution"
        try :
            setattr(self,varName, data[varName])
        except :
            pass
        varName = "source"
        try :
            setattr(self,varName, data[varName])
        except :
            pass
        varName = "status_Neuron_Sensors"
        try :
            if data[varName] != None:
                self.status_Neuron_Sensors=status.status()
                self.status_Neuron_Sensors.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode1ALxt_psd"
        try :
            if data[varName] != None:
                self.NeurNode1ALxt_psd=status.status()
                self.NeurNode1ALxt_psd.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode1ALxt_psd_max_env"
        try :
            if data[varName] != None:
                self.NeurNode1ALxt_psd_max_env=status.status()
                self.NeurNode1ALxt_psd_max_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode1ALxt_psd_min_env"
        try :
            if data[varName] != None:
                self.NeurNode1ALxt_psd_min_env=status.status()
                self.NeurNode1ALxt_psd_min_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode2ALxt_psd"
        try :
            if data[varName] != None:
                self.NeurNode2ALxt_psd=status.status()
                self.NeurNode2ALxt_psd.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode2ALxt_psd_max_env"
        try :
            if data[varName] != None:
                self.NeurNode2ALxt_psd_max_env=status.status()
                self.NeurNode2ALxt_psd_max_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode2ALxt_psd_min_env"
        try :
            if data[varName] != None:
                self.NeurNode2ALxt_psd_min_env=status.status()
                self.NeurNode2ALxt_psd_min_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode3ALxt_psd"
        try :
            if data[varName] != None:
                self.NeurNode3ALxt_psd=status.status()
                self.NeurNode3ALxt_psd.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode3ALxt_psd_max_env"
        try :
            if data[varName] != None:
                self.NeurNode3ALxt_psd_max_env=status.status()
                self.NeurNode3ALxt_psd_max_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode3ALxt_psd_min_env"
        try :
            if data[varName] != None:
                self.NeurNode3ALxt_psd_min_env=status.status()
                self.NeurNode3ALxt_psd_min_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode4ALxt_psd"
        try :
            if data[varName] != None:
                self.NeurNode4ALxt_psd=status.status()
                self.NeurNode4ALxt_psd.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode4ALxt_psd_max_env"
        try :
            if data[varName] != None:
                self.NeurNode4ALxt_psd_max_env=status.status()
                self.NeurNode4ALxt_psd_max_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode4ALxt_psd_min_env"
        try :
            if data[varName] != None:
                self.NeurNode4ALxt_psd_min_env=status.status()
                self.NeurNode4ALxt_psd_min_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode5ALxt_psd"
        try :
            if data[varName] != None:
                self.NeurNode5ALxt_psd=status.status()
                self.NeurNode5ALxt_psd.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode5ALxt_psd_max_env"
        try :
            if data[varName] != None:
                self.NeurNode5ALxt_psd_max_env=status.status()
                self.NeurNode5ALxt_psd_max_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode5ALxt_psd_min_env"
        try :
            if data[varName] != None:
                self.NeurNode5ALxt_psd_min_env=status.status()
                self.NeurNode5ALxt_psd_min_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode6ALxt_psd"
        try :
            if data[varName] != None:
                self.NeurNode6ALxt_psd=status.status()
                self.NeurNode6ALxt_psd.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode6ALxt_psd_max_env"
        try :
            if data[varName] != None:
                self.NeurNode6ALxt_psd_max_env=status.status()
                self.NeurNode6ALxt_psd_max_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode6ALxt_psd_min_env"
        try :
            if data[varName] != None:
                self.NeurNode6ALxt_psd_min_env=status.status()
                self.NeurNode6ALxt_psd_min_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode1ALyt_psd"
        try :
            if data[varName] != None:
                self.NeurNode1ALyt_psd=status.status()
                self.NeurNode1ALyt_psd.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode1ALyt_psd_max_env"
        try :
            if data[varName] != None:
                self.NeurNode1ALyt_psd_max_env=status.status()
                self.NeurNode1ALyt_psd_max_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode1ALyt_psd_min_env"
        try :
            if data[varName] != None:
                self.NeurNode1ALyt_psd_min_env=status.status()
                self.NeurNode1ALyt_psd_min_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode2ALyt_psd"
        try :
            if data[varName] != None:
                self.NeurNode2ALyt_psd=status.status()
                self.NeurNode2ALyt_psd.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode2ALyt_psd_max_env"
        try :
            if data[varName] != None:
                self.NeurNode2ALyt_psd_max_env=status.status()
                self.NeurNode2ALyt_psd_max_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode2ALyt_psd_min_env"
        try :
            if data[varName] != None:
                self.NeurNode2ALyt_psd_min_env=status.status()
                self.NeurNode2ALyt_psd_min_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode3ALyt_psd"
        try :
            if data[varName] != None:
                self.NeurNode3ALyt_psd=status.status()
                self.NeurNode3ALyt_psd.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode3ALyt_psd_max_env"
        try :
            if data[varName] != None:
                self.NeurNode3ALyt_psd_max_env=status.status()
                self.NeurNode3ALyt_psd_max_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode3ALyt_psd_min_env"
        try :
            if data[varName] != None:
                self.NeurNode3ALyt_psd_min_env=status.status()
                self.NeurNode3ALyt_psd_min_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode4ALyt_psd"
        try :
            if data[varName] != None:
                self.NeurNode4ALyt_psd=status.status()
                self.NeurNode4ALyt_psd.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode4ALyt_psd_max_env"
        try :
            if data[varName] != None:
                self.NeurNode4ALyt_psd_max_env=status.status()
                self.NeurNode4ALyt_psd_max_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode4ALyt_psd_min_env"
        try :
            if data[varName] != None:
                self.NeurNode4ALyt_psd_min_env=status.status()
                self.NeurNode4ALyt_psd_min_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode5ALyt_psd"
        try :
            if data[varName] != None:
                self.NeurNode5ALyt_psd=status.status()
                self.NeurNode5ALyt_psd.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode5ALyt_psd_max_env"
        try :
            if data[varName] != None:
                self.NeurNode5ALyt_psd_max_env=status.status()
                self.NeurNode5ALyt_psd_max_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode5ALyt_psd_min_env"
        try :
            if data[varName] != None:
                self.NeurNode5ALyt_psd_min_env=status.status()
                self.NeurNode5ALyt_psd_min_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode6ALyt_psd"
        try :
            if data[varName] != None:
                self.NeurNode6ALyt_psd=status.status()
                self.NeurNode6ALyt_psd.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode6ALyt_psd_max_env"
        try :
            if data[varName] != None:
                self.NeurNode6ALyt_psd_max_env=status.status()
                self.NeurNode6ALyt_psd_max_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode6ALyt_psd_min_env"
        try :
            if data[varName] != None:
                self.NeurNode6ALyt_psd_min_env=status.status()
                self.NeurNode6ALyt_psd_min_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode1ALzt_psd"
        try :
            if data[varName] != None:
                self.NeurNode1ALzt_psd=status.status()
                self.NeurNode1ALzt_psd.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode1ALzt_psd_max_env"
        try :
            if data[varName] != None:
                self.NeurNode1ALzt_psd_max_env=status.status()
                self.NeurNode1ALzt_psd_max_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode1ALzt_psd_min_env"
        try :
            if data[varName] != None:
                self.NeurNode1ALzt_psd_min_env=status.status()
                self.NeurNode1ALzt_psd_min_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode2ALzt_psd"
        try :
            if data[varName] != None:
                self.NeurNode2ALzt_psd=status.status()
                self.NeurNode2ALzt_psd.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode2ALzt_psd_max_env"
        try :
            if data[varName] != None:
                self.NeurNode2ALzt_psd_max_env=status.status()
                self.NeurNode2ALzt_psd_max_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode2ALzt_psd_min_env"
        try :
            if data[varName] != None:
                self.NeurNode2ALzt_psd_min_env=status.status()
                self.NeurNode2ALzt_psd_min_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode3ALzt_psd"
        try :
            if data[varName] != None:
                self.NeurNode3ALzt_psd=status.status()
                self.NeurNode3ALzt_psd.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode3ALzt_psd_max_env"
        try :
            if data[varName] != None:
                self.NeurNode3ALzt_psd_max_env=status.status()
                self.NeurNode3ALzt_psd_max_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode3ALzt_psd_min_env"
        try :
            if data[varName] != None:
                self.NeurNode3ALzt_psd_min_env=status.status()
                self.NeurNode3ALzt_psd_min_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode4ALzt_psd"
        try :
            if data[varName] != None:
                self.NeurNode4ALzt_psd=status.status()
                self.NeurNode4ALzt_psd.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode4ALzt_psd_max_env"
        try :
            if data[varName] != None:
                self.NeurNode4ALzt_psd_max_env=status.status()
                self.NeurNode4ALzt_psd_max_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode4ALzt_psd_min_env"
        try :
            if data[varName] != None:
                self.NeurNode4ALzt_psd_min_env=status.status()
                self.NeurNode4ALzt_psd_min_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode5ALzt_psd"
        try :
            if data[varName] != None:
                self.NeurNode5ALzt_psd=status.status()
                self.NeurNode5ALzt_psd.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode5ALzt_psd_max_env"
        try :
            if data[varName] != None:
                self.NeurNode5ALzt_psd_max_env=status.status()
                self.NeurNode5ALzt_psd_max_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode5ALzt_psd_min_env"
        try :
            if data[varName] != None:
                self.NeurNode5ALzt_psd_min_env=status.status()
                self.NeurNode5ALzt_psd_min_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode6ALzt_psd"
        try :
            if data[varName] != None:
                self.NeurNode6ALzt_psd=status.status()
                self.NeurNode6ALzt_psd.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode6ALzt_psd_max_env"
        try :
            if data[varName] != None:
                self.NeurNode6ALzt_psd_max_env=status.status()
                self.NeurNode6ALzt_psd_max_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "NeurNode6ALzt_psd_min_env"
        try :
            if data[varName] != None:
                self.NeurNode6ALzt_psd_min_env=status.status()
                self.NeurNode6ALzt_psd_min_env.loadFromJSONDict(data[varName])
        except :
            pass
        varName = "name"
        try :
            setattr(self,varName, data[varName])
        except :
            pass
        varName = "description"
        try :
            setattr(self,varName, data[varName])
        except :
            pass
    def loadHDF5(self,name=None, filePath=None):
        """Load an instance of the object from HDF5 format file
        Args:
            name (:obj:`str`, optional): Name of the object to load. Default is None.
            filePath (:obj:`str`, optional): Path of the HDF5 file to load. If None, the function looks for a file with name "self.name".h5.
        The HDF5 file must contain a datastructure representing an instance of this objects class, as generated by e.g. the function :func:`~saveHDF5`.
        """
        # Define filepath and name
        if not(name == None):
            self.name = name
        if (filePath == None):
            if hasattr(self, "name"):
                filePath = self.name + ".h5"
            else:
                raise Exception("name is required for loading")
        # Open hdf5 file
        h = h5py.File(filePath,"r")
        group = h[self.name]
        # Save the object to the group
        self.loadFromHDF5Handle(group)
        # Close file
        h.close()
        pass
    def loadFromHDF5Handle(self, gr):
        """Load the properties of the object from a hdf5 handle.
        Args:
            gr (:obj:`h5py.Group`): Handle used to read the object properties from
        """
        if ("time" in list(gr.keys())):
            self.time = gr["time"][:]
        if ("time_psd" in list(gr.keys())):
            self.time_psd = gr["time_psd"][:]
        if ("Frequency_psd" in list(gr.keys())):
            self.Frequency_psd = gr["Frequency_psd"][:]
        if ("institution" in list(gr.keys())):
            self.institution = gr["institution"][0].decode("ascii")
        if ("source" in list(gr.keys())):
            self.source = gr["source"][0].decode("ascii")
        if ("status_Neuron_Sensors" in list(gr.keys())):
            subgroup = gr["status_Neuron_Sensors"]
            self.status_Neuron_Sensors.loadFromHDF5Handle(subgroup)
        if ("NeurNode1ALxt_psd" in list(gr.keys())):
            subgroup = gr["NeurNode1ALxt_psd"]
            self.NeurNode1ALxt_psd.loadFromHDF5Handle(subgroup)
        if ("NeurNode1ALxt_psd_max_env" in list(gr.keys())):
            subgroup = gr["NeurNode1ALxt_psd_max_env"]
            self.NeurNode1ALxt_psd_max_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode1ALxt_psd_min_env" in list(gr.keys())):
            subgroup = gr["NeurNode1ALxt_psd_min_env"]
            self.NeurNode1ALxt_psd_min_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode2ALxt_psd" in list(gr.keys())):
            subgroup = gr["NeurNode2ALxt_psd"]
            self.NeurNode2ALxt_psd.loadFromHDF5Handle(subgroup)
        if ("NeurNode2ALxt_psd_max_env" in list(gr.keys())):
            subgroup = gr["NeurNode2ALxt_psd_max_env"]
            self.NeurNode2ALxt_psd_max_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode2ALxt_psd_min_env" in list(gr.keys())):
            subgroup = gr["NeurNode2ALxt_psd_min_env"]
            self.NeurNode2ALxt_psd_min_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode3ALxt_psd" in list(gr.keys())):
            subgroup = gr["NeurNode3ALxt_psd"]
            self.NeurNode3ALxt_psd.loadFromHDF5Handle(subgroup)
        if ("NeurNode3ALxt_psd_max_env" in list(gr.keys())):
            subgroup = gr["NeurNode3ALxt_psd_max_env"]
            self.NeurNode3ALxt_psd_max_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode3ALxt_psd_min_env" in list(gr.keys())):
            subgroup = gr["NeurNode3ALxt_psd_min_env"]
            self.NeurNode3ALxt_psd_min_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode4ALxt_psd" in list(gr.keys())):
            subgroup = gr["NeurNode4ALxt_psd"]
            self.NeurNode4ALxt_psd.loadFromHDF5Handle(subgroup)
        if ("NeurNode4ALxt_psd_max_env" in list(gr.keys())):
            subgroup = gr["NeurNode4ALxt_psd_max_env"]
            self.NeurNode4ALxt_psd_max_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode4ALxt_psd_min_env" in list(gr.keys())):
            subgroup = gr["NeurNode4ALxt_psd_min_env"]
            self.NeurNode4ALxt_psd_min_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode5ALxt_psd" in list(gr.keys())):
            subgroup = gr["NeurNode5ALxt_psd"]
            self.NeurNode5ALxt_psd.loadFromHDF5Handle(subgroup)
        if ("NeurNode5ALxt_psd_max_env" in list(gr.keys())):
            subgroup = gr["NeurNode5ALxt_psd_max_env"]
            self.NeurNode5ALxt_psd_max_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode5ALxt_psd_min_env" in list(gr.keys())):
            subgroup = gr["NeurNode5ALxt_psd_min_env"]
            self.NeurNode5ALxt_psd_min_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode6ALxt_psd" in list(gr.keys())):
            subgroup = gr["NeurNode6ALxt_psd"]
            self.NeurNode6ALxt_psd.loadFromHDF5Handle(subgroup)
        if ("NeurNode6ALxt_psd_max_env" in list(gr.keys())):
            subgroup = gr["NeurNode6ALxt_psd_max_env"]
            self.NeurNode6ALxt_psd_max_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode6ALxt_psd_min_env" in list(gr.keys())):
            subgroup = gr["NeurNode6ALxt_psd_min_env"]
            self.NeurNode6ALxt_psd_min_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode1ALyt_psd" in list(gr.keys())):
            subgroup = gr["NeurNode1ALyt_psd"]
            self.NeurNode1ALyt_psd.loadFromHDF5Handle(subgroup)
        if ("NeurNode1ALyt_psd_max_env" in list(gr.keys())):
            subgroup = gr["NeurNode1ALyt_psd_max_env"]
            self.NeurNode1ALyt_psd_max_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode1ALyt_psd_min_env" in list(gr.keys())):
            subgroup = gr["NeurNode1ALyt_psd_min_env"]
            self.NeurNode1ALyt_psd_min_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode2ALyt_psd" in list(gr.keys())):
            subgroup = gr["NeurNode2ALyt_psd"]
            self.NeurNode2ALyt_psd.loadFromHDF5Handle(subgroup)
        if ("NeurNode2ALyt_psd_max_env" in list(gr.keys())):
            subgroup = gr["NeurNode2ALyt_psd_max_env"]
            self.NeurNode2ALyt_psd_max_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode2ALyt_psd_min_env" in list(gr.keys())):
            subgroup = gr["NeurNode2ALyt_psd_min_env"]
            self.NeurNode2ALyt_psd_min_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode3ALyt_psd" in list(gr.keys())):
            subgroup = gr["NeurNode3ALyt_psd"]
            self.NeurNode3ALyt_psd.loadFromHDF5Handle(subgroup)
        if ("NeurNode3ALyt_psd_max_env" in list(gr.keys())):
            subgroup = gr["NeurNode3ALyt_psd_max_env"]
            self.NeurNode3ALyt_psd_max_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode3ALyt_psd_min_env" in list(gr.keys())):
            subgroup = gr["NeurNode3ALyt_psd_min_env"]
            self.NeurNode3ALyt_psd_min_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode4ALyt_psd" in list(gr.keys())):
            subgroup = gr["NeurNode4ALyt_psd"]
            self.NeurNode4ALyt_psd.loadFromHDF5Handle(subgroup)
        if ("NeurNode4ALyt_psd_max_env" in list(gr.keys())):
            subgroup = gr["NeurNode4ALyt_psd_max_env"]
            self.NeurNode4ALyt_psd_max_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode4ALyt_psd_min_env" in list(gr.keys())):
            subgroup = gr["NeurNode4ALyt_psd_min_env"]
            self.NeurNode4ALyt_psd_min_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode5ALyt_psd" in list(gr.keys())):
            subgroup = gr["NeurNode5ALyt_psd"]
            self.NeurNode5ALyt_psd.loadFromHDF5Handle(subgroup)
        if ("NeurNode5ALyt_psd_max_env" in list(gr.keys())):
            subgroup = gr["NeurNode5ALyt_psd_max_env"]
            self.NeurNode5ALyt_psd_max_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode5ALyt_psd_min_env" in list(gr.keys())):
            subgroup = gr["NeurNode5ALyt_psd_min_env"]
            self.NeurNode5ALyt_psd_min_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode6ALyt_psd" in list(gr.keys())):
            subgroup = gr["NeurNode6ALyt_psd"]
            self.NeurNode6ALyt_psd.loadFromHDF5Handle(subgroup)
        if ("NeurNode6ALyt_psd_max_env" in list(gr.keys())):
            subgroup = gr["NeurNode6ALyt_psd_max_env"]
            self.NeurNode6ALyt_psd_max_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode6ALyt_psd_min_env" in list(gr.keys())):
            subgroup = gr["NeurNode6ALyt_psd_min_env"]
            self.NeurNode6ALyt_psd_min_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode1ALzt_psd" in list(gr.keys())):
            subgroup = gr["NeurNode1ALzt_psd"]
            self.NeurNode1ALzt_psd.loadFromHDF5Handle(subgroup)
        if ("NeurNode1ALzt_psd_max_env" in list(gr.keys())):
            subgroup = gr["NeurNode1ALzt_psd_max_env"]
            self.NeurNode1ALzt_psd_max_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode1ALzt_psd_min_env" in list(gr.keys())):
            subgroup = gr["NeurNode1ALzt_psd_min_env"]
            self.NeurNode1ALzt_psd_min_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode2ALzt_psd" in list(gr.keys())):
            subgroup = gr["NeurNode2ALzt_psd"]
            self.NeurNode2ALzt_psd.loadFromHDF5Handle(subgroup)
        if ("NeurNode2ALzt_psd_max_env" in list(gr.keys())):
            subgroup = gr["NeurNode2ALzt_psd_max_env"]
            self.NeurNode2ALzt_psd_max_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode2ALzt_psd_min_env" in list(gr.keys())):
            subgroup = gr["NeurNode2ALzt_psd_min_env"]
            self.NeurNode2ALzt_psd_min_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode3ALzt_psd" in list(gr.keys())):
            subgroup = gr["NeurNode3ALzt_psd"]
            self.NeurNode3ALzt_psd.loadFromHDF5Handle(subgroup)
        if ("NeurNode3ALzt_psd_max_env" in list(gr.keys())):
            subgroup = gr["NeurNode3ALzt_psd_max_env"]
            self.NeurNode3ALzt_psd_max_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode3ALzt_psd_min_env" in list(gr.keys())):
            subgroup = gr["NeurNode3ALzt_psd_min_env"]
            self.NeurNode3ALzt_psd_min_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode4ALzt_psd" in list(gr.keys())):
            subgroup = gr["NeurNode4ALzt_psd"]
            self.NeurNode4ALzt_psd.loadFromHDF5Handle(subgroup)
        if ("NeurNode4ALzt_psd_max_env" in list(gr.keys())):
            subgroup = gr["NeurNode4ALzt_psd_max_env"]
            self.NeurNode4ALzt_psd_max_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode4ALzt_psd_min_env" in list(gr.keys())):
            subgroup = gr["NeurNode4ALzt_psd_min_env"]
            self.NeurNode4ALzt_psd_min_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode5ALzt_psd" in list(gr.keys())):
            subgroup = gr["NeurNode5ALzt_psd"]
            self.NeurNode5ALzt_psd.loadFromHDF5Handle(subgroup)
        if ("NeurNode5ALzt_psd_max_env" in list(gr.keys())):
            subgroup = gr["NeurNode5ALzt_psd_max_env"]
            self.NeurNode5ALzt_psd_max_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode5ALzt_psd_min_env" in list(gr.keys())):
            subgroup = gr["NeurNode5ALzt_psd_min_env"]
            self.NeurNode5ALzt_psd_min_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode6ALzt_psd" in list(gr.keys())):
            subgroup = gr["NeurNode6ALzt_psd"]
            self.NeurNode6ALzt_psd.loadFromHDF5Handle(subgroup)
        if ("NeurNode6ALzt_psd_max_env" in list(gr.keys())):
            subgroup = gr["NeurNode6ALzt_psd_max_env"]
            self.NeurNode6ALzt_psd_max_env.loadFromHDF5Handle(subgroup)
        if ("NeurNode6ALzt_psd_min_env" in list(gr.keys())):
            subgroup = gr["NeurNode6ALzt_psd_min_env"]
            self.NeurNode6ALzt_psd_min_env.loadFromHDF5Handle(subgroup)
        if ("name" in list(gr.keys())):
            self.name = gr["name"][0].decode("ascii")
        if ("description" in list(gr.keys())):
            self.description = gr["description"][0].decode("ascii")
    def loadFromXRDataSet(self, data,
    time_format = "%Y-%m-%d %H:%M:%S" ):
            """Load an instance of xarray dataset)

            Args:
                name (:obj:`xarray.Dataset`): A multi-dimensional, in memory, array database.

            """

            varName = "time"
            try :
                setattr(self,varName, np.array(data.time))
                # test is_set_function compatibility 
                len(getattr(self,varName))
            except TypeError as e :
                if str(e) == 'len() of unsized object':
                    setattr(self,varName, np.array([data.data]))
                else : pass
            except :
                pass
            varName = "time_psd"
            try :
                setattr(self,varName, np.array(data.time_psd))
                # test is_set_function compatibility 
                len(getattr(self,varName))
            except TypeError as e :
                if str(e) == 'len() of unsized object':
                    setattr(self,varName, np.array([data.data]))
                else : pass
            except :
                pass
            varName = "Frequency_psd"
            try :
                setattr(self,varName, np.array(data.Frequency_psd))
                # test is_set_function compatibility 
                len(getattr(self,varName))
            except TypeError as e :
                if str(e) == 'len() of unsized object':
                    setattr(self,varName, np.array([data.data]))
                else : pass
            except :
                pass
            varName = "institution"
            try :
                setattr(self,varName, data.institution)
            except :
                pass
            varName = "source"
            try :
                setattr(self,varName, data.source)
            except :
                pass
            varName = "status_Neuron_Sensors"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.status_Neuron_Sensors=status.status()
                    self.status_Neuron_Sensors.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode1ALxt_psd"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode1ALxt_psd=status.status()
                    self.NeurNode1ALxt_psd.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode1ALxt_psd_max_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode1ALxt_psd_max_env=status.status()
                    self.NeurNode1ALxt_psd_max_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode1ALxt_psd_min_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode1ALxt_psd_min_env=status.status()
                    self.NeurNode1ALxt_psd_min_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode2ALxt_psd"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode2ALxt_psd=status.status()
                    self.NeurNode2ALxt_psd.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode2ALxt_psd_max_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode2ALxt_psd_max_env=status.status()
                    self.NeurNode2ALxt_psd_max_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode2ALxt_psd_min_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode2ALxt_psd_min_env=status.status()
                    self.NeurNode2ALxt_psd_min_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode3ALxt_psd"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode3ALxt_psd=status.status()
                    self.NeurNode3ALxt_psd.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode3ALxt_psd_max_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode3ALxt_psd_max_env=status.status()
                    self.NeurNode3ALxt_psd_max_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode3ALxt_psd_min_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode3ALxt_psd_min_env=status.status()
                    self.NeurNode3ALxt_psd_min_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode4ALxt_psd"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode4ALxt_psd=status.status()
                    self.NeurNode4ALxt_psd.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode4ALxt_psd_max_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode4ALxt_psd_max_env=status.status()
                    self.NeurNode4ALxt_psd_max_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode4ALxt_psd_min_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode4ALxt_psd_min_env=status.status()
                    self.NeurNode4ALxt_psd_min_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode5ALxt_psd"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode5ALxt_psd=status.status()
                    self.NeurNode5ALxt_psd.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode5ALxt_psd_max_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode5ALxt_psd_max_env=status.status()
                    self.NeurNode5ALxt_psd_max_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode5ALxt_psd_min_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode5ALxt_psd_min_env=status.status()
                    self.NeurNode5ALxt_psd_min_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode6ALxt_psd"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode6ALxt_psd=status.status()
                    self.NeurNode6ALxt_psd.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode6ALxt_psd_max_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode6ALxt_psd_max_env=status.status()
                    self.NeurNode6ALxt_psd_max_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode6ALxt_psd_min_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode6ALxt_psd_min_env=status.status()
                    self.NeurNode6ALxt_psd_min_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode1ALyt_psd"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode1ALyt_psd=status.status()
                    self.NeurNode1ALyt_psd.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode1ALyt_psd_max_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode1ALyt_psd_max_env=status.status()
                    self.NeurNode1ALyt_psd_max_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode1ALyt_psd_min_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode1ALyt_psd_min_env=status.status()
                    self.NeurNode1ALyt_psd_min_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode2ALyt_psd"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode2ALyt_psd=status.status()
                    self.NeurNode2ALyt_psd.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode2ALyt_psd_max_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode2ALyt_psd_max_env=status.status()
                    self.NeurNode2ALyt_psd_max_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode2ALyt_psd_min_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode2ALyt_psd_min_env=status.status()
                    self.NeurNode2ALyt_psd_min_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode3ALyt_psd"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode3ALyt_psd=status.status()
                    self.NeurNode3ALyt_psd.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode3ALyt_psd_max_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode3ALyt_psd_max_env=status.status()
                    self.NeurNode3ALyt_psd_max_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode3ALyt_psd_min_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode3ALyt_psd_min_env=status.status()
                    self.NeurNode3ALyt_psd_min_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode4ALyt_psd"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode4ALyt_psd=status.status()
                    self.NeurNode4ALyt_psd.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode4ALyt_psd_max_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode4ALyt_psd_max_env=status.status()
                    self.NeurNode4ALyt_psd_max_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode4ALyt_psd_min_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode4ALyt_psd_min_env=status.status()
                    self.NeurNode4ALyt_psd_min_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode5ALyt_psd"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode5ALyt_psd=status.status()
                    self.NeurNode5ALyt_psd.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode5ALyt_psd_max_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode5ALyt_psd_max_env=status.status()
                    self.NeurNode5ALyt_psd_max_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode5ALyt_psd_min_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode5ALyt_psd_min_env=status.status()
                    self.NeurNode5ALyt_psd_min_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode6ALyt_psd"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode6ALyt_psd=status.status()
                    self.NeurNode6ALyt_psd.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode6ALyt_psd_max_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode6ALyt_psd_max_env=status.status()
                    self.NeurNode6ALyt_psd_max_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode6ALyt_psd_min_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode6ALyt_psd_min_env=status.status()
                    self.NeurNode6ALyt_psd_min_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode1ALzt_psd"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode1ALzt_psd=status.status()
                    self.NeurNode1ALzt_psd.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode1ALzt_psd_max_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode1ALzt_psd_max_env=status.status()
                    self.NeurNode1ALzt_psd_max_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode1ALzt_psd_min_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode1ALzt_psd_min_env=status.status()
                    self.NeurNode1ALzt_psd_min_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode2ALzt_psd"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode2ALzt_psd=status.status()
                    self.NeurNode2ALzt_psd.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode2ALzt_psd_max_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode2ALzt_psd_max_env=status.status()
                    self.NeurNode2ALzt_psd_max_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode2ALzt_psd_min_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode2ALzt_psd_min_env=status.status()
                    self.NeurNode2ALzt_psd_min_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode3ALzt_psd"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode3ALzt_psd=status.status()
                    self.NeurNode3ALzt_psd.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode3ALzt_psd_max_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode3ALzt_psd_max_env=status.status()
                    self.NeurNode3ALzt_psd_max_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode3ALzt_psd_min_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode3ALzt_psd_min_env=status.status()
                    self.NeurNode3ALzt_psd_min_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode4ALzt_psd"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode4ALzt_psd=status.status()
                    self.NeurNode4ALzt_psd.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode4ALzt_psd_max_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode4ALzt_psd_max_env=status.status()
                    self.NeurNode4ALzt_psd_max_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode4ALzt_psd_min_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode4ALzt_psd_min_env=status.status()
                    self.NeurNode4ALzt_psd_min_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode5ALzt_psd"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode5ALzt_psd=status.status()
                    self.NeurNode5ALzt_psd.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode5ALzt_psd_max_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode5ALzt_psd_max_env=status.status()
                    self.NeurNode5ALzt_psd_max_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode5ALzt_psd_min_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode5ALzt_psd_min_env=status.status()
                    self.NeurNode5ALzt_psd_min_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode6ALzt_psd"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode6ALzt_psd=status.status()
                    self.NeurNode6ALzt_psd.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode6ALzt_psd_max_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode6ALzt_psd_max_env=status.status()
                    self.NeurNode6ALzt_psd_max_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "NeurNode6ALzt_psd_min_env"
            try :
                if isinstance(data[varName], xr.DataArray) :
                    self.NeurNode6ALzt_psd_min_env=status.status()
                    self.NeurNode6ALzt_psd_min_env.loadFromXRDataArray(data[varName])
            except :
                pass
            varName = "name"
            try :
                setattr(self,varName, data.name)
            except :
                pass
            varName = "description"
            try :
                setattr(self,varName, data.description)
            except :
                pass
    #------------------------
    # is_set function
    #------------------------
    def is_set(self, varName): # pragma: no cover

        """Check if a given property of the object is set

        Args:
            varName (:obj:`str`): name of the property to check

        Returns:
            :obj:`bool`: True if the property is set, else False
        """

        if (isinstance(getattr(self,varName),list) ):
            if (len(getattr(self,varName)) > 0 and not any([np.any(a==None) for a in getattr(self,varName)])  ):
                return True
            else :
                return False
        if (isinstance(getattr(self,varName),np.ndarray) ):
            if (len(getattr(self,varName)) > 0 and not any([np.any(a==None) for a in getattr(self,varName)])  ):
                return True
            else :
                return False
        if (getattr(self,varName) != None):
            return True
        return False
    #------------------------
