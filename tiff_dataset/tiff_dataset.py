import os
from pathlib import PurePosixPath
from typing import Any, Dict
import fsspec
import rasterio
from kedro.io import AbstractDataSet
from kedro.io.core import get_filepath_str, get_protocol_and_path
import numpy as np
from rasterio.enums import Resampling
import logging

log = logging.getLogger(__name__)

class TiffDataSet(AbstractDataSet):
    """``TiffDataSet`` loads / save image data from a given filepath as `numpy` array using Pillow.

    Example:
    ::

        >>> TiffDataSet(filepath='/img/file/path.png')
    """

    def __init__(self, filepath: str):
        """Creates a new instance of TiffDataSet to load / save image data for given filepath.

        Args:
            filepath: The location of the image file to load / save data.
        """
        self._protocol, self.path = get_protocol_and_path(filepath)
        self._fs = fsspec.filesystem(self._protocol)
        self._filepath = PurePosixPath(self.path)
        self.path = get_filepath_str(self._filepath, self._protocol)

    def _load(self) -> rasterio.DatasetReader:
        """Gets reader of the image file.

        Returns:
            rasterIO dataset reader
            #####Numpy array of shape (nb_of_channels, rows, cols)
        """
        return rasterio.open(self.path, "r")

    def get_upscaling_factors(self, channels_readers: Dict[str, rasterio.DatasetReader]) -> Dict[str, int]:
        """
        Returns the max channel shape, and upscaling factors for each channel to match that max_shape.
        """
        max_shape = max(channels_readers.values(), key=lambda x: x.shape).shape
        upscaling_factors = {"max_shape": max_shape}

        for channnel, reader in channels_readers.items():
            channel_shape = reader.shape
            log.info("channel shape: %s" %str(channel_shape))
            upscaling_factors[channnel] = (max_shape[0]/channel_shape[0], max_shape[1]/channel_shape[1])

        return upscaling_factors

    def _save(self, data: Dict[str, rasterio.DatasetReader]) -> None:
        """Saves image data to the specified filepath.
        
        Args:
         Data is a dict that has channels as keys and single band readers as values

        """
        os.makedirs(os.path.dirname(self._filepath), exist_ok=True)
        number_of_channels = 0

        factors = self.get_upscaling_factors(data)
        all_bands = np.array([])

        for channel, reader in data.items():
            profile = reader.profile
            log.info("metadata: %s" %profile)
            channel_shape = reader.shape
            number_of_channels += channel_shape[0] # if rgb, adds 3
            print(type(reader))
            print(reader)

            if factors[channel] != (1,1): # resample
                log.info("Resampling...")
                new_data = reader.read(out_shape=(1, int(reader.height * factors[channel][0]),
                                        int(reader.width * factors[channel][1])),
                                        resampling=Resampling.bilinear)
                            
                # # scale image transform
                # transform = src.transform * src.transform.scale(
                #     (src.width / new_data.shape[-1]),
                #     (src.height / new_data.shape[-2])
                # )
            else:
                new_data = reader.read() # can't read
                transform = reader.transform

            reader.close()

            # Stack the arrays
            if not np.any(all_bands): # first band
                all_bands = new_data
            else:
                all_bands = np.concatenate((all_bands, new_data), axis=0)


        profile.update(transform=transform, count=number_of_channels)
        log.debug("new profile:", profile)
        with rasterio.open(self.path, 'w', **profile) as dst:
            for i, band in enumerate(all_bands):
                print("SHAPE:", str(band.shape))
                dst.write(band, i+1) # band index starts at 1

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._filepath, protocol=self._protocol)

    def __len__():
        return 1
