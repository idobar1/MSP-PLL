from dataclasses import dataclass
import numpy as np
from attr import frozen
from strenum import StrEnum


class FiltType(StrEnum):
    MA = "MA"
    GAIN = "GAIN"

@frozen
class Config:
    class FileConfig:
        fname = "Queen - Another One Bites the Dust.mp3"
        format = "mp3"

    class DebugConfig:
        cut_start_sec = 7
        cut_len_sec = 20

    class MathConfig:  # worked well
        loop_gain = 0.008
        loop_filt_type = FiltType.MA
        loop_filter_mem = 15000
        VCO_gain = 2*np.pi/100 
        f0 = 2.4 # [Hz]
