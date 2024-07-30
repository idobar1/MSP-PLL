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
        fname = "Daft Punk - Around the World.mp3"
        format = "mp3"

    class DebugConfig:
        cut_start_sec = 60
        cut_len_sec = 10

    class MathConfig:
        loop_gain = 0.5
        loop_filt_type = FiltType.MA
        loop_filter_mem = 10
        VCO_gain = 2*np.pi/100 
        f0 = 2  # [Hz]
