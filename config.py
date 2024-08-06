from strenum import StrEnum


class FiltType(StrEnum):
    MA = "MA"
    GAIN = "GAIN"

# @frozen
class Config:
    class FileConfig:
        fname = "Queen - Another One Bites the Dust.mp3"
        format = "mp3"

    class DebugConfig:
        cut_start_sec = 7
        cut_len_sec = 20
