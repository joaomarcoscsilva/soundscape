from .settings import settings

BIRD_CLASSES = [
    "basi_culi",
    "myio_leuc",
    "vire_chiv",
    "cycl_guja",
    "pita_sulp",
    "zono_cape",
]

FROG_CLASSES = [
    "dend_minu",
    "apla_leuc",
    "isch_guen",
    "phys_cuvi",
    "boan_albo",
    "aden_marm",
]

ALL_CLASSES = BIRD_CLASSES + FROG_CLASSES

CLASS_INDEX = {c: i + 1 for i, c in enumerate(ALL_CLASSES)}

MAX_EVENTS = 18

SR = 44100

CLASS_NAMES = {
    2: ["bird", "frog"],
    12: ALL_CLASSES,
    13: ["other"] + ALL_CLASSES,
}[settings["data"]["num_classes"]]
