from pandas import read_csv
from torch import load as torch_load
from pickle import load as load_pickle


def get_sounds(records_dir: str, names_dir: str) -> tuple:
    records = torch_load(records_dir)
    sounds = records[:, :, 1:].clone()

    with open(names_dir, "rb") as f:
        names = load_pickle(f)

    return sounds, names


def get_sound_notes(csv_dir: str, id_col: str, notes_col: str, names: dict):
    df = read_csv(csv_dir)
    df.set_index(id_col, inplace=True)

    sound_notes = []
    for x in names.values():
        note = df.loc[x, notes_col]
        sound_notes.append(note)

    return sound_notes


def sound_note_contains(sound_notes: list, contains: str) -> list:
    contains = contains.lower()

    note_contains = []
    for note in sound_notes:
        note_lower = note.lower()
        to_match = contains in note_lower
        note_contains.append(to_match * 1)

    return note_contains
