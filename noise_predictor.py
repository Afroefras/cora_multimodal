from utils.noise_utils import get_sounds, get_sound_notes, sound_note_contains

sounds, names = get_sounds(
    records_dir="data/tensors/resized_records_5sec.pt",
    names_dir="data/tensors/records_names_5sec.xz",
)

sound_notes = get_sound_notes(
    csv_dir="data/physionet.org/files/ephnogram/1.0.0/ECGPCGSpreadsheet.csv",
    id_col="Record Name",
    notes_col="PCG Notes",
    names=names,
)

sound_is_good = sound_note_contains(sound_notes, contains="Good")
