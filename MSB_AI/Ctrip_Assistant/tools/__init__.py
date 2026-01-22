from pathlib import Path

basic_dir = Path(__file__).resolve().parent.parent

old_db = f"{basic_dir}/travel.sqlite"

new_db = f"{basic_dir}/travel_new.sqlite"