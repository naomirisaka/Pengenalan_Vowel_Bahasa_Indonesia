import re
import pandas as pd

def load_dataset(data_dir):
    records = []
    pattern = r"^S(?P<speaker>\d+)_(?P<vowel>[AIEOU])_(?P<usage>CR|CS|OS)_(?P<condition>N|HP|LP|SD|LD)\.wav$"
    
    for file_path in sorted(data_dir.rglob("*.wav")):
        match = re.match(pattern, file_path.name, re.IGNORECASE)
        if match:
            groups = match.groupdict()
            records.append({
                "file_path": str(file_path.resolve()),
                "file_name": file_path.name,
                "speaker": f"S{int(groups['speaker']):02d}",
                "vowel": groups["vowel"].upper(),
                "usage": groups["usage"].upper(),        # CR, CS, OS
                "condition": groups["condition"].upper()  # N, HP, LP, SD, LD
            })
            
    df = pd.DataFrame(records)
    return df