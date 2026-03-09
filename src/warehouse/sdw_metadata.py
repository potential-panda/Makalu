import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any

# Mapping of prefix to human readable concept
PREFIX_MAP = {
    "ta_": "Trade Aggression (Buy/Sell pressure)",
    "bs_": "Book State (Static orderbook features)",
    "bc_": "Book Change (Dynamic orderbook changes, pulls/refills)",
    "ep_": "Execution Price (Expected slippage and spread)",
    "cf_": "Cross-Feature (Interactions between trades and book)",
    "ll_": "Lead-Lag (Relationship between leader and lagger)",
    "vo_": "Volatility and spread dynamics"
}

class FeatureMetadataExtractor:
    def __init__(self, src_path: str):
        self.src_path = Path(src_path)
        self.feature_pattern = re.compile(r'features\[f?["\'](?P<name>[\w_{}]+)["\']\]\s*=')
        self.comment_pattern = re.compile(r'#\s*(?P<comment>.*)')

    def extract_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        metadata = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            match = self.feature_pattern.search(line)
            if match:
                name = match.group('name')
                # Look for comment on the same line or line before
                description = ""
                if "#" in line:
                    description = line.split("#")[-1].strip()
                elif i > 0 and self.comment_pattern.search(lines[i-1]):
                    description = self.comment_pattern.search(lines[i-1]).group('comment').strip()
                
                # Identify prefix
                prefix = name[:3] if name[2] == "_" else ""
                category = PREFIX_MAP.get(prefix, "Unknown")
                
                metadata.append({
                    "name": name,
                    "description": description,
                    "category": category,
                    "prefix": prefix,
                    "source_file": file_path.name
                })
        return metadata

    def run(self):
        all_metadata = []
        fe_path = self.src_path / "feature_engineering"
        for py_file in fe_path.glob("*.py"):
            if py_file.name == "__init__.py" or py_file.name == "utils.py":
                continue
            all_metadata.extend(self.extract_from_file(py_file))
            
        return all_metadata

if __name__ == "__main__":
    # Assuming we run from workspace root
    extractor = FeatureMetadataExtractor("Makalu/src")
    data = extractor.run()
    
    output_path = Path("Makalu/src/warehouse/features.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Extracted {len(data)} features to {output_path}")
