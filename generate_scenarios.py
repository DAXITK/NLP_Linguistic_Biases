import json

def load_scenarios(filename="scenarios.json"):
    """Load scenarios with validation"""
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert all(lang in data for lang in ["English", "Hindi"]), \
           "Missing language in scenarios"
    return data

if __name__ == "__main__":
    scenarios = load_scenarios()
    print(f"Loaded {sum(len(v) for v in scenarios['English'].values())} English scenarios")
