"""Fixed SkillRL template retrieval from the public task description."""
import json
from pathlib import Path


def category(benchmark, goal):
    goal = goal.lower()
    if benchmark == "alfworld":
        if "look at" in goal and "under" in goal:
            return "look_at_obj_in_light"
        for key in ("clean", "heat", "cool"):
            if key in goal:
                return key
        return "examine" if "examine" in goal or "find" in goal else "pick_and_place"
    terms = {
        "apparel": "shirt dress jacket pant coat sweater blouse clothing clothes t-shirt",
        "footwear": "shoe boot sneaker sandal heel slipper footwear",
        "electronics": "laptop phone computer tablet charger cable headphone speaker camera electronic",
        "accessories": "necklace ring bracelet earring watch jewelry bag purse wallet",
        "home_decor": "furniture lamp curtain pillow bedding decor candle vase rug",
        "beauty_health": "cream lotion shampoo conditioner moisturizer serum makeup beauty vitamin supplement",
    }
    return next((key for key, words in terms.items() if any(w in goal for w in words.split())), "other")


def fixed_memory(benchmark):
    path = Path(__file__).resolve().parent.parent / "skills" / (benchmark + ".json")
    bank = json.loads(path.read_text(encoding="utf-8"))

    def retrieve(goal):
        rules = [f"{s['title']}: {s['principle']}" for s in bank["general_skills"][:6]]
        for s in bank["task_specific_skills"][category(benchmark, goal)]:
            rules.append(f"{s['title']}: {s['principle']} {s.get('when_to_apply', '')}")
        for s in bank["common_mistakes"][:5]:
            rules.append(f"{s['description']} {s.get('how_to_avoid', '')}")
        return "\n".join(rules)

    return retrieve
