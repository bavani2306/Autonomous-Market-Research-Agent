TOXIC_WORDS = ["kill", "hate", "stupid"]

def evaluate_report(content):
    try:
        for w in TOXIC_WORDS:
            if w in content.lower():
                return False, f"Toxic keyword detected: {w}"
        return True, "Passed - Clean report"
    except Exception:
        return False, "Evaluation error"