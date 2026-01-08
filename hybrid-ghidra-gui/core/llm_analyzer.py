def analyze_function(function_name: str) -> dict:
    """Return a simple analysis dict for the given function name."""
    if function_name == "parse_input":
        return {
            "summary": "Potential buffer overflow detected.",
            "prediction": "HIGH RISK FUNCTION",
            "fix": "Sanitize input length",
        }
    if function_name == "main":
        return {
            "summary": "Standard main function with typical control flow.",
            "prediction": "LOW RISK FUNCTION",
            "fix": "None required",
        }
    return {
        "summary": f"No specific analysis available for {function_name}.",
        "prediction": "UNKNOWN",
        "fix": "Review manually",
    }
