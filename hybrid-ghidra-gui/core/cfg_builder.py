def build_cfg(function_name: str | None = None) -> str:
    """Return a simple text representation of a control flow graph."""
    # This is a placeholder; replace with graph building from real data
    return (
        "main\n"
        "├── parse_input\n"
        "│   ├── validate_user\n"
        "│   └── error_exit\n"
        "└── crypto_routine\n"
        "    ├── encrypt_data\n"
        "    └── hash_password"
    )
