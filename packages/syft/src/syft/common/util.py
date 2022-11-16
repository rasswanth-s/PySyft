

def full_name_with_qualname(klass: type) -> str:
    """Returns the klass module name + klass qualname."""
    return f"{klass.__module__}.{klass.__qualname__}"

def full_name_with_name(klass: type) -> str:
    """Returns the klass module name + klass name."""
    return f"{klass.__module__}.{klass.__name__}"