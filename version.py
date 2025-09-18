"""CompeteAI version information."""

# This file is automatically updated by the release process
__version__ = "2.0.0"

# Major.Minor.Patch version components
VERSION_MAJOR = 2
VERSION_MINOR = 0
VERSION_PATCH = 0

# Development status
# Options: "alpha", "beta", "rc", "final"
VERSION_STATUS = "beta"

# Version string including development status
VERSION = f"{__version__}"
if VERSION_STATUS != "final":
    VERSION = f"{__version__}-{VERSION_STATUS}"