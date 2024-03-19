"""imbalance custom exceptions.

This module contains the implementation of Custom Exceptions.

"""


class InvalidWorkspacePathError(Exception):
    """
    Raised when the environment variable of workspace exist but path is invalid

    Args:
        Exception (Exception): InvalidWorkspacePathError
    """


class InvalidConfigError(Exception):
    """
    Raised when config value is not applicable
    Args:
        Exception (Exception): InvalidConfigError
    """


class ObjectCannotBeCreatedException(Exception):
    """
    Raised when object cannot be created for some reason
    Args:
        Exception (Exception): ObjectCannotBeCreatedException
    """
