# define some functions missing in python 2.4

if not 'any' in dir(__builtins__):
    def any(iterable):
        """
        Return True if any element of the iterable is true.
        New in version 2.5.
        """
        for element in iterable:
            if element:
                return True
        return False


if not 'all' in dir(__builtins__):
    def all(iterable):
        """
        Return True if all elements of the iterable are true.
        New in version 2.5.
        """
        for element in iterable:
            if not element:
                return False
        return True

