## @package ErrorManager
# \brief Error management for ValidationTools
#
# Developers:
#   Victor E. Bazterra
#   Kenneth James Smith

## Generic error for the server application
class JobMonitorError(Exception):
  ## Default constructor
  def __init__(self, value):
    self.value = value
  ## String overflow
  def __str__(self):
    return repr(self.value)

## Generic error for the interface classes
class InterfaceError(Exception):
  ## Default constructor
  def __init__(self, value):
    self.value = value
  ## String overflow
  def __str__(self):
    return repr(self.value)

## Generic error for the publisher classes
class PublisherError(Exception):
  ## Default constructor
  def __init__(self, value):
    self.value = value
  ## String overflow
  def __str__(self):
    return repr(self.value)

## Generic error for the publisher classes
class SubscriptionError(Exception):
  ## Default constructor
  def __init__(self, value):
    self.value = value
  ## String overflow
  def __str__(self):
    return repr(self.value)

## Generic error for the publisher classes
class RSSParserError(Exception):
  ## Default constructor
  def __init__(self, value):
    self.value = value
  ## String overflow
  def __str__(self):
    return repr(self.value)
