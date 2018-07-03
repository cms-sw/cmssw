import sys
import re
import bisect
import copy
import math
from python25 import *

class ElementLevel:
  name = ""
  copy = 0

  def parse(self, name):
    pattern  = re.compile(r'([A-Za-z0-9]+)\[([0-9]+)\]')
    match = pattern.match(name)
    self.name = match.group(1)
    self.copy = int(match.group(2))

  def full_name(self):
    if self.copy == 0:
      return self.name
    else:
      return "%s[%d]" % (self.name, self.copy)

  # self matches pattern iff:
  #  - they have the same name
  #  - they have the same index, or pattern has index 0
  def match(self, pattern):
    return (self.name == pattern.name) and ((pattern.copy == 0) or (self.copy == pattern.copy))

  def __init__(self, name, copy = None):
    if copy is None:
      self.parse(name)
    else:
      self.name = name
      self.copy = copy

  def __str__(self):
    return self.full_name()


# base class for Element and ElementFilter
# defines __hash__ for speed, but it's NOT generically immutable
class ElementBase:
  name = []

  def parse(self, name):
    self.name = [ElementLevel(item) for item in name.lstrip('/').split('/')]

  def full_name(self):
    return "//" + "/".join([item.full_name() for item in self.name])

  def levels(self):
    return len(self.name)

  def __init__(self, other = None):
    if other is None:
      self.name = []
    elif isinstance(other, ElementBase):
      self.name = other.name
    elif isinstance(other, str):
      self.parse(other)
    else:
      raise TypeError("Cannot initialize an ElementBase from type %s" % type(other))

  def __str__(self):
    return self.full_name()

  def __eq__(self, other):
    return self.name == other.name
  
  def __ne__(self, other):
    return self.name != other.name

  def __hash__(self):
    return hash(self.full_name())


class ElementFilter(ElementBase):

  def match(self, element):
    if self.levels() != element.levels():
      return False
    return all( [element.name[i].match( self.name[i] ) for i in range(self.levels())] )


class Element(ElementBase):
  directions = {
    'none'      : 0,
    'r'         : 1, 
    'z'         : 2, 
    'abs(z)'    : 3, 
    '|z|'       : 3, 
    'eta'       : 4, 
    'abs(eta)'  : 5,
    '|eta|'     : 5,
    'phi'       : 6 
  }
  dir_labels = {
    'none'      : 'None',
    'r'         : 'R', 
    'z'         : 'Z', 
    'abs(z)'    : 'Z', 
    '|z|'       : 'Z', 
    'eta'       : 'Eta', 
    'abs(eta)'  : 'Eta',
    '|eta|'     : 'Eta',
    'phi'       : 'Phi' 
  }
  position = (0, 0, 0, 0, 0, 0, 0)

  def match(self, pattern):
    if self.levels() != pattern.levels():
      return False
    return all( [self.name[i].match( pattern.name[i] ) for i in range(self.levels())] )

  def __init__(self, position, other):
    ElementBase.__init__(self, other)
    self.position = position


def parse(source):
  pattern = re.compile(r'(.*) *\(([0-9-.]+) *, *([0-9-.]+) *, *([0-9-.]+)\)')
  elements = []
  for line in source:
    match = pattern.match(line)
    if not match:
      print 'Warning: the following line does not match the parsing rules:'
      print line
      print
      continue
    r   = float(match.group(2))
    z   = float(match.group(3))
    phi = float(match.group(4))
    eta = -math.log(r / (z + math.sqrt(r*r + z*z)))
    position = (0, r, z, abs(z), eta, abs(eta), phi)
    name = match.group(1)
    elements.append(Element(position, name))
  return elements


# collapse the elements into a set of filters
def collapse(elements):
  size = elements[0].levels()
  names = [{} for i in range(size)]
  filters = [ ElementFilter(element) for element in elements ]
  for i in range(size):
    for filter in filters:
      name = filter.name[i].name
      copy = filter.name[i].copy
      if name not in names[i]:
        # new name
        names[i][name] = copy
      elif names[i][name] and names[i][name] != copy:
        # name with different copy number
        names[i][name] = 0
    for filter in filters:
      name = filter.name[i].name
      copy = filter.name[i].copy
      if names[i][name] == copy:
        filter.name[i].copy = 0
  return filters


# elements is as returned by parse()
# cuts must be sorted in ascending order
def split_along(direction, elements, cuts):
  filters = collapse(elements)
  groups  = [[] for i in range(len(cuts)+1)]
  for element in elements:
    i = bisect.bisect(cuts, element.position[direction])
    matching_filters = [ filter for filter in filters if filter.match(element) ]
    if len(matching_filters) == 0:
      print "Error: no matches for element %s" % element.full_name()
    elif len(matching_filters) > 1:
      print "Error: too many matches for element %s" % element.full_name()
    else:
      groups[i].append( matching_filters[0] )
  return groups

# return True if and only if all elements match one and only one filter group
def check_groups(elements, groups):
  for element in elements:
    matching_groups = set()
    for (index, group) in enumerate(groups):
      for filter in group:
        if element.match(filter):
          matching_groups.add(index)
          break
    if len(matching_groups) == 0:
      # filters have lost validity
      return False
    elif len(matching_groups) > 1:
      # filters have lost discriminating power
      return False
  # all elements match one and only one filter group
  return True


def remove_copy_number(elements, old_groups):
  levels = elements[0].levels()
  for level in range(levels):
    # if this level has no copy number, skip it
    if not any([filter.name[level].copy for group in old_groups for filter in group]):
      continue
    # try to remove the copy number for this level
    new_groups = [[] for group in old_groups]
    for (new, old) in zip(new_groups, old_groups):
      cache = []
      for filter in old:
        new_filter = copy.deepcopy(filter)
        new_filter.name[level].copy = 0
        new_hash = hash(new_filter.full_name())
        if new_hash not in cache:
          new.append(new_filter)
          cache.append(new_hash)
        
    # check if the filter are still working (each element must belong to one and only one group)
    if check_groups(elements, new_groups):
      # the copy number can be safely removed
      old_groups = new_groups
  
  return old_groups

