#! /usr/bin/env python

import sys
import material

def usage():
  print """Usage:
    dump.py DIRECTION

Read a list of detectors from standard input and dump their coordinate along DIRECTION.
"""

def dump():
  if (len(sys.argv) < 2) or (sys.argv[1] not in material.Element.directions):
    usage()
    sys.exit(1)
  dir = material.Element.directions[sys.argv[1]]

  elements  = material.parse(sys.stdin)
  if len(elements) == 0:
    sys.exit(1)

  positions = set()
  for element in elements:
    positions.add(element.position[dir])
  positions = sorted(positions)
  for position in positions:
    print position


if __name__ == "__main__":
  dump()
