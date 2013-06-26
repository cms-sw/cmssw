#! /usr/bin/env python

import sys
import xml.dom
from xml.dom import minidom
import material
from domtools import DOMIterator, dom_strip

def usage():
  print """Usage:
    split.py NAME [DIRECTION CUT [CUT ...]] 

Read a list of detectors from standard input, splits them into subgrouos at the CUTs position along the given DIRECTION, named after NAME, DIRECTION and relevant CUT. 
The groups are appended to the trackingMaterialGroups.xml file - if not present an empty one is created beforehand.
"""

def split():
  if (len(sys.argv) < 2):
    usage()
    sys.exit(1)

  basename = sys.argv[1]
  if (len(sys.argv) in (2,3)):
    dir_name  = 'none'
    cuts = []
  else:
    dir_name = sys.argv[2].lower()
    if not dir_name in material.Element.directions:
      usage()
      sys.exit(1)
    cuts = [float(x) for x in sys.argv[3:]]
  direction = material.Element.directions[dir_name]
  dir_label = material.Element.dir_labels[dir_name]
  
  elements = material.parse(sys.stdin)
  if len(elements) == 0:
    sys.exit(1)

  groups = material.split_along(direction, elements, cuts)
  groups = material.remove_copy_number(elements, groups)

  try:
    # parse trackingMaterialGroups.xml
    document = minidom.parse("trackingMaterialGroups.xml")
    # remove text
    dom_strip(document)
    section = document.getElementsByTagName("SpecParSection")[0]
  except:
    # invalid document or no document to parse, create a new one
    document = minidom.getDOMImplementation().createDocument(xml.dom.XML_NAMESPACE, "DDDefinition", None)
    document.documentElement.setAttribute("xmlns", "http://www.cern.ch/cms/DDL")
    document.documentElement.setAttribute("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    document.documentElement.setAttribute("xsi:schemaLocation", "http://www.cern.ch/cms/DDL ../../../DetectorDescription/Schema/DDLSchema.xsd")
    section = document.createElement("SpecParSection")
    section.setAttribute("label", "spec-pars2.xml")
    section.appendChild(document.createTextNode(""))
    document.documentElement.appendChild(section)

  for (index, group) in enumerate(groups):
    if len(group) == 0:
    # empty group
      continue

    if len(groups) == 1:
      # layer with no subgroups, use simple name
      group_name = basename
    else:
      # layer with subgroups, build sensible names
      if index == 0:
        group_name = "%s_%s%d"% (basename, dir_label, 0)
      else:
        group_name = "%s_%s%d"% (basename, dir_label, int(round(cuts[index-1])))
    
    specpar = document.createElement("SpecPar")
    specpar.setAttribute("name", group_name)
    specpar.setAttribute("eval", "true")
    for filter in group:
      selector = document.createElement("PartSelector")
      selector.setAttribute("path", filter.full_name())
      specpar.appendChild(selector)
    groupname = document.createElement("Parameter")
    groupname.setAttribute("name", "TrackingMaterialGroup")
    groupname.setAttribute("value", group_name)
    specpar.appendChild(groupname)
    section.appendChild(specpar)
    section.appendChild(document.createTextNode(""))

  # write the updated XML
  out = open("trackingMaterialGroups.xml", "w")
  out.write(document.toprettyxml("  ", "\n", "utf-8"))
  out.close()


if __name__ == "__main__":
  split()
