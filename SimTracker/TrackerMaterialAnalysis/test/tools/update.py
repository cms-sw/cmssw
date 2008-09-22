#! /usr/bin/env python
# reads trackingMaterialGroups.xml (take it from SimTracker/TrackerMaterialAnalysis/data) 
# and parameters.xml (output from SimTracker/TrackerMaterialAnalysis/test/trackingMaterialAnalyser.py)
# and produces trackerRecoMaterial.xml (suitable for Geometry/TrackerRecoData/data)

import sys
import xml.dom
from xml.dom import minidom
import material
from domtools import DOMIterator, dom_strip

# Look for a child node like <Parameter name="name" value="value"/> .
# Return a tuple (child, value) is such node is found.
def findParameter(element, parameter):
  for child in element.getElementsByTagName("Parameter"):
    if child.hasAttribute("name") and child.getAttribute("name") == parameter:
      return (child, child.getAttribute("value"))
  return (None, "")


def update():
  # parse trackingMaterialGroups.xml
  document = minidom.parse("trackingMaterialGroups.xml")
  # remove text
  dom_strip(document)

  # parse parameters.xml into a map
  data = minidom.parse("parameters.xml")
  material = dict()
  for group in data.getElementsByTagName("Group"):
    name = group.getAttribute("name")
    radlen = findParameter(group, "TrackerRadLength")[1]
    dedx   = findParameter(group, "TrackerXi")[1]
    material[name] = (radlen, dedx)

  for group in document.getElementsByTagName("SpecPar"):
    (parameter, name) = findParameter(group, "TrackingMaterialGroup")
    parameter.parentNode.removeChild(parameter)
    parameter.unlink()
    parameter = document.createElement("Parameter")
    parameter.setAttribute("name",  "TrackerRadLength")
    parameter.setAttribute("value", material[name][0])
    group.appendChild(parameter)
    parameter = document.createElement("Parameter")
    parameter.setAttribute("name",  "TrackerXi")
    parameter.setAttribute("value", material[name][1])
    group.appendChild(parameter)

  # write the updated XML
  out = open("trackerRecoMaterial.xml", "w")
  out.write(document.toprettyxml("  ", "\n", "utf-8"))
  out.close()


if __name__ == "__main__":
  update()
