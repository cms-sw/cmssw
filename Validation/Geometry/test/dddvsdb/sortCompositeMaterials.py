#! /usr/bin/env python

import sys
import xml.etree.ElementTree as ET

inputFile = sys.argv[1]
print ("Reading input file ", inputFile)

tree = ET.parse(inputFile)
root = tree.getroot()

sortList = []
elem = root.find('{http://www.cern.ch/cms/DDL}MaterialSection')
for subelem in elem :
     key = subelem.get('name')
     print (key)
     sortList.append((key, subelem))

sortList.sort()
for item in sortList :
  print (item[0])

elem[:] = [item[-1] for item in sortList]

outputFile = sys.argv[2]
tree.write(outputFile)
