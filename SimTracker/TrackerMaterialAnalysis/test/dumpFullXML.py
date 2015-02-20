#!/usr/bin/env python

import os, sys
from shutil import copy2
import xml.etree.ElementTree as ET

TAG_PREFIX='{http://www.cern.ch/cms/DDL}'
CMSSW_NOT_SET=1
TRACKER_MATERIAL_FILE_MISSING=2

def checkEnvironment():
    if not 'CMSSW_RELEASE_BASE' in os.environ.keys():
        print 'CMSSW Environments not setup, quitting\n'
        sys.exit(CMSSW_NOT_SET)

def getTrackerRecoMaterialCopy():
    tracker_reco_material = os.path.join(os.environ['CMSSW_RELEASE_BASE'],
                                         'src/Geometry/TrackerRecoData/data/trackerRecoMaterial.xml')
    if not os.path.exists(tracker_reco_material):
        print 'Something is wrong with the CMSSW installation. The file %s is missing. Quitting.\n' % tracker_reco_material
        sys.exit(TRACKER_MATERIAL_FILE_MISSING)
    copy2(tracker_reco_material, './trackerRecoMaterial.xml')
                                                
def main():
    tracker_reco_material = './trackerRecoMaterial.xml'
    tracker_reco_material_updated = './parameters.xml'
    ET.register_namespace('', "http://www.cern.ch/cms/DDL")
    tree = ET.parse(tracker_reco_material)
    root = tree.getroot()
    tree_updated = ET.parse(tracker_reco_material_updated)
    root_updated = tree_updated.getroot()
    sections = root.getchildren()
    for child in sections[0]:
        print child.attrib['name']

    for spec_par in root.iter('%sSpecPar' % TAG_PREFIX):
        current_detector = spec_par.attrib['name']
        for parameter in spec_par.iter('%sParameter' % TAG_PREFIX):
            print current_detector, parameter.attrib['name'], parameter.attrib['value']
            updated_current_detector_node = root_updated.find(".//Group[@name='%s']" % current_detector)
            for child in updated_current_detector_node:
                if child.attrib['name'] == parameter.attrib['name']:
                    parameter.set('name', child.attrib['name'])
                    parameter.set('value', child.attrib['value'])
                    print current_detector, parameter.attrib['name'], parameter.attrib['value']
    tree.write('trackerRecoMaterialUpdated.xml', encoding='UTF-8', xml_declaration=True)

if __name__ == '__main__':
    checkEnvironment()
    getTrackerRecoMaterialCopy()
    main()
