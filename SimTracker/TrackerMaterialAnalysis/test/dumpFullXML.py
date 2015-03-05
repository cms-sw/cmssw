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

def getTrackerRecoMaterialCopy(filename):
    tracker_reco_material = os.path.join(os.environ['CMSSW_RELEASE_BASE'],
                                         'src/Geometry/TrackerRecoData/data/trackerRecoMaterial.xml')
    if not os.path.exists(tracker_reco_material):
        print 'Something is wrong with the CMSSW installation. The file %s is missing. Quitting.\n' % tracker_reco_material
        sys.exit(TRACKER_MATERIAL_FILE_MISSING)
    copy2(tracker_reco_material, filename)
                                                
def produceXMLFromParameterFile():
    """
    Starting from the file parameters.xml produced by the
    TrackingMaterialAnalyser via cmsRun, it writes out a new XML,
    taking into account the proper names and grouping of detectors
    together.

    The skeleton of the XML is taken directly from the release the
    user is currently using, i.e. from
    $CMSSW_RELEASE_BASE/src/Geometry/TrackerRecoData/data/trackerRecoMaterial.xml.

    A new file, named trackerRecoMaterialUpdated.xml, is saved in the
    current directory.
    """

    tracker_reco_material = './trackerRecoMaterialFromRelease.xml'
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

def compareNewXMLWithOld():
    """
    Comutes the difference between the old values, stored in the
    central repository for the current release, i.e. from
    $CMSSW_RELEASE_BASE/src/Geometry/TrackerRecoData/data/trackerRecoMaterial.xml,
    and the new values that we assume are present in the same file
    under the locally installed release, i.e. under
    $CMSSW_BASE/src/Geometry/TrackerRecoData/data/trackerRecoMaterial.xml. No
    check is performed to guarantee that the files are already there:
    a missing file will result in an exception.

    The output of this function is a formatted structured as:
    ComponentsName KindOfParameter OldValue NewValue Difference
    where the Difference is computed as (NewValue-OldValue)/OldValue.

    Results are flushed at the terminal, nothing is saved.
    """
    
    tracker_reco_material = './trackerRecoMaterialFromRelease.xml'
    tracker_reco_material_updated = os.path.join(os.environ['CMSSW_BASE'],
                                                 'src/Geometry/TrackerRecoData/data/trackerRecoMaterial.xml')
    ET.register_namespace('', "http://www.cern.ch/cms/DDL")
    tree = ET.parse(tracker_reco_material)
    root = tree.getroot()
    tree_updated = ET.parse(tracker_reco_material_updated)
    root_updated = tree_updated.getroot()
    sections = root.getchildren()

    for spec_par in root.iter('%sSpecPar' % TAG_PREFIX):
        current_detector = spec_par.attrib['name']
        for parameter in spec_par.iter('%sParameter' % TAG_PREFIX):
            updated_current_detector_node = root_updated.find(".//%sSpecPar[@name='%s']" % (TAG_PREFIX,current_detector))
            if updated_current_detector_node:
                for child in updated_current_detector_node:
                    name = child.get('name', None)
                    if name and name == parameter.attrib['name']:
                        print "%s %s %s %s %f%%" % (current_detector,
                                                    parameter.attrib['name'],
                                                    parameter.attrib['value'],
                                                    child.attrib['value'],
                                                    (float(child.attrib['value'])-float(parameter.attrib['value']))
                                                    /float(parameter.attrib['value'])*100.
                                                    )
            else:
                print 'Element not found: %s' % current_detector
    
    
if __name__ == '__main__':
    checkEnvironment()
    getTrackerRecoMaterialCopy('trackerRecoMaterialFromRelease.xml')
#    produceXMLFromParameterFile()
    compareNewXMLWithOld()
