#!/usr/bin/env python

import argparse
import os, sys
import pprint
from shutil import copy2
import xml.etree.ElementTree as ET

TAG_PREFIX='{http://www.cern.ch/cms/DDL}'
CMSSW_NOT_SET=1
TRACKER_MATERIAL_FILE_MISSING=2
LOCAL_RM = 'trackerRecoMaterialFromRelease.xml'

HEADER = """
#ifndef SIMTRACKER_TRACKERMATERIALANALYSIS_LISTGROUPS_MATERIALDIFFERENCE_H
#define SIMTRACKER_TRACKERMATERIALANALYSIS_LISTGROUPS_MATERIALDIFFERENCE_H

void ListGroups::fillMaterialDifferences() {
"""

TRAILER = """
}

#endif //  SIMTRACKER_TRACKERMATERIALANALYSIS_LISTGROUPS_MATERIALDIFFERENCE_H
"""

def checkEnvironment():
    """
    Check if the CMSSW environment is set. If not, quit the program.
    """
    if not 'CMSSW_RELEASE_BASE' in os.environ.keys():
        print 'CMSSW Environments not setup, quitting\n'
        sys.exit(CMSSW_NOT_SET)

def checkFileInRelease(filename):
   """
   Check if the supplied @filename is avaialble under the central release installation.
   Returns None if the file is not present, the full pathname otherwise.
   """
   fullpathname = os.path.join(os.environ['CMSSW_RELEASE_BASE'], filename)
   if not os.path.exists(fullpathname):
     return None
   return fullpathname

def getTrackerRecoMaterialCopy(source_xml, filename):
    """
    Take the central @source_XML file file from either the local CMSSW
    installation or, if missing, from the central one and copies it over
    locally into a file named @filename.
    The full relative pathname of source_XML must be supplied, staring
    from the src folder (included).
    If the file cannot be found anywhere, quit the program.
    """
    tracker_reco_material = os.path.join(os.environ['CMSSW_BASE'],
                                         source_xml)
    if not os.path.exists(tracker_reco_material):
      tracker_reco_material = os.path.join(os.environ['CMSSW_RELEASE_BASE'],
                                           source_xml)
      if not os.path.exists(tracker_reco_material):
          print 'Something is wrong with the CMSSW installation. The file %s is missing. Quitting.\n' % source_xml
          sys.exit(TRACKER_MATERIAL_FILE_MISSING)
    copy2(tracker_reco_material, filename)

def produceXMLFromParameterFile(args):
    """
    Starting from the file parameters.xml produced by the
    TrackingMaterialAnalyser via cmsRun, it writes out a new XML,
    taking into account the proper names and grouping of detectors
    together.

    The skeleton of the XML file is taken as an input paramter to this
    function, in order to support running on several geometries
    with one single script.

    A new file, named trackerRecoMaterial.xml, is saved in the
    current directory.
    """

    getTrackerRecoMaterialCopy(args.xml, LOCAL_RM)
    tracker_reco_material = LOCAL_RM
    tracker_reco_material_updated = './parameters.xml'
    ET.register_namespace('', "http://www.cern.ch/cms/DDL")
    tree = ET.parse(tracker_reco_material)
    root = tree.getroot()
    tree_updated = ET.parse(tracker_reco_material_updated)
    root_updated = tree_updated.getroot()
    sections = root.getchildren()
    if args.verbose:
        for child in sections[0]:
            print child.attrib['name']

    for spec_par in root.iter('%sSpecPar' % TAG_PREFIX):
        current_detector = spec_par.attrib['name']
        for parameter in spec_par.iter('%sParameter' % TAG_PREFIX):
            if args.verbose:
                print "Current Detector: %r, name=%s, value=%s" % (current_detector,
                                                                   parameter.attrib['name'],
                                                                   parameter.attrib['value'])
            updated_current_detector_node = root_updated.find(".//Group[@name='%s']" % current_detector)
            if updated_current_detector_node is not None:
              for child in updated_current_detector_node:
                  if child.attrib['name'] == parameter.attrib['name']:
                      parameter.set('name', child.attrib['name'])
                      parameter.set('value', child.attrib['value'])
                      if args.verbose:
                          print "Updated Detector: %r, name=%s, value=%s\n" % (child.attrib['name'],
                                                                               parameter.attrib['name'],
                                                                               parameter.attrib['value'])
            else:
              print "Missing group: %s" % current_detector
    tree.write('trackerRecoMaterial.xml', encoding='UTF-8', xml_declaration=True)

def compareNewXMLWithOld(args):
    """Computes the difference between the old values, stored in the
    central repository for the current release, e.g. from
    $CMSSW_RELEASE_BASE/src/Geometry/TrackerRecoData/data/trackerRecoMaterial.xml,
    and the new values that we assume are present in the same file
    under the locally installed release, i.e. under
    $CMSSW_BASE/src/Geometry/TrackerRecoData/data/trackerRecoMaterial.xml. No
    check is performed to guarantee that the files are already
    there. If the file is not there, it is searched in the current
    folder. A missing file will result in an exception.

    The output of this function is a formatted structured as:
    ComponentsName KindOfParameter OldValue NewValue Difference
    where the Difference is computed as (NewValue-OldValue)/OldValue.

    Results are flushed at the terminal. The header file
    ListGroupsMaterialDifference.h is automatically created.

    """

    getTrackerRecoMaterialCopy(args.xml, LOCAL_RM)
    tracker_reco_material = LOCAL_RM
    tracker_reco_material_updated = os.path.join(os.environ['CMSSW_BASE'],
                                                 'src/SimTracker/TrackerMaterialAnalysis/test/trackerRecoMaterial.xml')
    if not os.path.exists(tracker_reco_material_updated):
        tracker_reco_material_updated = './trackerRecoMaterial.xml'
        if not os.path.exists(tracker_reco_material_updated):
            raise os.error('Missing trackerRecoMaterial.xml file.')
    ET.register_namespace('', "http://www.cern.ch/cms/DDL")
    tree = ET.parse(tracker_reco_material)
    root = tree.getroot()
    tree_updated = ET.parse(tracker_reco_material_updated)
    root_updated = tree_updated.getroot()
    sections = root.getchildren()

    header = open(os.path.join(os.environ['CMSSW_BASE'],
                               'src/SimTracker/TrackerMaterialAnalysis/plugins/ListGroupsMaterialDifference.h'), 'w')
    header.write(HEADER)
    differences = {}
    values = {}
    ordered_keys = []
    for spec_par in root.iter('%sSpecPar' % TAG_PREFIX):
        current_detector = spec_par.attrib['name']
        ordered_keys.append(current_detector)
        for parameter in spec_par.iter('%sParameter' % TAG_PREFIX):
            updated_current_detector_node = root_updated.find(".//%sSpecPar[@name='%s']" % (TAG_PREFIX,current_detector))
            if updated_current_detector_node is not None:
                for child in updated_current_detector_node:
                    name = child.get('name', None)
                    if name and name == parameter.attrib['name']:
                        differences.setdefault(current_detector, {}).setdefault(name, [float(parameter.attrib['value']),
                                                                                       float(child.attrib['value']),
                                                                                       ((float(child.attrib['value'])-float(parameter.attrib['value']))
                                                                                       /float(parameter.attrib['value'])*100.)]
                                                                                )
            else:
                print 'Element not found: %s' % current_detector
    for group in differences.keys():
        header.write('  m_diff["%s"] = std::make_pair<float, float>(%f, %f);\n' % (group,
                                                                                   differences[group]['TrackerRadLength'][2],
                                                                                   differences[group]['TrackerXi'][2]))
    for group in differences.keys():
        header.write('  m_values["%s"] = std::make_pair<float, float>(%f, %f);\n' % (group,
                                                                                     differences[group]['TrackerRadLength'][1],
                                                                                     differences[group]['TrackerXi'][1]))
#    pprint.pprint(differences)
    for i in xrange(len(ordered_keys)):
        key = ordered_keys[i]
        if args.twiki:
            print "| %s | %f | %f | %f%% | %f | %f | %f%% |" % (key,
                                                                differences[key]['TrackerRadLength'][0],
                                                                differences[key]['TrackerRadLength'][1],
                                                                differences[key]['TrackerRadLength'][2],
                                                                differences[key]['TrackerXi'][0],
                                                                differences[key]['TrackerXi'][1],
                                                                differences[key]['TrackerXi'][2]
                                                                )
        else:
            print "%s %f %f %f%% %f %f %f%%" % (key,
                                                differences[key]['TrackerRadLength'][0],
                                                differences[key]['TrackerRadLength'][1],
                                                differences[key]['TrackerRadLength'][2],
                                                differences[key]['TrackerXi'][0],
                                                differences[key]['TrackerXi'][1],
                                                differences[key]['TrackerXi'][2]
                                                )
    header.write(TRAILER)
    header.close

def createTMGFromRelease(args):
    """
    Create a local TrackingMaterialGroup file starting from
    a tracking material reco XML file in release. This is
    useful for the very first time the users want to test a
    geoemetry for which no such file has ever been created
    in this package.
    """
    tracker_reco_material = checkFileInRelease(args.createTMG)
    if not tracker_reco_material:
      print "Input file not found in release, quitting"
      sys.exit(1)
    ET.register_namespace('', "http://www.cern.ch/cms/DDL")
    tree = ET.parse(tracker_reco_material)
    root = tree.getroot()
    sections = root.getchildren()

    for spec_par in root.iter('%sSpecPar' % TAG_PREFIX):
        spec_par.attrib.pop('eval', None)
        # Cannot remove elements in place: store them all here and remove them later on.
        to_be_removed = []
        for parameter in spec_par.iter('%sParameter' % TAG_PREFIX):
          to_be_removed.append(parameter)
        el = ET.Element("Parameter")
        el.set('name', 'TrackingMaterialGroup')
        el.set('value', spec_par.attrib['name'])
        spec_par.append(el)
        for d in to_be_removed:
          spec_par.remove(d)
    tree.write('trackingMaterialGroupFromRelease.xml', encoding='UTF-8', xml_declaration=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Easily manipulate and inspect XML files related to Tracking Material.')
    parser.add_argument('-x', '--xml',
                        default = 'src/Geometry/TrackerRecoData/data/trackerRecoMaterial.xml',
                        help="""Source XML file used to perform
                        all actions against it.
                        For PhaseI use:
                        src/Geometry/TrackerRecoData/data/PhaseI/pixfwd/trackerRecoMaterial.xml
                        For phaseII use:
                        src/Geometry/TrackerRecoData/data/PhaseII/TiltedTracker/trackerRecoMaterial.xml
                        """,
                        required=False)
    parser.add_argument('-p', '--produce', action='store_true',
                        default=False,
                        help='Produce a trackerRecoMaterial.xml starting from the paramters.xml file produced by the trackingMaterialProducer.')
    parser.add_argument('-c', '--compare', action='store_true',
                        default=False,
                        help='Compares a local trackerRecoMaterial.xml against the one bundled with the release.')
    parser.add_argument('-w', '--twiki', action='store_true',
                        default=False,
                        help="""Compares a local trackerRecoMaterial.xml against the one bundled
                                with the release and produces and output that is Twiki compatible
                                to be put into a table.""")
    parser.add_argument('--createTMG',
                       help="""Given an input trackerRecoMaterial.xml from the release,
                               it will produce locally a trackingMaterialGroups.xml files.
                               The main difference is the addition of a proper naming
                               parameter, so that the listGroups will be able to identify
                               and print the detectors gathered in a single group.
                               No definition of radiation length nor energy loss is
                               maintained in the converion process. It is mainly useful
                               only at the very beginnig of the tuning process, when
                               maybe a local trackingMaterialGroups is missing.""")
    parser.add_argument('-v', '--verbose',
                        default=False,
                        help="""Be verbose while performing
                        the required action""",
                        action='store_true')
    args = parser.parse_args()
    checkEnvironment()
    if args.produce:
      produceXMLFromParameterFile(args)
    if args.compare or args.twiki:
      compareNewXMLWithOld(args)
    if args.createTMG != None:
      createTMGFromRelease(args)
