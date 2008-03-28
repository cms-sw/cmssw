import FWCore.ParameterSet.Config as cms

# geometry
from Geometry.CMSCommonData.cmsMTCCGeometryXML_cfi import *
# tracker geometry
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
# tracker numbering
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
import copy
from RecoTracker.RingESSource.RingESSource_cfi import *
# rings esproducer
ringsMTCC = copy.deepcopy(rings)
ringsMTCC.InputFileName = 'RecoTracker/RingESSource/data/rings_mtcc-0004.dat'
ringsMTCC.ComponentName = 'MTCC'

