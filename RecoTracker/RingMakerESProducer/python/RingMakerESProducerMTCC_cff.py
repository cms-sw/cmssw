import FWCore.ParameterSet.Config as cms

# geometry
from Geometry.CMSCommonData.cmsMTCCGeometryXML_cfi import *
# tracker geometry
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
# tracker numbering
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
import copy
from RecoTracker.RingMakerESProducer.RingMakerESProducer_cfi import *
# rings esproducer
ringsMTCC = copy.deepcopy(rings)
ringsMTCC.RingAsciiFileName = 'rings_mtcc.dat'
ringsMTCC.ComponentName = 'MTCC'

