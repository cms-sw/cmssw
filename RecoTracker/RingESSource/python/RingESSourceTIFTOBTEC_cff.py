import FWCore.ParameterSet.Config as cms

# geometry
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
# tracker geometry
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
# tracker numbering
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
import copy
from RecoTracker.RingESSource.RingESSource_cfi import *
# rings esproducer
ringsTIFTOBTEC = copy.deepcopy(rings)
ringsTIFTOBTEC.InputFileName = 'RecoTracker/RingESSource/data/rings_tiftobtec-0004.dat'
ringsTIFTOBTEC.ComponentName = 'TIFTOBTEC'

