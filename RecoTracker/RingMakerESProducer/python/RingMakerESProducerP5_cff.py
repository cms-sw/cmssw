import FWCore.ParameterSet.Config as cms

# geometry
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
# tracker geometry
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
# tracker numbering
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
import copy
from RecoTracker.RingMakerESProducer.RingMakerESProducer_cfi import *
# rings esproducer
ringsP5 = copy.deepcopy(rings)
ringsP5.Configuration = 'P5'
ringsP5.RingAsciiFileName = 'rings_p5.dat'
ringsP5.ComponentName = 'P5'

