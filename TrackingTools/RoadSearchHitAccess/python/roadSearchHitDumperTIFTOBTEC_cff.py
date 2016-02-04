import FWCore.ParameterSet.Config as cms

# Rings
from RecoTracker.RingESSource.RingESSourceTIFTOBTEC_cff import *
import copy
from TrackingTools.RoadSearchHitAccess.roadSearchHitDumper_cfi import *
# include RoadSearchHitDumper
roadSearchHitDumperTIFTOBTEC = copy.deepcopy(roadSearchHitDumper)
roadSearchHitDumperTIFTOBTEC.RingsLabel = 'TIFTOBTEC'

