import FWCore.ParameterSet.Config as cms

# Rings
from RecoTracker.RingESSource.RingESSourceTIFTIB_cff import *
import copy
from TrackingTools.RoadSearchHitAccess.roadSearchHitDumper_cfi import *
# include RoadSearchHitDumper
roadSearchHitDumperTIFTIB = copy.deepcopy(roadSearchHitDumper)
roadSearchHitDumperTIFTIB.RingsLabel = 'TIFTIB'

