import FWCore.ParameterSet.Config as cms

# Rings
from RecoTracker.RingESSource.RingESSourceTIFTOB_cff import *
import copy
from TrackingTools.RoadSearchHitAccess.roadSearchHitDumper_cfi import *
# include RoadSearchHitDumper
roadSearchHitDumperTIFTOB = copy.deepcopy(roadSearchHitDumper)
roadSearchHitDumperTIFTOB.RingsLabel = 'TIFTOB'

