import FWCore.ParameterSet.Config as cms

# Rings
from RecoTracker.RingESSource.RingESSourceTIFTIBTOB_cff import *
import copy
from TrackingTools.RoadSearchHitAccess.roadSearchHitDumper_cfi import *
# include RoadSearchHitDumper
roadSearchHitDumperTIFTIBTOB = copy.deepcopy(roadSearchHitDumper)
roadSearchHitDumperTIFTIBTOB.RingsLabel = 'TIFTIBTOB'

