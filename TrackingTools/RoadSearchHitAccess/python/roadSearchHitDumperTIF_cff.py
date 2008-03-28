import FWCore.ParameterSet.Config as cms

# Rings
from RecoTracker.RingESSource.RingESSourceTIF_cff import *
import copy
from TrackingTools.RoadSearchHitAccess.roadSearchHitDumper_cfi import *
# include RoadSearchHitDumper
roadSearchHitDumperTIF = copy.deepcopy(roadSearchHitDumper)
roadSearchHitDumperTIF.RingsLabel = 'TIF'

