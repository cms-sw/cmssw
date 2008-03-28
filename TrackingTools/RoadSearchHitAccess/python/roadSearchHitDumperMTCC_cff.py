import FWCore.ParameterSet.Config as cms

# Rings
from RecoTracker.RingESSource.RingESSourceMTCC_cff import *
import copy
from TrackingTools.RoadSearchHitAccess.roadSearchHitDumper_cfi import *
# include RoadSearchHitDumper
roadSearchHitDumperMTCC = copy.deepcopy(roadSearchHitDumper)
roadSearchHitDumperMTCC.RingsLabel = 'MTCC'

