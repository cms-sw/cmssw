import FWCore.ParameterSet.Config as cms

# Rings
from RecoTracker.RingMakerESProducer.RingMakerESProducerMTCC_cff import *
import copy
from RecoTracker.RoadSearchSeedFinder.roadSearchSeedDumper_cfi import *
# include RoadSearchSeedDumper
roadSearchSeedDumperMTCC = copy.deepcopy(roadSearchSeedDumper)
roadSearchSeedDumperMTCC.RoadSearchSeedInputTag = 'roadSearchSeedsMTCC'
roadSearchSeedDumperMTCC.RingsLabel = 'MTCC'
roadSearchSeedsMTCC.InnerSeedRecHitAccessMode = 'STANDARD'
roadSearchSeedsMTCC.InnerSeedRecHitAccessUseRPhi = True
roadSearchSeedsMTCC.InnerSeedRecHitAccessUseStereo = True
roadSearchSeedsMTCC.OuterSeedRecHitAccessMode = 'STANDARD'
roadSearchSeedsMTCC.OuterSeedRecHitAccessUseRPhi = True
roadSearchSeedsMTCC.OuterSeedRecHitAccessUseStereo = True

