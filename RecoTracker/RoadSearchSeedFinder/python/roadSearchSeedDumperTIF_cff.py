import FWCore.ParameterSet.Config as cms

# Rings
from RecoTracker.RingMakerESProducer.RingMakerESProducerTIF_cff import *
import copy
from RecoTracker.RoadSearchSeedFinder.roadSearchSeedDumper_cfi import *
# include RoadSearchSeedDumper
roadSearchSeedDumperTIF = copy.deepcopy(roadSearchSeedDumper)
roadSearchSeedDumperTIF.RoadSearchSeedInputTag = 'roadSearchSeedsTIF'
roadSearchSeedDumperTIF.RingsLabel = 'TIF'
roadSearchSeedsTIF.InnerSeedRecHitAccessMode = 'STANDARD'
roadSearchSeedsTIF.InnerSeedRecHitAccessUseRPhi = True
roadSearchSeedsTIF.InnerSeedRecHitAccessUseStereo = True
roadSearchSeedsTIF.OuterSeedRecHitAccessMode = 'STANDARD'
roadSearchSeedsTIF.OuterSeedRecHitAccessUseRPhi = True
roadSearchSeedsTIF.OuterSeedRecHitAccessUseStereo = True

