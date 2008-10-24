import FWCore.ParameterSet.Config as cms

# magnetic field
from MagneticField.Engine.uniformMagneticField_cfi import *
# cms geometry
# tracker geometry
# tracker numbering
# roads
from RecoTracker.RoadMapMakerESProducer.RoadMapMakerESProducerTIF_cff import *
import copy
from RecoTracker.RoadSearchSeedFinder.RoadSearchSeeds_cfi import *
# RoadSearchSeedFinder
roadSearchSeedsTIF = copy.deepcopy(roadSearchSeeds)
roadSearchSeedsTIF.Mode = 'STRAIGHT-LINE'
roadSearchSeedsTIF.doClusterCheck = True
roadSearchSeedsTIF.RoadsLabel = 'TIF'
roadSearchSeedsTIF.InnerSeedRecHitAccessMode = 'STANDARD'
roadSearchSeedsTIF.InnerSeedRecHitAccessUseRPhi = True
roadSearchSeedsTIF.InnerSeedRecHitAccessUseStereo = True
roadSearchSeedsTIF.OuterSeedRecHitAccessMode = 'STANDARD'
roadSearchSeedsTIF.OuterSeedRecHitAccessUseRPhi = True
roadSearchSeedsTIF.OuterSeedRecHitAccessUseStereo = True

