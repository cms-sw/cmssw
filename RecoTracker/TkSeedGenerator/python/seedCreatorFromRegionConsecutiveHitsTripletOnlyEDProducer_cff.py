from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer_cfi import *
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer,
   magneticField = '',
   propagator = 'PropagatorWithMaterial',
)
