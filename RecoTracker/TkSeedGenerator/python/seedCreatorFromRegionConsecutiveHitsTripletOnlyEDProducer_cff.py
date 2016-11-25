from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer_cfi import *
from Configuration.Eras.Modifier_trackingPhase1PU70_cff import trackingPhase1PU70
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase1PU70.toModify(seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer,
   magneticField = '',
   propagator = 'PropagatorWithMaterial',
)
trackingPhase2PU140.toModify(seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer,
   magneticField = '',
   propagator = 'PropagatorWithMaterial',
)
