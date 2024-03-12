from RecoTracker.TkHitPairs.hitPairEDProducerDefault_cfi import hitPairEDProducerDefault as _hitPairEDProducerDefault

hitPairEDProducer = _hitPairEDProducerDefault.clone()
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(hitPairEDProducer, maxElement=0)
# foo bar baz
# FdnZy0ZKPS0W3
# 45MFkGql5FuEX
