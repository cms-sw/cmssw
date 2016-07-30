from Configuration.StandardSequences.Eras import eras
from RecoTracker.TkSeedGenerator.clusterCheckerEDProducer_cfi import *
# Disable too many clusters check until we have an updated cut string for phase1
eras.phase1Pixel.toModify(clusterCheckerEDProducer, doClusterCheck=False) # FIXME
