import FWCore.ParameterSet.Config as cms
#
# pat tuple sequence for fullsim
#

from PhysicsTools.PatAlgos.patLayer0_cff                        import *
from PhysicsTools.PatAlgos.patLayer1_cff                        import *
from TopQuarkAnalysis.TopObjectProducers.patTuple_Reco_cff      import *
from TopQuarkAnalysis.TopObjectProducers.patTuple_Defaults_cff  import *

## std sequence for tqafLayer1 production
patTuple = cms.Sequence(patLayer0_patTuple *
                        patLayer1 *
                        patCaloTaus
                        )
