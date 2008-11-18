import FWCore.ParameterSet.Config as cms
#
# pat tuple sequence for fullsim
#

from PhysicsTools.PatAlgos.patLayer0_cff                        import *
from PhysicsTools.PatAlgos.patLayer1_cff                        import *
from TopQuarkAnalysis.TopObjectProducers.patTuple_Defaults_cff  import *
#from TopQuarkAnalysis.TopObjectProducers.patTuple_Reco_cff     import *
from PhysicsTools.HepMCCandAlgos.genEventProcID_cfi             import *

## std sequence for patTuple production
patTuple = cms.Sequence(patLayer0_patTuple * ## to be used from PhysicsTools/PatAlgos V04-14-03 onwards
                        genEventProcID *     ## needs HepMCProduct in the event content
                        #patLayer0 *
                        patLayer1# *
#                       patCaloTaus          ## skipped as common taus are caloTaus at the moment
                        )
