import FWCore.ParameterSet.Config as cms

#-----------------------------------------------------------------
#
#
# pat Layer 0 & 1 standard sequences with refined defaults for 
# top analyses
#
#
#-----------------------------------------------------------------

# build tqafLayer0 objects (jets, muons, electrons, mets, taus)
from PhysicsTools.PatAlgos.patLayer0_cff                                      import *
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer0_objectCleaning_cff   import *
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer0_triggerMatching_cff  import *
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer0_mcMatching_cff       import *

# build tqafLayer1 objects (jets, muons, electrons, mets, taus)
from PhysicsTools.PatAlgos.patLayer1_cff                                      import *
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer1_objectProduction_cff import *
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer1_objectSelection_cff  import *


#-----------------------------------------------------------------
#
#
# pat layer 0 & 1 sequences for top analyses in addition to pat 
# standard sequences
#
#
#-----------------------------------------------------------------

## std sequence for tqafLayer1 production
tqafLayer1 = cms.Sequence(patLayer0_patTuple  *
                          patLayer1
                          )

## std sequence for tqafLayer1 production (w/o trigger)
tqafLayer1_withoutTrigMatch = cms.Sequence(patLayer0_withoutTrigMatch *
                                           patLayer1
                                           )

