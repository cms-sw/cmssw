import FWCore.ParameterSet.Config as cms

#
# tqaf layer1 default sequence for fastsim
#

## extra includes for fastsim not needed anymore?
## from PhysicsTools.PatAlgos.famos.famosSequences_cff import *


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

# build tqafLayer1 objects (calo taus from scratch)
from TopQuarkAnalysis.TopObjectProducers.tqafLayer1_caloTaus_cff              import *
# build tqafLayer2 commons (genEvt for top decay chain)
from TopQuarkAnalysis.TopEventProducers.tqafLayer2_common_cff                 import *




## std sequence for tqafLayer1 production (w/o trigger)
tqafLayer1_withoutTrigMatch = cms.Sequence(patLayer0_withoutTrigMatch *
                                           patLayer1                  *
                                           tqafLayer1_caloTaus        *
                                           tqafLayer2_common
                                           )

