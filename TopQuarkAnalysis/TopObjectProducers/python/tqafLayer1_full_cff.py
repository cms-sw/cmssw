import FWCore.ParameterSet.Config as cms

#
# tqaf layer1 default sequence for fullsim
#

#-----------------------------------------------------------------
# build tqafLayer0 objects (jets, muons, electrons, mets, taus)
#-----------------------------------------------------------------
from PhysicsTools.PatAlgos.patLayer0_cff import *

## define the tqafLayer0 input
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer0_objectCleaning_cff import *
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer0_mcMatching_cff import *


#-----------------------------------------------------------------
# build tqafLayer1 objects (jets, muons, electrons, mets, taus)
#-----------------------------------------------------------------
from PhysicsTools.PatAlgos.patLayer1_cff import *

## define the tqafLayer1 input
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer1_objectProduction_cff import *
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer1_objectSelection_cff import *

## std sequence for tqafLayer1 production
tqafLayer1 = cms.Sequence(patLayer0*patLayer1)

## std sequence for tqafLayer1 production (w/o trigger)
#tqafLayer1_withoutTrigMatch = cms.Sequence(patLayer0_withoutTrigMatch*patLayer1)

