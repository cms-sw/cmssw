import FWCore.ParameterSet.Config as cms

#
# tqaf layer1 default sequence for fastsim
#

## extra includes for fastsim
from PhysicsTools.PatAlgos.famos.famosSequences_cff import *

#-----------------------------------------------------------------
# build tqafLayer0 objects (jets, muons, electrons, mets, taus)
#-----------------------------------------------------------------
from PhysicsTools.PatAlgos.patLayer0_cff import *

## define the tqafLayer0 input
from TopQuarkAnalysis.TopObjectProducers.fast.tqafLayer0_objectCleaning_cff import *
from TopQuarkAnalysis.TopObjectProducers.fast.tqafLayer0_mcMatching_cff import *

#-----------------------------------------------------------------
# build tqafLayer1 objects (jets, muons, electrons, mets, taus)
#-----------------------------------------------------------------
from PhysicsTools.PatAlgos.patLayer1_cff import *

## define the tqafLayer1 input
from TopQuarkAnalysis.TopObjectProducers.fast.tqafLayer1_objectProduction_cff import *
from TopQuarkAnalysis.TopObjectProducers.fast.tqafLayer1_objectSelection_cff import *

## std sequence for tqafLayer1 production (w/o trigger for fastsim)
tqafLayer1_withoutTrigMatch = cms.Sequence(patLayer0_withoutTrigMatch*patLayer1)


