import FWCore.ParameterSet.Config as cms

#
# tqaf layer1 default sequence for fastsim
#

#-----------------------------------------------------------------
# build tqafLayer0 objects (jets, muons, electrons, mets, taus)
#-----------------------------------------------------------------
from PhysicsTools.PatAlgos.patLayer0_cff import *

## define the tqafLayer0 input
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer0_Jets_cff import *
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer0_Muons_cff import *
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer0_Elecs_cff import *
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer0_Taus_cff import *
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer0_METs_cff import *

#-----------------------------------------------------------------
# build tqafLayer1 objects (jets, muons, electrons, mets, taus)
#-----------------------------------------------------------------
from PhysicsTools.PatAlgos.patLayer1_cff import *

## define the tqafLayer1 input
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer1_Jets_cff import *
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer1_Muons_cff import *
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer1_Elecs_cff import *
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer1_Taus_cff import *
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer1_METs_cff import *

## std sequence for tqafLayer1 production
tqafLayer1 = cms.Sequence(patLayer0*patLayer1)

## std sequence for tqafLayer1 production (w/o trigger)
tqafLayer1_withoutTrigMatch = cms.Sequence(patLayer0_withoutTrigMatch*patLayer1)

