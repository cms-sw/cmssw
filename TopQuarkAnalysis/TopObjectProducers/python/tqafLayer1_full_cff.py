import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.patLayer0_cff import *
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer0_Jets_cff import *
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer0_Muons_cff import *
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer0_Elecs_cff import *
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer0_Taus_cff import *
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer0_METs_cff import *
from PhysicsTools.PatAlgos.patLayer1_cff import *
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer1_Jets_cff import *
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer1_Muons_cff import *
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer1_Elecs_cff import *
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer1_Taus_cff import *
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer1_METs_cff import *
tqafLayer1 = cms.Sequence(patLayer0*patLayer1)
tqafLayer1_withoutTrigMatch = cms.Sequence(patLayer0_withoutTrigMatch*patLayer1)

