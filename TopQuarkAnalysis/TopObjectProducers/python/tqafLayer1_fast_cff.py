import FWCore.ParameterSet.Config as cms

#
# tqaf layer 1 default sequence
#
# extra includes and replacements
from PhysicsTools.PatAlgos.famos.famosSequences_cff import *
#-----------------------------------------------------------------
# build the TopLayer0 Objects (Jets, Muons, Electrons, METs, Taus)
#-----------------------------------------------------------------
from PhysicsTools.PatAlgos.patLayer0_cff import *
# include "PhysicsTools/PatAlgos/data/famos/patLayer0_FamosSetup.cff"
# define the TopLayer0 input
from TopQuarkAnalysis.TopObjectProducers.fast.tqafLayer0_Jets_cff import *
from TopQuarkAnalysis.TopObjectProducers.fast.tqafLayer0_Muons_cff import *
from TopQuarkAnalysis.TopObjectProducers.fast.tqafLayer0_Elecs_cff import *
from TopQuarkAnalysis.TopObjectProducers.fast.tqafLayer0_Taus_cff import *
from TopQuarkAnalysis.TopObjectProducers.fast.tqafLayer0_METs_cff import *
#-----------------------------------------------------------------
# build the TopLayer1 Objects (Jets, Muons, Electrons, METs, Taus)
#-----------------------------------------------------------------
from PhysicsTools.PatAlgos.patLayer1_cff import *
# include "PhysicsTools/PatAlgos/data/famos/patLayer1_FamosSetup.cff"
# define the TopLayer1 Object input selection
from TopQuarkAnalysis.TopObjectProducers.fast.tqafLayer1_Jets_cff import *
from TopQuarkAnalysis.TopObjectProducers.fast.tqafLayer1_Muons_cff import *
from TopQuarkAnalysis.TopObjectProducers.fast.tqafLayer1_Elecs_cff import *
from TopQuarkAnalysis.TopObjectProducers.fast.tqafLayer1_Taus_cff import *
from TopQuarkAnalysis.TopObjectProducers.fast.tqafLayer1_METs_cff import *
tqafLayer1_withoutTrigMatch = cms.Sequence(patLayer0_withoutTrigMatch*patLayer1)
allLayer1Photons.addTrigMatch = False

