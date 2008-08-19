import FWCore.ParameterSet.Config as cms

#
# tqaf layer1 event content is equivalent to pat layer0 & 1
#
from PhysicsTools.PatAlgos.patLayer1_EventContent_cff import *

tqafLayer1EventContent_slim = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'drop *_genParticles_*_*'
    )
)

tqafLayer1EventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'drop *_genParticles_*_*',
    'keep *_selectedLayer1JetsKT4Calo_*_*'  ##,
##  'keep *_selectedLayer1JetsKT5Calo_*_*' ,
##  'keep *_selectedLayer1JetsSC5PFlow_*_*',
##  'keep *_selectedLayer1JetsKT4PFlow_*_*',
##  'keep *_selectedLayer1JetsKT6PFlow_*_*'
    )
)
