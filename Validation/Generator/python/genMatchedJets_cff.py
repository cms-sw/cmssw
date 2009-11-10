import FWCore.ParameterSet.Config as cms

## make parton matched calo jets
from Validation.Generator.genParticleMatching_cff import jetPartonMatch
from Validation.Generator.GenMatchedJets_cfi import partonMatchedJets
makePartonMatchedJets = cms.Sequence(jetPartonMatch * partonMatchedJets)

## make gen jet matched calo jets
from Validation.Generator.genParticleMatching_cff import jetGenJetMatch
from Validation.Generator.GenMatchedJets_cfi import genJetMatchedJets
makeGenJetMatchedJets = cms.Sequence(jetGenJetMatch * genJetMatchedJets)
