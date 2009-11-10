import FWCore.ParameterSet.Config as cms

## make gen matched electrons
from Validation.Generator.genParticleMatching_cff import muonMatch
from Validation.Generator.GenMatchedMuons_cfi import matchedMuons
makeMatchedMuons = cms.Sequence(muonMatch * matchedMuons)
