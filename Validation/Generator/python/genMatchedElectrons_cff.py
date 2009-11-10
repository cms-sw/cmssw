import FWCore.ParameterSet.Config as cms

## make gen matched electrons
from Validation.Generator.genParticleMatching_cff import electronMatch
from Validation.Generator.GenMatchedElectrons_cfi import matchedGsfElectrons
makeMatchedGsfElectrons = cms.Sequence(electronMatch * matchedGsfElectrons)
