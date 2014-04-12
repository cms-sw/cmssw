import FWCore.ParameterSet.Config as cms

### Misc

# Energy density per jet area
from RecoJets.Configuration.RecoJets_cff import ak5CaloJets
from RecoJets.Configuration.JetIDProducers_cff import ak5JetID
from RecoJets.Configuration.RecoPFJets_cff import ak5PFJets

### Producers

# CiC electron ID
electronIDSources = cms.PSet(
  mvaTrigV0    = cms.InputTag("mvaTrigV0")
, mvaNonTrigV0 = cms.InputTag("mvaNonTrigV0")
)

### Selection filters
