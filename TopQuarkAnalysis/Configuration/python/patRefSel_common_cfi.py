import FWCore.ParameterSet.Config as cms

### Misc

# Energy density per jet area
from RecoJets.Configuration.RecoJets_cff import ak5CaloJets
from RecoJets.Configuration.JetIDProducers_cff import ak5JetID
from RecoJets.Configuration.RecoPFJets_cff import ak5PFJets

# Average energy density
from RecoJets.Configuration.RecoPFJets_cff import kt6PFJets
kt6PFJetsChs = kt6PFJets.clone(
  rParam        = cms.double( 0.6 )
, src           = cms.InputTag( 'pfNoElectron' )
, doAreaFastjet = cms.bool( True )
, doRhoFastjet  = cms.bool( True )
, voronoiRfact  = cms.double( -0.9 )
)

### Producers

# CiC electron ID
electronIDSources = cms.PSet(
  mvaTrigV0    = cms.InputTag("mvaTrigV0")
, mvaNonTrigV0 = cms.InputTag("mvaNonTrigV0")
)

### Selection filters
