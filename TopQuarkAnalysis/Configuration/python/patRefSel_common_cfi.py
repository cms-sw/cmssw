import FWCore.ParameterSet.Config as cms

### Misc

# Energy density per jet area
from RecoJets.Configuration.RecoPFJets_cff import kt6PFJets
kt6PFJetsChs = kt6PFJets.clone(
  rParam        = cms.double( 0.6 )
, src           = cms.InputTag( 'pfNoElectron' )
, doAreaFastjet = cms.bool( True )
, doRhoFastjet  = cms.bool( True )
, voronoiRfact  = cms.double( 0.9 )
)

### Producers

### Selection filters
