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
  eidTight            = cms.InputTag( 'eidTight' )
, eidLoose            = cms.InputTag( 'eidLoose' )
, eidRobustTight      = cms.InputTag( 'eidRobustTight' )
, eidRobustHighEnergy = cms.InputTag( 'eidRobustHighEnergy' )
, eidRobustLoose      = cms.InputTag( 'eidRobustLoose' )
, eidVeryLooseMC      = cms.InputTag( 'eidVeryLooseMC' )
, eidLooseMC          = cms.InputTag( 'eidLooseMC' )
, eidMediumMC         = cms.InputTag( 'eidMediumMC' )
, eidTightMC          = cms.InputTag( 'eidTightMC' )
, eidSuperTightMC     = cms.InputTag( 'eidSuperTightMC' )
, eidHyperTight1MC    = cms.InputTag( 'eidHyperTight1MC' )
, eidHyperTight2MC    = cms.InputTag( 'eidHyperTight2MC' )
, eidHyperTight3MC    = cms.InputTag( 'eidHyperTight3MC' )
, eidHyperTight4MC    = cms.InputTag( 'eidHyperTight4MC' )
)

### Selection filters
