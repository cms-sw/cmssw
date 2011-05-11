import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.patSequences_cff import *

### Muons

intermediatePatMuons = cleanPatMuons.clone(
  preselection  = '' # looseMuonCut
)
loosePatMuons = cleanPatMuons.clone(
  src           = cms.InputTag( 'intermediatePatMuons' )
, checkOverlaps = cms.PSet(
    jets = cms.PSet(
      src                 = cms.InputTag( 'goodPatJets' )
    , algorithm           = cms.string( 'byDeltaR' )
    , preselection        = cms.string( '' )
    , deltaR              = cms.double( 0. ) # muonJetsDR
    , checkRecoComponents = cms.bool( False )
    , pairCut             = cms.string( '' )
    , requireNoOverlaps   = cms.bool( True)
    )
  )
)
step3b = cms.EDFilter(
  "PATCandViewCountFilter"
, src = cms.InputTag( 'loosePatMuons' )
, minNumber = cms.uint32( 1 )
, maxNumber = cms.uint32( 1 )
)

tightPatMuons = cleanPatMuons.clone(
  src           = cms.InputTag( 'loosePatMuons' )
, preselection  = '' # tightMuonCut
, checkOverlaps = cms.PSet()
)
step3a = step3b.clone( src = cms.InputTag( 'tightPatMuons' ) )

step4 = cms.EDFilter(
  "PATCandViewCountFilter"
, src = cms.InputTag( 'selectedPatMuons' )
, minNumber = cms.uint32( 0 )
, maxNumber = cms.uint32( 1 ) # includes the signal muon
)

### Jets

goodPatJets = cleanPatJets.clone(
  preselection  = '' # jetCut
, checkOverlaps = cms.PSet(
    muons = cms.PSet(
      src                 = cms.InputTag( 'intermediatePatMuons' )
    , algorithm           = cms.string( 'byDeltaR' )
    , preselection        = cms.string( '' )
    , deltaR              = cms.double( 0. ) # jetMuonsDR
    , checkRecoComponents = cms.bool( False )
    , pairCut             = cms.string( '' )
    , requireNoOverlaps   = cms.bool( True)
    )
  )
)

from RecoJets.JetProducers.kt4PFJets_cfi import kt4PFJets
kt6PFJets = kt4PFJets.clone(
  rParam = cms.double( 0.6 )
, src    = cms.InputTag( 'pfNoElectron' )
, doAreaFastjet = cms.bool( True )
, doRhoFastjet = cms.bool( True )
, voronoiRfact = cms.double( 0.9 )
)

step6a = cms.EDFilter(
  "PATCandViewCountFilter"
, src = cms.InputTag( 'goodPatJets' )
, minNumber = cms.uint32( 1 )
, maxNumber = cms.uint32( 999999 )
)
step6b = step6a.clone( minNumber = 2 )
step6c = step6a.clone( minNumber = 3 )
step7  = step6a.clone( minNumber = 4 )

### Electrons

step5 = cms.EDFilter(
  "PATCandViewCountFilter"
, src = cms.InputTag( 'selectedPatElectrons' )
, minNumber = cms.uint32( 0 )
, maxNumber = cms.uint32( 0 )
)

