import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.patSequences_cff import *
from TopQuarkAnalysis.Configuration.patRefSel_common_cfi import *

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

step4 = countPatMuons.clone( maxNumber = 1 ) # includes the signal muon

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

step5 = countPatElectrons.clone( maxNumber = 0 )

