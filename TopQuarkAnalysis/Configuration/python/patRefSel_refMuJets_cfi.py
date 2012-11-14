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
step1a = cms.EDFilter(
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
step1b = step1a.clone( src = cms.InputTag( 'tightPatMuons' ) )

step2 = countPatMuons.clone( maxNumber = 1 ) # includes the signal muon

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

step4a = cms.EDFilter(
  "PATCandViewCountFilter"
, src = cms.InputTag( 'goodPatJets' )
, minNumber = cms.uint32( 1 )
, maxNumber = cms.uint32( 999999 )
)
step4b = step4a.clone( minNumber = 2 )
step4c = step4a.clone( minNumber = 3 )
step5  = step4a.clone( minNumber = 4 )

### Electrons

step3 = countPatElectrons.clone( maxNumber = 0 )

