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
tightPatMuons = cleanPatMuons.clone(
  src           = cms.InputTag( 'loosePatMuons' )
, preselection  = '' # tightMuonCut
, checkOverlaps = cms.PSet()
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

step3a = cms.EDFilter(
  "PATCandViewCountFilter"
, src = cms.InputTag( 'goodPatJets' )
, minNumber = cms.uint32( 6 )
, maxNumber = cms.uint32( 999999 )
)
step3b_1 = step3a.clone( src = 'goodPatJets60', minNumber = 4 )
step3b_2 = step3a.clone( src = 'goodPatJets50', minNumber = 5 )
step3b_3 = step3a.clone( src = 'goodPatJets'  , minNumber = 6 )

step3b = cms.Sequence(step3b_1 *
                      step3b_2 *
                      step3b_3
                      )
