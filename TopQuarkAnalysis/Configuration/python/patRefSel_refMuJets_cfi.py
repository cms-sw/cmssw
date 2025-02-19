import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.patSequences_cff import *
from TopQuarkAnalysis.Configuration.patRefSel_common_cfi import *

### Muons

intermediatePatMuons = selectedPatMuons.clone(
  src = cms.InputTag( 'selectedPatMuons' )
, cut = '' # signalMuonCut
)
goodPatMuons = cms.EDProducer(
  "MuonSelectorVertex"
, muonSource   = cms.InputTag( 'intermediatePatMuons' )
, vertexSource = cms.InputTag( 'offlinePrimaryVertices' )
, maxDZ        = cms.double( 999. ) # muonVertexMaxDZ
)

step1 = cms.EDFilter(
  "PATCandViewCountFilter"
, src = cms.InputTag( 'goodPatMuons' )
, minNumber = cms.uint32( 1 )
, maxNumber = cms.uint32( 1 )
)

step2 = countPatMuons.clone(
  maxNumber = 1 # includes the signal muon
)

### Jets

veryLoosePatJets = selectedPatJets.clone(
  src = 'selectedPatJets'
, cut = '' # veryLooseJetCut
)
loosePatJets = selectedPatJets.clone(
  src = 'veryLoosePatJets'
, cut = '' # looseJetCut
)
tightPatJets = selectedPatJets.clone(
  src = 'loosePatJets'
, cut = '' # tightJetCut
)

step4a = cms.EDFilter(
  "PATCandViewCountFilter"
, src = cms.InputTag( 'tightPatJets' )
, minNumber = cms.uint32( 1 )
, maxNumber = cms.uint32( 999999 )
)
step4b = step4a.clone(
  minNumber = 2
)
step4cTight = step4a.clone(
  minNumber = 3
)
step4cLoose = step4a.clone(
  src       = 'loosePatJets'
, minNumber = 3
)
step5  = step4a.clone(
  src       = 'veryLoosePatJets'
, minNumber = 4
)

### Electrons

step3 = countPatElectrons.clone( maxNumber = 0 )

