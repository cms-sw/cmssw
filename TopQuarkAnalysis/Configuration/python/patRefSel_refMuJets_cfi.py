import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.selectionLayer1.muonSelector_cfi import selectedPatMuons
from EgammaAnalysis.ElectronTools.electronRegressionEnergyProducer_cfi import eleRegressionEnergy
from EgammaAnalysis.ElectronTools.calibratedPatElectrons_cfi import calibratedPatElectrons
from PhysicsTools.PatAlgos.selectionLayer1.electronSelector_cfi import selectedPatElectrons
from PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi import selectedPatJets

# Step 1

selectedMuons = selectedPatMuons.clone(
  src = cms.InputTag( 'patMuons' )
, cut = '' # muonCut
)

preSignalMuons = selectedPatMuons.clone(
  src = cms.InputTag( 'selectedMuons' )
, cut = '' # signalMuonCut
)

signalMuons = cms.EDProducer(
  "MuonSelectorVertex"
, muonSource   = cms.InputTag( 'preSignalMuons' )
, vertexSource = cms.InputTag( 'offlinePrimaryVertices' )
, maxDZ        = cms.double( 999. ) # muonVertexMaxDZ
)

standAloneSignalMuonFilter = cms.EDFilter(
  "PATCandViewCountFilter"
, src       = cms.InputTag( 'signalMuons' )
, minNumber = cms.uint32( 1 )
, maxNumber = cms.uint32( 1 )
)

# Step 2

standAloneLooseMuonVetoFilter = cms.EDFilter(
  "PATCandViewCountFilter"
, src       = cms.InputTag( 'selectedMuons' )
, minNumber = cms.uint32( 0 )
, maxNumber = cms.uint32( 1 )
)

# Step 3

electronsWithRegression = eleRegressionEnergy.clone(
  inputElectronsTag = cms.InputTag( 'patElectrons' )
, rhoCollection     = cms.InputTag( 'fixedGridRhoFastjetAll' )
, vertexCollection  = cms.InputTag( 'offlinePrimaryVertices' )
)
calibratedElectrons = calibratedPatElectrons.clone(
  inputPatElectronsTag = cms.InputTag( 'electronsWithRegression' )
, inputDataset         = 'Summer12'
)

selectedElectrons = selectedPatElectrons.clone(
  src = cms.InputTag( 'patElectrons' )
, cut = '' # electronCut
)

standAloneElectronVetoFilter = cms.EDFilter(
  "PATCandViewCountFilter"
, src       = cms.InputTag( 'selectedElectrons' )
, minNumber = cms.uint32( 0 )
, maxNumber = cms.uint32( 0 )
)

# Step 4

selectedJets = selectedPatJets.clone(
  src = cms.InputTag( 'patJets' )
, cut = '' # jetCut
)

signalVeryTightJets = selectedPatJets.clone(
  src = cms.InputTag( 'selectedJets' )
, cut = '' # veryTightJetCut
)

standAloneSignalVeryTightJetsFilter = cms.EDFilter(
  "PATCandViewCountFilter"
, src       = cms.InputTag( 'signalVeryTightJets' )
, minNumber = cms.uint32( 1 )
, maxNumber = cms.uint32( 99 )
)

signalTightJets = selectedPatJets.clone(
  src = cms.InputTag( 'selectedJets' )
, cut = '' # tightJetCut
)

standAloneSignalTightJetsFilter = cms.EDFilter(
  "PATCandViewCountFilter"
, src       = cms.InputTag( 'signalTightJets' )
, minNumber = cms.uint32( 2 )
, maxNumber = cms.uint32( 99 )
)

signalLooseJets = selectedPatJets.clone(
  src = cms.InputTag( 'selectedJets' )
, cut = '' # looseJetCut
)

standAloneSignalLooseJetsFilter = cms.EDFilter(
  "PATCandViewCountFilter"
, src       = cms.InputTag( 'signalLooseJets' )
, minNumber = cms.uint32( 3 )
, maxNumber = cms.uint32( 99 )
)

# Step 5

signalVeryLooseJets = selectedPatJets.clone(
  src = cms.InputTag( 'selectedJets' )
, cut = '' # veryLooseJetCut
)

standAloneSignalVeryLooseJetsFilter = cms.EDFilter(
  "PATCandViewCountFilter"
, src       = cms.InputTag( 'signalVeryLooseJets' )
, minNumber = cms.uint32( 4 )
, maxNumber = cms.uint32( 99 )
)

# Step 6

selectedBTagJets = selectedPatJets.clone(
  src = cms.InputTag( 'selectedJets' )
, cut = '' # btagCut
)

standAloneSignalBTagsFilter = cms.EDFilter(
  "PATCandViewCountFilter"
, src       = cms.InputTag( 'selectedBTagJets' )
, minNumber = cms.uint32( 2 )
, maxNumber = cms.uint32( 99 )
)
