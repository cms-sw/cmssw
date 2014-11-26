import FWCore.ParameterSet.Config as cms

# Step 1

from PhysicsTools.PatAlgos.selectionLayer1.muonSelector_cfi import selectedPatMuons

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

from EgammaAnalysis.ElectronTools.electronRegressionEnergyProducer_cfi import eleRegressionEnergy
electronsWithRegression = eleRegressionEnergy.clone(
  inputElectronsTag = cms.InputTag( 'patElectrons' )
, rhoCollection     = cms.InputTag( 'fixedGridRhoFastjetAll' )
, vertexCollection  = cms.InputTag( 'offlinePrimaryVertices' )
)
from EgammaAnalysis.ElectronTools.calibratedPatElectrons_cfi import calibratedPatElectrons
calibratedElectrons = calibratedPatElectrons.clone(
  inputPatElectronsTag = cms.InputTag( 'electronsWithRegression' )
, inputDataset         = 'Summer12'
)

from PhysicsTools.PatAlgos.selectionLayer1.electronSelector_cfi import selectedPatElectrons

selectedElectrons = selectedPatElectrons.clone(
  src = cms.InputTag( 'calibratedElectrons' )
, cut = '' # electronCut
)

standAloneElectronVetoFilter = cms.EDFilter(
  "PATCandViewCountFilter"
, src       = cms.InputTag( 'selectedElectrons' )
, minNumber = cms.uint32( 0 )
, maxNumber = cms.uint32( 0 )
)

# Step 4

from PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi import selectedPatJets

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
