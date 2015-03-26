#
# This file contains the Top PAG reference selection work-flow for mu + jets analysis.
# as defined in
# https://twiki.cern.ch/twiki/bin/view/CMS/TWikiTopRefEventSel#mu_jets_Channel
#
# Command line arguments:
# - standard command line arguments as defined in FWCore.ParameterSet.VarParsing.VarParsing( 'standard' )
#   + 'maxEvent' (int , default: -1)
# - 'runOnMC'          (bool, default: True ): decide if run on MC or real data
# - 'runOnMiniAOD'     (bool, default: True ): decide if run on miniAOD or AOD input
# - 'useElecEAIsoCorr' (bool, default: True ): decide, if EA (rho) or Delta beta corrections are used for electron isolation
# - 'useCalibElec'     (bool, default: False): decide, if electron re-calibration using regression energies is used
# - 'addTriggerMatch'  (bool, default: True ): decide, if trigger objects are matched to signal muons
#


import sys

import FWCore.ParameterSet.Config as cms


# Command line parsing

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing ( 'standard' )
options.register( 'runOnMC'         , True , VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, 'decide, if run on MC or real data' )
options.register( 'runOnMiniAOD'    , True , VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, 'decide, if run on miniAOD or AOD input' )
options.register( 'useElecEAIsoCorr', True , VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, 'decide, if EA (rho) or Delta beta corrections are used for electron isolation is used' )
options.register( 'useCalibElec'    , False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, 'decide, if electron re-calibration using regression energies is used' )
options.register( 'addTriggerMatch' , True , VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, 'decide, if trigger objects are matched to signal muons' )
# parsing command line arguments
if( hasattr( sys, 'argv' ) ):
  if( len( sys.argv ) > 2 ):
    print 'Parsing command line arguments:'
  for args in sys.argv :
    arg = args.split(',')
    for val in arg:
      val = val.split( '=' )
      if( len( val ) == 2 ):
        print 'Setting "', val[0], '" to:', val[1]
        setattr( options, val[0], val[1] )


process = cms.Process( 'USER' )
#process.Tracer = cms.Service( "Tracer" )


### ======================================================================== ###
###                                                                          ###
###                                 Constants                                ###
###                            (user job steering)                           ###
###                                                                          ###
### ======================================================================== ###


from TopQuarkAnalysis.Configuration.patRefSel_refMuJets import *

inputFiles = []


### Selection steps
# If a step is switched off here, its results will still be available in the coreespondinng TriggerResults of this process.

# Event filter
# This parameter defines the level, at which events will be filtered for output.
# The available levels (paths) are (ordered!):
# 0a. pTrigger
# 0b. pEventCleaning
# 0c. pGoodVertex
# 1.  pSignalMuon
# 2.  pLooseMuonVeto
# 3.  pElectronVeto
# 4a. p1Jet
# 4b. p2Jets
# 4c. p3Jets
# 5.  p4Jets
# 6.  pBTags
# Each level includes the former ones, but also the corresponding stand-alone selection steps are available, adding a
# 'StandAlone' after the prefix 'p' (e.g. 'pLooseMuonVeto' --> 'pStandAloneLooseMuonVeto').
# All corresponding flags are available in the TriggerResults collection produced by this process later.
selectEvents = 'pGoodVertex'

# Step 0
#triggerSelectionData = ''
#triggerSelectionMC   = ''

# Step 1
#muonCut       = ''
#signalMuonCut = ''
#muonVertexMaxDZ = 0.5

# Step 2

# Step 3
useElecEAIsoCorr = options.useElecEAIsoCorr
useCalibElec     = options.useCalibElec
#electronGsfCut   = ''
#electronCalibCut = ''
electronCut = electronGsfCut

# Step 4

#jetCut = ''
# Step4a
#veryTightJetCut = ''
# Step4b
#tightJetCut     = ''
# Step4c
#looseJetCut     = ''

# Step 5
#veryLooseJetCut = ''

# Step 6
bTagSrc = 'selectedJets'
#bTagCut = ''
minBTags = 2

# TriggerMatching
addTriggerMatch = options.addTriggerMatch
#triggerObjectSelectionData = 'type("TriggerMuon") && ( path("%s") )'%( triggerSelectionData )
#triggerObjectSelectionMC   = 'type("TriggerMuon") && ( path("%s") )'%( triggerSelectionMC )


### Input

runOnMC      = options.runOnMC
runOnMiniAOD = options.runOnMiniAOD

# maximum number of events
maxEvents = options.maxEvents

### Conditions

# GlobalTags
globalTagMC   = 'DEFAULT'
globalTagData = 'DEFAULT'

### Output

# output file
outputFile = 'patRefSel_muJets.root'

# event frequency of Fwk report
fwkReportEvery = max( 1, int( maxEvents / 100 ) )

# switch for 'TrigReport'/'TimeReport' at job end
wantSummary = True


### ======================================================================== ###
###                                                                          ###
###                              End of constants                            ###
###                            (user job steering)                           ###
###                                                                          ###
### ======================================================================== ###


triggerSelection       = triggerSelectionData
triggerObjectSelection = triggerObjectSelectionData
if runOnMC:
  triggerSelection       = triggerSelectionMC
  triggerObjectSelection = triggerObjectSelectionMC


###
### Basic configuration
###

process.load( "TopQuarkAnalysis.Configuration.patRefSel_basics_cff" )
process.MessageLogger.cerr.FwkReport.reportEvery = fwkReportEvery
process.options.wantSummary = wantSummary
from Configuration.AlCa.GlobalTag import GlobalTag
if runOnMC:
  if globalTagMC == 'DEFAULT':
    process.GlobalTag = GlobalTag( process.GlobalTag, 'auto:run2_mc' )
  else:
    process.GlobalTag.globaltag = globalTagMC
else:
  if globalTagData == 'DEFAULT':
    process.GlobalTag = GlobalTag( process.GlobalTag, 'auto:run2_data' )
  else:
    process.GlobalTag.globaltag = globalTagData


###
### Input configuration
###

if len( inputFiles ) == 0:
  if runOnMiniAOD:
    if runOnMC:
      from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValTTbarPileUpMINIAODSIM
      inputFiles = filesRelValTTbarPileUpMINIAODSIM
    else:
      from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValSingleMuMINIAOD
      inputFiles = filesRelValSingleMuMINIAOD
  else:
    if runOnMC:
      from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValProdTTbarAODSIM
      inputFiles = filesRelValProdTTbarAODSIM
    else:
      from PhysicsTools.PatAlgos.patInputFiles_cff import filesSingleMuRECO # not available at CERN
      inputFiles = filesSingleMuRECO
process.load( "TopQuarkAnalysis.Configuration.patRefSel_inputModule_cfi" )
process.source.fileNames = inputFiles
process.maxEvents.input  = maxEvents


###
### PAT configuration
###

if not runOnMiniAOD:
  process.load( "PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff" )



###
### Output configuration
###

process.load( "TopQuarkAnalysis.Configuration.patRefSel_outputModule_cff" )
# output file name
process.out.fileName = outputFile
from TopQuarkAnalysis.Configuration.patRefSel_eventContent_cff import refMuJets_eventContent
process.out.outputCommands += refMuJets_eventContent
if runOnMiniAOD:
  from TopQuarkAnalysis.Configuration.patRefSel_eventContent_cff import miniAod_eventContent
  process.out.outputCommands += miniAod_eventContent
else:
  from TopQuarkAnalysis.Configuration.patRefSel_eventContent_cff import aod_eventContent
  process.out.outputCommands += aod_eventContent
# clear event selection
process.out.SelectEvents.SelectEvents = cms.vstring( selectEvents )


###
### Selection configuration
###

# Individual steps

# Step 0

from TopQuarkAnalysis.Configuration.patRefSel_triggerSelection_cff import triggerResults
process.triggerSelection = triggerResults.clone( triggerConditions = [ triggerSelection ] )
process.sStandAloneTrigger = cms.Sequence( process.triggerSelection
                                         )
process.pStandAloneTrigger = cms.Path( process.sStandAloneTrigger )

process.load( 'TopQuarkAnalysis.Configuration.patRefSel_eventCleaning_cff' )
process.sStandAloneEventCleaning = cms.Sequence()
if runOnMiniAOD:
  process.sStandAloneEventCleaning += process.eventCleaningMiniAOD
  if runOnMC:
    process.sStandAloneEventCleaning += process.eventCleaningMiniAODMC
  else:
    process.sStandAloneEventCleaning += process.eventCleaningMiniAODData
else:
  process.sStandAloneEventCleaning += process.eventCleaning
  if runOnMC:
    process.sStandAloneEventCleaning += process.eventCleaningMC
  else:
    process.sStandAloneEventCleaning += process.eventCleaningData
process.pStandAloneEventCleaning = cms.Path( process.sStandAloneEventCleaning )

from CommonTools.ParticleFlow.goodOfflinePrimaryVertices_cfi import goodOfflinePrimaryVertices
process.goodOfflinePrimaryVertices = goodOfflinePrimaryVertices.clone( filter = True )
if runOnMiniAOD:
  process.goodOfflinePrimaryVertices.src = 'offlineSlimmedPrimaryVertices'
process.sStandAloneGoodVertex = cms.Sequence( process.goodOfflinePrimaryVertices
                                            )
process.pStandAloneGoodVertex = cms.Path( process.sStandAloneGoodVertex )

# Step 1

from TopQuarkAnalysis.Configuration.patRefSel_refMuJets_cfi import selectedMuons, preSignalMuons, signalMuons, standAloneSignalMuonFilter
process.selectedMuons = selectedMuons.clone( cut = muonCut )
if runOnMiniAOD:
  process.selectedMuons.src = 'slimmedMuons'
process.preSignalMuons = preSignalMuons.clone( cut = signalMuonCut )
process.signalMuons = signalMuons.clone( maxDZ = muonVertexMaxDZ )
if runOnMiniAOD:
  process.signalMuons.vertexSource = 'offlineSlimmedPrimaryVertices'
process.standAloneSignalMuonFilter = standAloneSignalMuonFilter.clone()
process.sStandAloneSignalMuon = cms.Sequence( process.standAloneSignalMuonFilter )
process.pStandAloneSignalMuon = cms.Path( process.sStandAloneSignalMuon )

# Step 2

from TopQuarkAnalysis.Configuration.patRefSel_refMuJets_cfi import standAloneLooseMuonVetoFilter
process.standAloneLooseMuonVetoFilter = standAloneLooseMuonVetoFilter.clone()
process.sStandAloneLooseMuonVeto = cms.Sequence( process.standAloneLooseMuonVetoFilter )
process.pStandAloneLooseMuonVeto = cms.Path( process.sStandAloneLooseMuonVeto )

# Step 3

if not runOnMiniAOD:
  from PhysicsTools.SelectorUtils.tools.vid_id_tools import switchOnVIDElectronIdProducer, setupAllVIDIdsInModule, setupVIDElectronSelection
  electron_ids = [ 'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_CSA14_50ns_V1_cff'
                 , 'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_CSA14_PU20bx25_V0_cff'
                  ]
  switchOnVIDElectronIdProducer( process )
  process.electronIDValueMapProducer.ebReducedRecHitCollection = cms.InputTag( 'reducedEcalRecHitsEB' )
  process.electronIDValueMapProducer.eeReducedRecHitCollection = cms.InputTag( 'reducedEcalRecHitsEE' )
  process.electronIDValueMapProducer.esReducedRecHitCollection = cms.InputTag( 'reducedEcalRecHitsES' )
  for idmod in electron_ids:
    setupAllVIDIdsInModule( process, idmod, setupVIDElectronSelection )

if useElecEAIsoCorr:
  from EgammaAnalysis.ElectronTools.electronIsolatorFromEffectiveArea_cfi import elPFIsoValueEA03
  if runOnMiniAOD:
    process.patElPFIsoValueEA03 = elPFIsoValueEA03.clone( gsfElectrons = ''
                                                        , pfElectrons  = ''
                                                        , patElectrons = cms.InputTag( 'slimmedElectrons' )
                                                        , rhoIso       = cms.InputTag( 'fixedGridRhoFastjetAll' )
                                                        )
    from EgammaAnalysis.ElectronTools.patElectronEAIsoCorrectionProducer_cfi import patElectronEAIso03CorrectionProducer
    process.electronsWithEA03Iso = patElectronEAIso03CorrectionProducer.clone( patElectrons  = 'slimmedElectrons'
                                                                             , eaIsolator    = 'patElPFIsoValueEA03'
                                                                             )
  else:
    process.elPFIsoValueEA03 = elPFIsoValueEA03.clone( gsfElectrons = 'gedGsfElectrons'
                                                     , pfElectrons  = ''
                                                     , rhoIso       = cms.InputTag( 'fixedGridRhoFastjetAll' )
                                                     )
    process.patElectrons.isolationValues.user = cms.VInputTag( cms.InputTag( 'elPFIsoValueEA03' ) )
else:
  electronGsfCut.replace( '-1.0*userIsolation("User1Iso")', '-0.5*puChargedHadronIso' )
  electronCalibCut.replace( '-1.0*userIsolation("User1Iso")', '-0.5*puChargedHadronIso' )
  electronCut.replace( '-1.0*userIsolation("User1Iso")', '-0.5*puChargedHadronIso' )

if useCalibElec:
  from TopQuarkAnalysis.Configuration.patRefSel_refMuJets_cfi import electronsWithRegression, calibratedElectrons
  process.electronsWithRegression = electronsWithRegression.clone()
  if runOnMiniAOD:
    if useElecEAIsoCorr:
      process.electronsWithRegression.inputElectronsTag = 'electronsWithEA03Iso'
    else:
      process.electronsWithRegression.inputElectronsTag = 'slimmedElectrons'
    process.electronsWithRegression.vertexCollection  = 'offlineSlimmedPrimaryVertices'
  process.calibratedElectrons = calibratedElectrons.clone( isMC = runOnMC )
  if runOnMC:
    process.calibratedElectrons.inputDataset = 'Summer12_LegacyPaper' # FIXME: Update as soon as available
  else:
    process.calibratedElectrons.inputDataset = '22Jan2013ReReco' # FIXME: Update as soon as available
  process.RandomNumberGeneratorService = cms.Service( "RandomNumberGeneratorService"
                                                    , calibratedElectrons = cms.PSet( initialSeed = cms.untracked.uint32( 1 )
                                                                                    , engineName  = cms.untracked.string('TRandom3')
                                                                                    )
                                                    )
  electronCut = electronCalibCut

from TopQuarkAnalysis.Configuration.patRefSel_refMuJets_cfi import selectedElectrons, standAloneElectronVetoFilter
process.selectedElectrons = selectedElectrons.clone( cut = electronCut )
if useCalibElec:
  process.selectedElectrons.src = 'calibratedElectrons'
elif useElecEAIsoCorr and runOnMiniAOD:
  process.selectedElectrons.src = 'electronsWithEA03Iso'
elif runOnMiniAOD:
  process.selectedElectrons.src = 'slimmedElectrons'

process.standAloneElectronVetoFilter = standAloneElectronVetoFilter.clone()
process.sStandAloneElectronVeto = cms.Sequence( process.standAloneElectronVetoFilter )
process.pStandAloneElectronVeto = cms.Path( process.sStandAloneElectronVeto )

# Step 4

from TopQuarkAnalysis.Configuration.patRefSel_refMuJets_cfi import selectedJets
process.selectedJets = selectedJets.clone( cut = jetCut )
if runOnMiniAOD:
  process.selectedJets.src = 'slimmedJets'

from TopQuarkAnalysis.Configuration.patRefSel_refMuJets_cfi import signalVeryTightJets, standAloneSignalVeryTightJetsFilter
process.signalVeryTightJets = signalVeryTightJets.clone( cut = veryTightJetCut )
process.standAloneSignalVeryTightJetsFilter = standAloneSignalVeryTightJetsFilter.clone()
process.sStandAlone1Jet = cms.Sequence( process.standAloneSignalVeryTightJetsFilter )
process.pStandAlone1Jet = cms.Path( process.sStandAlone1Jet )

from TopQuarkAnalysis.Configuration.patRefSel_refMuJets_cfi import signalTightJets, standAloneSignalTightJetsFilter
process.signalTightJets = signalTightJets.clone( cut = tightJetCut )
process.standAloneSignalTightJetsFilter = standAloneSignalTightJetsFilter.clone()
process.sStandAlone2Jets = cms.Sequence( process.standAloneSignalTightJetsFilter )
process.pStandAlone2Jets = cms.Path( process.sStandAlone2Jets )

from TopQuarkAnalysis.Configuration.patRefSel_refMuJets_cfi import signalLooseJets, standAloneSignalLooseJetsFilter
process.signalLooseJets = signalLooseJets.clone( cut = looseJetCut )
process.standAloneSignalLooseJetsFilter = standAloneSignalLooseJetsFilter.clone()
process.sStandAlone3Jets = cms.Sequence( process.standAloneSignalLooseJetsFilter )
process.pStandAlone3Jets = cms.Path( process.sStandAlone3Jets )

# Step 5

from TopQuarkAnalysis.Configuration.patRefSel_refMuJets_cfi import signalVeryLooseJets, standAloneSignalVeryLooseJetsFilter
process.signalVeryLooseJets = signalVeryLooseJets.clone( cut = veryLooseJetCut )
process.standAloneSignalVeryLooseJetsFilter = standAloneSignalVeryLooseJetsFilter.clone()
process.sStandAlone4Jets = cms.Sequence( process.standAloneSignalVeryLooseJetsFilter )
process.pStandAlone4Jets = cms.Path( process.sStandAlone4Jets )

# Step 6

from TopQuarkAnalysis.Configuration.patRefSel_refMuJets_cfi import selectedBTagJets, standAloneSignalBTagsFilter
process.selectedBTagJets = selectedBTagJets.clone( src = bTagSrc
                                                 , cut = bTagCut
                                                 )
process.standAloneSignalBTagsFilter = standAloneSignalBTagsFilter.clone( minNumber = minBTags )
process.sStandAloneBTags = cms.Sequence( process.standAloneSignalBTagsFilter )
process.pStandAloneBTags = cms.Path( process.sStandAloneBTags )

# Consecutive steps

process.sTrigger       = cms.Sequence( process.sStandAloneTrigger
                                     )
process.sEventCleaning = cms.Sequence( process.sTrigger
                                     + process.sStandAloneEventCleaning
                                     )
process.sGoodVertex    = cms.Sequence( process.sEventCleaning
                                     + process.sStandAloneGoodVertex
                                     )
process.sSignalMuon    = cms.Sequence( process.sGoodVertex
                                     + process.sStandAloneSignalMuon
                                     )
process.sLooseMuonVeto = cms.Sequence( process.sSignalMuon
                                     + process.sStandAloneLooseMuonVeto
                                     )
process.sElectronVeto  = cms.Sequence( process.sLooseMuonVeto
                                     + process.sStandAloneElectronVeto
                                     )
process.s1Jet          = cms.Sequence( process.sElectronVeto
                                     + process.sStandAlone1Jet
                                     )
process.s2Jets         = cms.Sequence( process.s1Jet
                                     + process.sStandAlone2Jets
                                     )
process.s3Jets         = cms.Sequence( process.s2Jets
                                     + process.sStandAlone2Jets
                                     )
process.s4Jets         = cms.Sequence( process.s3Jets
                                     + process.sStandAlone4Jets
                                     )
process.sBTags         = cms.Sequence( process.s4Jets
                                     + process.sStandAloneBTags
                                     )

process.pTrigger       = cms.Path( process.sTrigger )
process.pEventCleaning = cms.Path( process.sEventCleaning )
process.pGoodVertex    = cms.Path( process.sGoodVertex )
process.pSignalMuon    = cms.Path( process.sSignalMuon )
process.pLooseMuonVeto = cms.Path( process.sLooseMuonVeto )
process.pElectronVeto  = cms.Path( process.sElectronVeto )
process.p1Jet          = cms.Path( process.s1Jet )
process.p2Jets         = cms.Path( process.s2Jets )
process.p3Jets         = cms.Path( process.s2Jets )
process.p4Jets         = cms.Path( process.s4Jets )
process.pBTags         = cms.Path( process.sBTags )

# Trigger matching

if addTriggerMatch:
  from TopQuarkAnalysis.Configuration.patRefSel_triggerMatching_cff import muonTriggerMatch
  process.muonTriggerMatch = muonTriggerMatch.clone( matchedCuts = triggerObjectSelection )
  if not runOnMiniAOD:
    from PhysicsTools.PatAlgos.tools.trigTools import switchOnTriggerMatchEmbedding
    switchOnTriggerMatchEmbedding( process, triggerMatchers = [ 'muonTriggerMatch' ] )
  else:
    from TopQuarkAnalysis.Configuration.patRefSel_triggerMatching_cff import unpackedPatTrigger
    process.selectedTriggerUnpacked = unpackedPatTrigger.clone()
    process.muonTriggerMatch.matched = 'selectedTriggerUnpacked'
    from TopQuarkAnalysis.Configuration.patRefSel_triggerMatching_cff import signalMuonsTriggerMatch
    process.signalMuonsTriggerMatch = signalMuonsTriggerMatch.clone()
    process.out.outputCommands += [ 'drop *_signalMuons_*_*'
                                  , 'keep *_signalMuonsTriggerMatch_*_*'
                                  ]

