#
# This file contains the Top PAG reference selection work-flow for mu + jets analysis.
# as defined in
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopLeptonPlusJetsRefSel_mu#Selection_Version_SelV4_valid_fr
#

import sys

import FWCore.ParameterSet.Config as cms

process = cms.Process( 'PAT' )


### ======================================================================== ###
###                                                                          ###
###                                 Constants                                ###
###                            (user job steering)                           ###
###                                                                          ###
### ======================================================================== ###


### Data or MC?
runOnMC = True

### Standard and PF reconstruction
useStandardPAT = True
runPF2PAT      = True

### Switch on/off selection steps

# Step 1
useTrigger      = True
# Step 2
useGoodVertex   = True
# Step 3b
useLooseMuon    = True
# Step 3a
useTightMuon    = True
# Step 4
useMuonVeto     = True
# Step 5
useElectronVeto = True
# Step 6a
use1Jet         = True
# Step 6b
use2Jets        = False
# Step 6c
use3Jets        = False
# Step 7
use4Jets        = False

addTriggerMatching = True

### Reference selection

from TopQuarkAnalysis.Configuration.patRefSel_refMuJets import *
#muonsUsePV             = False
#muonEmbedTrack         = True
#muonCutPF                = ''
#looseMuonCutPF           = ''
#tightMuonCutPF           = ''
#muonJetsDR             = 0.3
#jetCutPF               = ''
#jetMuonsDRPF           = 0.1
#electronCutPF            = ''

# Trigger selection according to run range resp. MC sample:
# lower range limits for data available as suffix;
# available are: 000000, 147196, 160404, 163270 (default)
# sample name for MC available as suffix;
# available are: Summer11 (default)
#triggerSelectionData       = triggerSelection_163270
#triggerObjectSelectionData = triggerObjectSelection_163270
#triggerSelectionMC       = triggerSelection_Summer11
#triggerObjectSelectionMC = triggerObjectSelection_Summer11

### Particle flow
### takes effect only, if 'runPF2PAT' = True

postfix = 'PF' # needs to be a non-empty string, if 'useStandardPAT' = True

# subtract charged hadronic pile-up particles (from wrong PVs)
# effects also JECs
usePFnoPU = True # before any top projection

# other switches for PF top projections (default: all 'True')
useNoMuon     = True # before electron top projection
useNoElectron = True # before jet top projection
useNoJet      = True # before tau top projection
useNoTau      = True # before MET top projection

### JEC levels

# levels to be accessible from the jets
# jets are corrected to L3Absolute (MC), L2L3Residual (data) automatically, if enabled here
# and remain uncorrected, if none of these levels is enabled here
useL1FastJet    = True  # needs useL1Offset being off, error otherwise
useL1Offset     = False # needs useL1FastJet being off, error otherwise
useL2Relative   = True
useL3Absolute   = True
# useL2L3Residual = True  # takes effect only on data; currently disabled for CMSSW_4_2_X GlobalTags!
useL5Flavor     = True
useL7Parton     = True

### Input

# list of input files
useRelVals = True # if 'False', "inputFiles" is used
inputFiles = [ '/store/data/Run2011A/MuHad/AOD/PromptReco-v4/000/165/129/42DDEE9E-7B81-E011-AF29-003048F024DC.root'
             , '/store/data/Run2011A/MuHad/AOD/PromptReco-v4/000/165/103/8AF0C4CD-EE80-E011-96B1-003048F1BF68.root'
             , '/store/data/Run2011A/MuHad/AOD/PromptReco-v4/000/165/102/1298E7B3-CF80-E011-86E9-001617E30F4C.root'
             , '/store/data/Run2011A/MuHad/AOD/PromptReco-v4/000/165/099/16D66DF0-D07F-E011-A922-003048F118C4.root'
             , '/store/data/Run2011A/MuHad/AOD/PromptReco-v4/000/165/098/66526A73-DA7F-E011-84B4-003048F1BF68.root'
             , '/store/data/Run2011A/MuHad/AOD/PromptReco-v4/000/165/093/2E545F72-C27F-E011-8CC1-003048D2C1C4.root'
             , '/store/data/Run2011A/MuHad/AOD/PromptReco-v4/000/165/088/DED0B748-E47F-E011-9C6E-0030487C6A66.root'
             , '/store/data/Run2011A/MuHad/AOD/PromptReco-v4/000/165/071/283C53A0-C37F-E011-A06F-0030487CD6D8.root'
             ] # overwritten, if "useRelVals" is 'True'

# maximum number of events
maxInputEvents = -1 # reduce for testing

### Conditions

# GlobalTags (w/o suffix '::All')
globalTagData = 'GR_R_42_V12' # default for CMSSW_4_2_3 RelVals: 'GR_R_42_V12'
globalTagMC   = 'START42_V12' # default for CMSSW_4_2_3 RelVals: 'START42_V12'

### Output

# output file
outputFile = 'patRefSel_muJets.root'

# event frequency of Fwk report
fwkReportEvery = 1000

# switch for 'TrigReport'/'TimeReport' at job end
wantSummary = True


###                              End of constants                            ###
###                                                                          ###
### ======================================================================== ###


###
### Basic configuration
###

process.load( "TopQuarkAnalysis.Configuration.patRefSel_basics_cff" )
process.MessageLogger.cerr.FwkReport.reportEvery = fwkReportEvery
process.options.wantSummary = wantSummary
if runOnMC:
  process.GlobalTag.globaltag = globalTagMC   + '::All'
else:
  process.GlobalTag.globaltag = globalTagData + '::All'


###
### Input configuration
###

process.load( "TopQuarkAnalysis.Configuration.patRefSel_inputModule_cfi" )
if useRelVals:
  from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles
  if runOnMC:
    inputFiles = pickRelValInputFiles( cmsswVersion  = 'CMSSW_4_2_3'
                                     , relVal        = 'RelValTTbar'
                                     , globalTag     = globalTagMC
                                     , numberOfFiles = -1 # "-1" means "all"
                                     )
  else:
    inputFiles = pickRelValInputFiles( cmsswVersion  = 'CMSSW_4_2_3'
                                     , relVal        = 'Mu'
                                     , dataTier      = 'RECO'
                                     #, globalTag     = globalTagData + '_RelVal_mu2010B'
                                     , globalTag     = globalTagData + '_mu2010B' # wrong naming scheme in CMSSW_4_2_3
                                     , numberOfFiles = -1 # "-1" means "all"
                                     )
process.source.fileNames = inputFiles
process.maxEvents.input  = maxInputEvents


###
### Output configuration
###

process.load( "TopQuarkAnalysis.Configuration.patRefSel_outputModule_cff" )
# output file name
process.out.fileName = outputFile
# event content
from PhysicsTools.PatAlgos.patEventContent_cff import patEventContent
process.out.outputCommands += patEventContent
# clear event selection
process.out.SelectEvents.SelectEvents = []


###
### Cleaning and trigger selection configuration
###

### Event cleaning
process.load( 'TopQuarkAnalysis.Configuration.patRefSel_eventCleaning_cff' )

### Trigger selection
if runOnMC:
  triggerSelection = triggerSelectionMC
else:
  triggerSelection = triggerSelectionData
from TopQuarkAnalysis.Configuration.patRefSel_triggerSelection_cff import triggerResults
process.step1 = triggerResults.clone(
  triggerConditions = [ triggerSelection ]
)

### Good vertex selection
process.load( "TopQuarkAnalysis.Configuration.patRefSel_goodVertex_cfi" )


###
### PAT/PF2PAT configuration
###

if useStandardPAT and runPF2PAT:
  if postfix == '':
    sys.exit( 'ERROR: running standard PAT and PF2PAT in parallel requires a defined "postfix" for PF2PAT' )
if not useStandardPAT and not runPF2PAT:
  sys.exit( 'ERROR: standard PAT and PF2PAT are both switched off' )

process.load( "PhysicsTools.PatAlgos.patSequences_cff" )
from PhysicsTools.PatAlgos.tools.coreTools import *

### Check JECs

# JEC set
jecSet   = jecSetBase + 'Calo'
jecSetPF = jecSetBase + 'PF'
if usePFnoPU:
  jecSetPF += 'chs'

# JEC levels
if useL1FastJet and useL1Offset:
  sys.exit( 'ERROR: switch off either "L1FastJet" or "L1Offset"' )
jecLevels = []
if useL1FastJet:
  jecLevels.append( 'L1FastJet' )
if useL1Offset:
  jecLevels.append( 'L1Offset' )
if useL2Relative:
  jecLevels.append( 'L2Relative' )
if useL3Absolute:
  jecLevels.append( 'L3Absolute' )
# if useL2L3Residual and not runOnMC:
#   jecLevelsPF.append( 'L2L3Residual' )
if useL5Flavor:
  jecLevels.append( 'L5Flavor' )
if useL7Parton:
  jecLevels.append( 'L7Parton' )

### Switch configuration

if runPF2PAT:
  from PhysicsTools.PatAlgos.tools.pfTools import usePF2PAT
  usePF2PAT( process
           , runPF2PAT      = runPF2PAT
           , runOnMC        = runOnMC
           , jetAlgo        = jetAlgo
           , postfix        = postfix
           , jetCorrections = ( jecSetPF
                              , jecLevels
                              )
           )
  applyPostfix( process, 'pfNoPileUp'  , postfix ).enable = usePFnoPU
  applyPostfix( process, 'pfNoMuon'    , postfix ).enable = useNoMuon
  applyPostfix( process, 'pfNoElectron', postfix ).enable = useNoElectron
  applyPostfix( process, 'pfNoJet'     , postfix ).enable = useNoJet
  applyPostfix( process, 'pfNoTau'     , postfix ).enable = useNoTau
  if useL1FastJet:
    applyPostfix( process, 'pfPileUp', postfix ).Vertices            = cms.InputTag( 'goodOfflinePrimaryVertices' )
    applyPostfix( process, 'pfPileUp', postfix ).checkClosestZVertex = False
    applyPostfix( process, 'pfJets', postfix ).doAreaFastjet = True
    applyPostfix( process, 'pfJets', postfix ).doRhoFastjet  = False

# remove MC matching, object cleaning, photons and taus
if useStandardPAT:
  if not runOnMC:
    runOnData( process )
  removeSpecificPATObjects( process
                          , names = [ 'Photons', 'Taus' ]
                          ) # includes 'removeCleaning'
if runPF2PAT:
  if not runOnMC:
    runOnData( process
             , names = [ 'PFAll' ]
             , postfix = postfix
             )
  removeSpecificPATObjects( process
                          , names = [ 'Photons', 'Taus' ]
                          , postfix = postfix
                          ) # includes 'removeCleaning'

# JetCorrFactorsProducer configuration has to be fixed _after_ any call to 'removeCleaning()':
if useStandardPAT:
  process.patJetCorrFactors.payload = jecSet
  process.patJetCorrFactors.levels  = jecLevels
# additional event content has to be (re-)added _after_ the call to 'removeCleaning()':
process.out.outputCommands += [ 'keep edmTriggerResults_*_*_*'
                              , 'keep *_hltTriggerSummaryAOD_*_*'
                              # vertices and beam spot
                              , 'keep *_offlineBeamSpot_*_*'
                              , 'keep *_offlinePrimaryVertices*_*_*'
                              , 'keep *_goodOfflinePrimaryVertices*_*_*'
                              ]
if runOnMC:
  process.out.outputCommands += [ 'keep GenEventInfoProduct_*_*_*'
                                , 'keep recoGenParticles_*_*_*'
                                ]


###
### Additional configuration
###

from TopQuarkAnalysis.Configuration.patRefSel_refMuJets_cfi import *

if useStandardPAT:

  process.intermediatePatMuons = intermediatePatMuons.clone()
  process.loosePatMuons        = loosePatMuons.clone()
  process.step3b               = step3b.clone()
  process.tightPatMuons        = tightPatMuons.clone()
  process.step3a               = step3a.clone()
  process.step4                = step4.clone()

  if useL1FastJet:
    process.kt6PFJets = kt6PFJets.clone( doRhoFastjet = True )
    process.patDefaultSequence.replace( process.patJetCorrFactors
                                      , process.kt6PFJets * process.patJetCorrFactors
                                      )
    process.out.outputCommands.append( 'keep double_*_*_' + process.name_() )

  process.goodPatJets = goodPatJets.clone()
  process.step6a      = step6a.clone()
  process.step6b      = step6b.clone()
  process.step6c      = step6c.clone()
  process.step7       = step7.clone()

  process.step5 = step5.clone()

if runPF2PAT:

  ### Muons

  intermediatePatMuonsPF = intermediatePatMuons.clone( src = cms.InputTag( 'selectedPatMuons' + postfix ) )
  setattr( process, 'intermediatePatMuons' + postfix, intermediatePatMuonsPF )

  loosePatMuonsPF = loosePatMuons.clone( src = cms.InputTag( 'intermediatePatMuons' + postfix ) )
  setattr( process, 'loosePatMuons' + postfix, loosePatMuonsPF )
  getattr( process, 'loosePatMuons' + postfix ).checkOverlaps.jets.src = cms.InputTag( 'goodPatJets' + postfix )

  step3bPF = step3b.clone( src = cms.InputTag( 'loosePatMuons' + postfix ) )
  setattr( process, 'step3b' + postfix, step3bPF )

  tightPatMuonsPF = tightPatMuons.clone( src = cms.InputTag( 'loosePatMuons' + postfix ) )
  setattr( process, 'tightPatMuons' + postfix, tightPatMuonsPF )

  step3aPF = step3a.clone( src = cms.InputTag( 'tightPatMuons' + postfix ) )
  setattr( process, 'step3a' + postfix, step3aPF )

  step4PF = step4.clone( src = cms.InputTag( 'selectedPatMuons' + postfix ) )
  setattr( process, 'step4' + postfix, step4PF )

  ### Jets

  applyPostfix( process, 'patJetCorrFactors', postfix ).primaryVertices = cms.InputTag( 'goodOfflinePrimaryVertices' )
  if useL1FastJet:
    if usePFnoPU:
      kt6PFJetsPFChs = kt6PFJetsChs.clone( src = cms.InputTag( 'pfNoElectron' + postfix ) )
      setattr( process, 'kt6PFJetsChs' + postfix, kt6PFJetsPFChs )
      getattr( process, 'patPF2PATSequence' + postfix).replace( getattr( process, 'patJetCorrFactors' + postfix )
                                                              , getattr( process, 'kt6PFJetsChs' + postfix ) * getattr( process, 'patJetCorrFactors' + postfix )
                                                              )
      applyPostfix( process, 'patJetCorrFactors', postfix ).rho = cms.InputTag( 'kt6PFJetsChs' + postfix, 'rho' )
    else:
      kt6PFJetsPF = kt6PFJets.clone( doRhoFastjet = True )
      setattr( process, 'kt6PFJets' + postfix, kt6PFJetsPF )
      getattr( process, 'patPF2PATSequence' + postfix).replace( getattr( process, 'patJetCorrFactors' + postfix )
                                                              , getattr( process, 'kt6PFJets' + postfix ) * getattr( process, 'patJetCorrFactors' + postfix )
                                                              )
      applyPostfix( process, 'patJetCorrFactors', postfix ).rho = cms.InputTag( 'kt6PFJets' + postfix, 'rho' )
    process.out.outputCommands.append( 'keep double_*' + postfix + '*_*_' + process.name_() )

  goodPatJetsPF = goodPatJets.clone( src = cms.InputTag( 'selectedPatJets' + postfix ) )
  setattr( process, 'goodPatJets' + postfix, goodPatJetsPF )
  getattr( process, 'goodPatJets' + postfix ).checkOverlaps.muons.src = cms.InputTag( 'intermediatePatMuons' + postfix )

  step6aPF = step6a.clone( src = cms.InputTag( 'goodPatJets' + postfix ) )
  setattr( process, 'step6a' + postfix, step6aPF )
  step6bPF = step6b.clone( src = cms.InputTag( 'goodPatJets' + postfix ) )
  setattr( process, 'step6b' + postfix, step6bPF )
  step6cPF = step6c.clone( src = cms.InputTag( 'goodPatJets' + postfix ) )
  setattr( process, 'step6c' + postfix, step6cPF )
  step7PF = step7.clone( src = cms.InputTag( 'goodPatJets' + postfix ) )
  setattr( process, 'step7'  + postfix, step7PF  )

  ### Electrons

  step5PF = step5.clone( src = cms.InputTag( 'selectedPatElectrons' + postfix ) )
  setattr( process, 'step5' + postfix, step5PF )

process.out.outputCommands.append( 'keep *_intermediatePatMuons*_*_*' )
process.out.outputCommands.append( 'keep *_loosePatMuons*_*_*' )
process.out.outputCommands.append( 'keep *_tightPatMuons*_*_*' )
process.out.outputCommands.append( 'keep *_goodPatJets*_*_*' )


###
### Selection configuration
###

if useStandardPAT:

  ### Muons

  process.patMuons.usePV      = muonsUsePV
  process.patMuons.embedTrack = muonEmbedTrack

  process.selectedPatMuons.cut = muonCut

  process.intermediatePatMuons.preselection = looseMuonCut

  process.loosePatMuons.checkOverlaps.jets.deltaR = muonJetsDR

  process.tightPatMuons.preselection = tightMuonCut

  ### Jets

  process.goodPatJets.preselection = jetCut

  ### Electrons

  process.selectedPatElectrons.cut = electronCut

if runPF2PAT:

  applyPostfix( process, 'patMuons', postfix ).usePV      = muonsUsePV
  applyPostfix( process, 'patMuons', postfix ).embedTrack = muonEmbedTrack

  applyPostfix( process, 'selectedPatMuons', postfix ).cut = muonCutPF

  getattr( process, 'intermediatePatMuons' + postfix ).preselection = looseMuonCutPF

  getattr( process, 'loosePatMuons' + postfix ).preselection              = looseMuonCutPF
  getattr( process, 'loosePatMuons' + postfix ).checkOverlaps.jets.deltaR = muonJetsDR

  getattr( process, 'tightPatMuons' + postfix ).preselection = tightMuonCutPF

  ### Jets

  getattr( process, 'goodPatJets' + postfix ).preselection               = jetCutPF
  getattr( process, 'goodPatJets' + postfix ).checkOverlaps.muons.deltaR = jetMuonsDRPF

  ### Electrons

  applyPostfix( process, 'selectedPatElectrons', postfix ).cut = electronCutPF


###
### Scheduling
###

# The additional sequence

if useStandardPAT:
  process.patAddOnSequence = cms.Sequence(
    process.intermediatePatMuons
  * process.goodPatJets
  * process.loosePatMuons
  * process.tightPatMuons
  )
if runPF2PAT:
  patAddOnSequence = cms.Sequence(
    getattr( process, 'intermediatePatMuons' + postfix )
  * getattr( process, 'goodPatJets' + postfix )
  * getattr( process, 'loosePatMuons' + postfix )
  * getattr( process, 'tightPatMuons' + postfix )
  )
  setattr( process, 'patAddOnSequence' + postfix, patAddOnSequence )

# The paths
if useStandardPAT:
  process.p = cms.Path()
  if not runOnMC:
    process.p += process.eventCleaning
  if useTrigger:
    process.p += process.step1
  process.p += process.goodOfflinePrimaryVertices
  if useGoodVertex:
    process.p += process.step2
  process.p += process.patDefaultSequence
  process.p += process.patAddOnSequence
  if useLooseMuon:
    process.p += process.step3b
  if useTightMuon:
    process.p += process.step3a
  if useMuonVeto:
    process.p += process.step4
  if useElectronVeto:
    process.p += process.step5
  if use1Jet:
    process.p += process.step6a
  if use2Jets:
    process.p += process.step6b
  if use3Jets:
    process.p += process.step6c
  if use4Jets:
    process.p += process.step7
  process.out.SelectEvents.SelectEvents.append( 'p' )

if runPF2PAT:
  pPF = cms.Path()
  if not runOnMC:
    pPF += process.eventCleaning
  if useTrigger:
    pPF += process.step1
  pPF += process.goodOfflinePrimaryVertices
  if useGoodVertex:
    pPF += process.step2
  pPF += getattr( process, 'patPF2PATSequence' + postfix )
  pPF += getattr( process, 'patAddOnSequence' + postfix )
  if useLooseMuon:
    pPF += getattr( process, 'step3b' + postfix )
  if useTightMuon:
    pPF += getattr( process, 'step3a' + postfix )
  if useMuonVeto:
    pPF += getattr( process, 'step4' + postfix )
  if useElectronVeto:
    pPF += getattr( process, 'step5' + postfix )
  if use1Jet:
    pPF += getattr( process, 'step6a' + postfix )
  if use2Jets:
    pPF += getattr( process, 'step6b' + postfix )
  if use3Jets:
    pPF += getattr( process, 'step6c' + postfix )
  if use4Jets:
    pPF += getattr( process, 'step7' + postfix )
  setattr( process, 'p' + postfix, pPF )
  process.out.SelectEvents.SelectEvents.append( 'p' + postfix )


###
### Trigger matching
###

if addTriggerMatching:

  if runOnMC:
    triggerObjectSelection = triggerObjectSelectionMC
  else:
    triggerObjectSelection = triggerObjectSelectionData
  ### Trigger matching configuration
  from PhysicsTools.PatAlgos.triggerLayer1.triggerProducer_cfi import patTrigger
  from TopQuarkAnalysis.Configuration.patRefSel_triggerMatching_cfi import patMuonTriggerMatch
  from PhysicsTools.PatAlgos.tools.trigTools import *
  if useStandardPAT:
    triggerProducer = patTrigger.clone()
    setattr( process, 'patTrigger', triggerProducer )
    process.triggerMatch = patMuonTriggerMatch.clone( matchedCuts = triggerObjectSelection )
    switchOnTriggerMatchEmbedding( process
                                 , triggerMatchers = [ 'triggerMatch' ]
                                 )
    removeCleaningFromTriggerMatching( process )
    process.intermediatePatMuons.src = cms.InputTag( 'selectedPatMuonsTriggerMatch' )
  if runPF2PAT:
    triggerProducerPF = patTrigger.clone()
    setattr( process, 'patTrigger' + postfix, triggerProducerPF )
    triggerMatchPF = patMuonTriggerMatch.clone( matchedCuts = triggerObjectSelection )
    setattr( process, 'triggerMatch' + postfix, triggerMatchPF )
    switchOnTriggerMatchEmbedding( process
                                 , triggerProducer = 'patTrigger' + postfix
                                 , triggerMatchers = [ 'triggerMatch' + postfix ]
                                 , sequence        = 'patPF2PATSequence' + postfix
                                 , postfix         = postfix
                                 )
    removeCleaningFromTriggerMatching( process
                                     , sequence = 'patPF2PATSequence' + postfix
                                     )
    getattr( process, 'intermediatePatMuons' + postfix ).src = cms.InputTag( 'selectedPatMuons' + postfix + 'TriggerMatch' )
