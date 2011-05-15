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

### Import and modify reference selection

from TopQuarkAnalysis.Configuration.patRefSel_refMuJets import *
#muonsUsePV             = False
#muonEmbedTrack         = True
#muonCut                = ''
#looseMuonCut           = ''
#tightMuonCut           = ''
#muonJetsDR             = 0.3
#jetCut                 = ''
#electronCut            = ''

# Trigger selection according to run range:
# lower range limits available as suffix;
# available are: 000000, 147196 (default)
#triggerSelection       = triggerSelection_147196
#triggerObjectSelection = triggerObjectSelection_147196

### JEC levels

# levels to be accessible from the jets
# jets are corrected to L3Absolute (MC), L2L3Residual (data) automatically
useL1FastJet    = True  # needs useL1Offset being off, error otherwise
useL1Offset     = False # needs useL1FastJet being off, error otherwise
useL2Relative   = True
useL3Absolute   = True
useL2L3Residual = True  # takes effect only on data
useL5Flavor     = True
useL7Parton     = True

### Input

# list of input files
useRelVals = True # if 'False', "inputFiles" is used
inputFiles = [ '/store/relval/CMSSW_4_2_3/RelValTTbar/GEN-SIM-DIGI-RECO/START42_V12_FastSim_PU_156BxLumiPileUp-v1/0072/0635AA67-B37C-E011-B61F-002618943944.root'
             , '/store/relval/CMSSW_4_2_3/RelValTTbar/GEN-SIM-DIGI-RECO/START42_V12_FastSim_PU_156BxLumiPileUp-v1/0072/0E153885-B17C-E011-8C7D-001A928116E0.root'
             , '/store/relval/CMSSW_4_2_3/RelValTTbar/GEN-SIM-DIGI-RECO/START42_V12_FastSim_PU_156BxLumiPileUp-v1/0072/105E01FE-B57C-E011-9AB4-0018F3D09708.root'
             , '/store/relval/CMSSW_4_2_3/RelValTTbar/GEN-SIM-DIGI-RECO/START42_V12_FastSim_PU_156BxLumiPileUp-v1/0072/120718C8-B67C-E011-A070-001A928116D2.root'
             , '/store/relval/CMSSW_4_2_3/RelValTTbar/GEN-SIM-DIGI-RECO/START42_V12_FastSim_PU_156BxLumiPileUp-v1/0072/1232DFFA-AF7C-E011-983D-002618943831.root'
             ]   # overwritten, if "useRelVals" is 'True'

# maximum number of events
maxInputEvents = -1 # reduce for testing

### Conditions

# GlobalTags (w/o suffix '::All')
globalTagData = 'GR_R_42_V12' # default for CMSSW_4_2_2 RelVals: 'GR_R_42_V10'
globalTagMC   = 'START42_V12' # default for CMSSW_4_2_2 RelVals: 'START42_V11'

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
                                     , numberOfFiles = 0 # "0" means "all"
                                     )
  else:
    inputFiles = pickRelValInputFiles( cmsswVersion  = 'CMSSW_4_2_3'
                                     , relVal        = 'Mu'
                                     , dataTier      = 'RECO'
                                     #, globalTag     = globalTagData + '_RelVal_mu2010B'
                                     , globalTag     = globalTagData + '_mu2010B'
                                     , numberOfFiles = 0 # "0" means "all"
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


###
### Cleaning and trigger selection configuration
###

### Event cleaning
process.load( 'TopQuarkAnalysis.Configuration.patRefSel_eventCleaning_cff' )

### Trigger selection
from TopQuarkAnalysis.Configuration.patRefSel_triggerSelection_cff import triggerResults
process.step1 = triggerResults.clone(
  triggerConditions = [ triggerSelection ]
)

### Good vertex selection
process.load( "TopQuarkAnalysis.Configuration.patRefSel_goodVertex_cfi" )


###
### PAT configuration
###

### Check JECs

# JEC levels
if useL1FastJet and useL1Offset:
  print 'ERROR: switch off either "L1FastJet" or "L1Offset"'
  exit
jecLevelsPF = []
if useL1FastJet:
  jecLevelsPF.append( 'L1FastJet' )
if useL1Offset:
  jecLevelsPF.append( 'L1Offset' )
if useL2Relative:
  jecLevelsPF.append( 'L2Relative' )
if useL3Absolute:
  jecLevelsPF.append( 'L3Absolute' )
if useL2L3Residual and not runOnMC:
  jecLevelsPF.append( 'L2L3Residual' )
if useL5Flavor:
  jecLevelsPF.append( 'L5Flavor' )
if useL7Parton:
  jecLevelsPF.append( 'L7Parton' )

process.load( "PhysicsTools.PatAlgos.patSequences_cff" )

# remove MC matching, object cleaning, photons and taus and adapt JECs
from PhysicsTools.PatAlgos.tools.coreTools import *
if not runOnMC:
  runOnData( process )
#removeCleaning( process )
removeSpecificPATObjects( process
                        , names = [ 'Photons', 'Taus' ]
                        ) # includes 'removeCleaning'
# additional event content has to be (re-)added _after_ the call to 'removeCleaning()':
process.out.outputCommands += [ 'keep edmTriggerResults_*_*_*'
                              , 'keep *_hltTriggerSummaryAOD_*_*'
                              # tracks, vertices and beam spot
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

process.load( "TopQuarkAnalysis.Configuration.patRefSel_refMuJets_cfi" )

if useL1FastJet:
  process.kt6PFJets.doRhoFastjet = True
  process.patDefaultSequence.replace( process.patJetCorrFactors
                                    , process.kt6PFJets * process.patJetCorrFactors
                                    )
  process.out.outputCommands.append( 'keep double_*_*_' + process.name_() )


###
### Selection configuration
###

### Muons

process.patMuons.usePV      = muonsUsePV
process.patMuons.embedTrack = muonEmbedTrack

process.selectedPatMuons.cut = muonCut

process.intermediatePatMuons.preselection = looseMuonCut
process.out.outputCommands.append( 'keep *_intermediatePatMuons_*_*' )

process.loosePatMuons.checkOverlaps.jets.deltaR = muonJetsDR
process.out.outputCommands.append( 'keep *_loosePatMuons_*_*' )

process.tightPatMuons.preselection = tightMuonCut
process.out.outputCommands.append( 'keep *_tightPatMuons_*_*' )

### Jets

process.goodPatJets.preselection = jetCut
process.out.outputCommands.append( 'keep *_goodPatJets_*_*' )

### Electrons

process.selectedPatElectrons.cut = electronCut


###
### Scheduling
###

# The additional sequence
process.patAddOnSequence = cms.Sequence(
  process.intermediatePatMuons
* process.goodPatJets
* process.loosePatMuons
* process.tightPatMuons
)

# The path
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


###
### Trigger matching
###

if addTriggerMatching:

  ### Trigger matching configuration
  from TopQuarkAnalysis.Configuration.patRefSel_triggerMatching_cfi import patMuonTriggerMatch
  process.triggerMatch = patMuonTriggerMatch.clone( matchedCuts = triggerObjectSelection )

  ### Enabling trigger matching and embedding
  from PhysicsTools.PatAlgos.tools.trigTools import *
  switchOnTriggerMatchEmbedding( process
                               , triggerMatchers = [ 'triggerMatch' ]
                               )
  # remove object cleaning as for the PAT default sequence
  removeCleaningFromTriggerMatching( process )
  switchOnTriggerMatchEmbedding( process
                               , triggerMatchers = [ 'triggerMatch' ]
                               ) # once more in order to fix event content
  # adapt input sources for additional muon collections
  process.loosePatMuons.src = cms.InputTag( 'selectedPatMuonsTriggerMatch' )
  process.tightPatMuons.src = cms.InputTag( 'selectedPatMuonsTriggerMatch' )
