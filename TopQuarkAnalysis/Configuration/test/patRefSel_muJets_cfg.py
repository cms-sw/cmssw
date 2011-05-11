import FWCore.ParameterSet.Config as cms

process = cms.Process( 'PAT' )


### ======================================================================== ###
###                                                                          ###
###                                 Constants                                ###
###                            (user job steering)                           ###
###                                                                          ###
### ======================================================================== ###


### Switch on/off selection steps

useTrigger      = True
useGoodVertex   = True
useLooseMuon    = True
useTightMuon    = True
useMuonVeto     = True
useElectronVeto = True
use1Jet         = True
use2Jets        = False
use3Jets        = False
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

### Input

# data or MC?
runOnMC = True

# list of input files
inputFiles = [] # overwritten, if "useRelVals" is 'True'

# test with CMSSW_4_2_2 RelVals?
useRelVals = True # if 'False', "inputFiles" must be defined

# maximum number of events
maxInputEvents = -1 # reduce for testing

### Conditions

# GlobalTags (w/o suffix '::All')
globalTagData = 'GR_R_42_V10' # default for CMSSW_4_2_2 RelVals: 'GR_R_42_V10'
globalTagMC   = 'START42_V11' # default for CMSSW_4_2_2 RelVals: 'START42_V11'

### Output

# output file
outputFile = 'patRefSel_muJets.root'

# event frequency of Fwk report
fwkReportEvery = 100

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

process.load( "TopQuarkAnalysis.Configuration.patRefSel_inputModule_cff" )
if useRelVals:
  from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles
  if runOnMC:
    inputFiles = pickRelValInputFiles( cmsswVersion  = 'CMSSW_4_2_2'
                                     , relVal        = 'RelValTTbar'
                                     , globalTag     = globalTagMC
                                     , numberOfFiles = 0 # "0" means "all"
                                     )
  else:
    inputFiles = pickRelValInputFiles( cmsswVersion  = 'CMSSW_4_2_2'
                                     , relVal        = 'Mu'
                                     , dataTier      = 'RECO'
                                     , globalTag     = globalTagData + '_RelVal_mu2010B'
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
  throw             = False
, triggerConditions = [ triggerSelection ]
)

### Good vertex selection
from TopQuarkAnalysis.Configuration.patRefSel_goodVertex_cff import goodVertex
process.step2 = goodVertex.clone()


###
### PAT configuration
###

process.load( "PhysicsTools.PatAlgos.patSequences_cff" )
# remove MC matching, object cleaning, photons and taus and adapt JECs
from PhysicsTools.PatAlgos.tools.coreTools import *
if not runOnMC:
  runOnData( process )
#removeCleaning( process )
removeSpecificPATObjects( process
                        , names = [ 'Photons', 'Taus' ]
                        ) # includes 'removeCleaning'
# additional event content has to be added _after_ the call to 'removeCleaning()':
process.out.outputCommands += [ 'keep edmTriggerResults_*_*_*'
                              , 'keep *_hltTriggerSummaryAOD_*_*'
                              # tracks, vertices and beam spot
                              , 'keep *_offlineBeamSpot_*_*'
                              , 'keep *_offlinePrimaryVertices*_*_*'
                              ]
if runOnMC:
  process.out.outputCommands += [ 'keep GenEventInfoProduct_*_*_*'
                                , 'keep recoGenParticles_*_*_*'
                                ]


###
### Selection configuration
###

process.load( "TopQuarkAnalysis.Configuration.patRefSel_refMuJets_cfi" )

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
