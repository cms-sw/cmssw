import sys

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

# setup 'standard' options
options = VarParsing.VarParsing ('standard')
options.register('runOnMC', True, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, "decide if run on MC or data")

# parsing command line arguments
if( hasattr(sys, "argv") ):
  #options.parseArguments()
  if(len(sys.argv) > 1):
    print "Parsing command line arguments:"
  for args in sys.argv :
    arg = args.split(',')
    for val in arg:
      val = val.split('=')
      if(len(val)==2):
        print "Setting *", val[0], "* to:", val[1]
        setattr(options,val[0], val[1])

process = cms.Process( 'PAT' )


### ======================================================================== ###
###                                                                          ###
###                                 Constants                                ###
###                            (user job steering)                           ###
###                                                                          ###
### ======================================================================== ###


### Data or MC?
runOnMC = options.runOnMC

### Switch on/off selection steps

# Step 1
# (trigger selection: QuadJet50_Jet40)
useTrigger      = True
# Step 2
# (good vertex selection)
useGoodVertex   = True
# Step 3a
# (6 jets: pt > 30 GeV & |eta| < 2.4)
use6JetsLoose   = True
# Step 3b
# (4 jets: pt > 60 GeV, 5 jets: pt > 50 GeV, 6 jets: pt > 30 GeV, for all jets: |eta| < 2.4)
# (the cuts for the 4 and 5 leading jets are configurable via jetCutHard / jetCutMedium respectivly)
use6JetsTight   = True

addTriggerMatching = True

### Reference selection

from TopQuarkAnalysis.Configuration.patRefSel_refAllJets import *
# Muons
#muonCut = ''
# Electrons
#electronCut = ''
# Jets
#jetCut          = ''
#veryLooseJetCut = 'pt > 35.'
#looseJetCut     = 'pt > 50.'
#tightJetCut     = 'pt > 60.'

# Trigger and trigger object
#triggerSelectionData       = ''
#triggerObjectSelectionData = ''
#triggerSelectionMC       = ''
#triggerObjectSelectionMC = ''

### Particle flow

postfix = 'PF'

# subtract charged hadronic pile-up particles (from wrong PVs)
# effects also JECs
usePFnoPU       = True # before any top projection
usePfIsoLessCHS = True # switch to new PF isolation with L1Fastjet CHS

# other switches for PF top projections (default: all 'True')
useNoMuon     = True # before electron top projection
useNoElectron = True # before jet top projection
useNoJet      = True # before tau top projection
useNoTau      = True # before MET top projection

# cuts used in top projections
from TopQuarkAnalysis.Configuration.patRefSel_PF2PAT import *
# vertices
#pfVertices = 'goodOfflinePrimaryVertices'
#pfD0Cut   = 0.2
#pfDzCut   = 0.5
# muons
#pfMuonSelectionCut = 'pt > 5.'
useMuonCutBasePF = False # use minimal (veto) muon selection cut on top of 'pfMuonSelectionCut'
#pfMuonIsoConeR03 = False
#pfMuonCombIsoCut = 0.2
# electrons
#pfElectronSelectionCut  = 'pt > 5. && gsfTrackRef.isNonnull && gsfTrackRef.trackerExpectedHitsInner.numberOfLostHits < 2'
useElectronCutBasePF  = False # use minimal (veto) electron selection cut on top of 'pfElectronSelectionCut'
#pfElectronIsoConeR03 = True
#pfElectronCombIsoCut  = 0.2

### JEC levels

# levels to be accessible from the jets
# jets are corrected to L3Absolute (MC), L2L3Residual (data) automatically, if enabled here
# and remain uncorrected, if none of these levels is enabled here
useL1FastJet    = True  # needs useL1Offset being off, error otherwise
useL1Offset     = False # needs useL1FastJet being off, error otherwise
useL2Relative   = True
useL3Absolute   = True
useL2L3Residual = True
useL5Flavor     = False
useL7Parton     = False

typeIMetCorrections = True

### Input

# list of input files
useRelVals = True # if 'False', "inputFiles" is used
inputFiles = []   # overwritten, if "useRelVals" is 'True'


# maximum number of events
maxEvents = -1 # reduce for testing

### Conditions

# GlobalTags (w/o suffix '::All')
globalTagData = 'GR_R_52_V7D::All' # incl. Summer12 JEC and new b-tag SF
globalTagMC   = 'START52_V9C::All' # incl. Summer12 JEC and new b-tag SF

### Output

# output file
outputFile = 'patRefSel_allJets.root'

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
  process.GlobalTag.globaltag = globalTagMC
else:
  process.GlobalTag.globaltag = globalTagData


###
### Input configuration
###

process.load( "TopQuarkAnalysis.Configuration.patRefSel_inputModule_cfi" )
if useRelVals:
  from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles
  if runOnMC:
    inputFiles = pickRelValInputFiles( cmsswVersion  = 'CMSSW_5_2_5_cand1'
                                     , relVal        = 'RelValTTbar'
                                     , globalTag     = 'START52_V9'
                                     , maxVersions   = 1
                                     )
  else:
    print 'running on *Jet* data stream (instead of MultiJet) as no better stream exists as RelVal'
    inputFiles = pickRelValInputFiles( cmsswVersion  = 'CMSSW_5_2_5_cand1'
                                     , relVal        = 'Jet'
                                     , dataTier      = 'RECO'
                                     , globalTag     = 'GR_R_52_V7_RelVal_jet2011B'
                                     , maxVersions   = 1
                                     )
process.source.fileNames = inputFiles
process.maxEvents.input  = maxEvents


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

### Trigger selection
if runOnMC:
  triggerSelection = triggerSelectionMC
else:
  if useRelVals:
    triggerSelection = triggerSelectionDataRelVals
  else:
    triggerSelection = triggerSelectionData
from TopQuarkAnalysis.Configuration.patRefSel_triggerSelection_cff import triggerResults
process.step1 = triggerResults.clone(
  triggerConditions = [ triggerSelection ]
)

### Good vertex selection
process.load( "TopQuarkAnalysis.Configuration.patRefSel_goodVertex_cfi" )
process.step2 = process.goodOfflinePrimaryVertices.clone( filter = True )

### Event cleaning
process.load( 'TopQuarkAnalysis.Configuration.patRefSel_eventCleaning_cff' )
process.trackingFailureFilter.VertexSource = cms.InputTag( pfVertices )


###
### PAT/PF2PAT configuration
###

process.load( "PhysicsTools.PatAlgos.patSequences_cff" )

### Check JECs

# JEC set
jecSet = 'AK5PF'
if usePFnoPU:
  jecSet += 'chs'

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
if useL2L3Residual and not runOnMC:
  jecLevels.append( 'L2L3Residual' )
if useL5Flavor:
  jecLevels.append( 'L5Flavor' )
if useL7Parton:
  jecLevels.append( 'L7Parton' )

### Switch configuration
from PhysicsTools.PatAlgos.tools.pfTools import usePF2PAT
usePF2PAT( process
         , runPF2PAT           = True
         , runOnMC             = runOnMC
         , jetAlgo             = jetAlgo
         , postfix             = postfix
         , jetCorrections      = ( jecSet
                                 , jecLevels
                                 )
         , typeIMetCorrections = typeIMetCorrections
         , pvCollection        = cms.InputTag( pfVertices )
         )

if useMuonCutBasePF:
  pfMuonSelectionCut += ' && %s'%( muonCut )
if useElectronCutBasePF:
  pfElectronSelectionCut += ' && %s'%( electronCut )

getattr( process, 'pfNoPileUp'   + postfix ).enable = usePFnoPU
getattr( process, 'pfNoMuon'     + postfix ).enable = useNoMuon
getattr( process, 'pfNoElectron' + postfix ).enable = useNoElectron
getattr( process, 'pfNoJet'      + postfix ).enable = useNoJet
getattr( process, 'pfNoTau'      + postfix ).enable = useNoTau

if useL1FastJet:
  getattr( process, 'pfPileUpIso' + postfix ).checkClosestZVertex = usePfIsoLessCHS

getattr( process, 'pfMuonsFromVertex'     + postfix ).d0Cut    = pfD0Cut
getattr( process, 'pfMuonsFromVertex'     + postfix ).dzCut    = pfDzCut
getattr( process, 'pfSelectedMuons'       + postfix ).cut = pfMuonSelectionCut
getattr( process, 'pfIsolatedMuons'       + postfix ).isolationCut = pfMuonCombIsoCut
if pfMuonIsoConeR03:
  getattr( process, 'pfIsolatedMuons' + postfix ).isolationValueMapsCharged  = cms.VInputTag( cms.InputTag( 'muPFIsoValueCharged03' + postfix )
                                                                                                )
  getattr( process, 'pfIsolatedMuons' + postfix ).deltaBetaIsolationValueMap = cms.InputTag( 'muPFIsoValuePU03' + postfix )
  getattr( process, 'pfIsolatedMuons' + postfix ).isolationValueMapsNeutral  = cms.VInputTag( cms.InputTag( 'muPFIsoValueNeutral03' + postfix )
                                                                                                , cms.InputTag( 'muPFIsoValueGamma03' + postfix )
                                                                                                )
  getattr( process, 'pfMuons' + postfix ).isolationValueMapsCharged  = cms.VInputTag( cms.InputTag( 'muPFIsoValueCharged03' + postfix )
                                                                                        )
  getattr( process, 'pfMuons' + postfix ).deltaBetaIsolationValueMap = cms.InputTag( 'muPFIsoValuePU03' + postfix )
  getattr( process, 'pfMuons' + postfix ).isolationValueMapsNeutral  = cms.VInputTag( cms.InputTag( 'muPFIsoValueNeutral03' + postfix )
                                                                                        , cms.InputTag( 'muPFIsoValueGamma03' + postfix )
                                                                                        )
  getattr( process, 'patMuons' + postfix ).isolationValues.pfNeutralHadrons   = cms.InputTag( 'muPFIsoValueNeutral03' + postfix )
  getattr( process, 'patMuons' + postfix ).isolationValues.pfChargedAll       = cms.InputTag( 'muPFIsoValueChargedAll03' + postfix )
  getattr( process, 'patMuons' + postfix ).isolationValues.pfPUChargedHadrons = cms.InputTag( 'muPFIsoValuePU03' + postfix )
  getattr( process, 'patMuons' + postfix ).isolationValues.pfPhotons          = cms.InputTag( 'muPFIsoValueGamma03' + postfix )
  getattr( process, 'patMuons' + postfix ).isolationValues.pfChargedHadrons   = cms.InputTag( 'muPFIsoValueCharged03' + postfix )
getattr( process, 'pfElectronsFromVertex'     + postfix ).d0Cut    = pfD0Cut
getattr( process, 'pfElectronsFromVertex'     + postfix ).dzCut    = pfDzCut
getattr( process, 'pfSelectedElectrons'       + postfix ).cut = pfElectronSelectionCut
getattr( process, 'pfIsolatedElectrons'       + postfix ).isolationCut = pfElectronCombIsoCut
if pfElectronIsoConeR03:
  getattr( process, 'pfIsolatedElectrons' + postfix ).isolationValueMapsCharged  = cms.VInputTag( cms.InputTag( 'elPFIsoValueCharged03PFId' + postfix )
                                                                                                     )
  getattr( process, 'pfIsolatedElectrons' + postfix ).deltaBetaIsolationValueMap = cms.InputTag( 'elPFIsoValuePU03PFId' + postfix )
  getattr( process, 'pfIsolatedElectrons' + postfix ).isolationValueMapsNeutral  = cms.VInputTag( cms.InputTag( 'elPFIsoValueNeutral03PFId' + postfix )
                                                                                                    , cms.InputTag( 'elPFIsoValueGamma03PFId'   + postfix )
                                                                                                    )
  getattr( process, 'pfElectrons' + postfix ).isolationValueMapsCharged  = cms.VInputTag( cms.InputTag( 'elPFIsoValueCharged03PFId' + postfix )
                                                                                             )
  getattr( process, 'pfElectrons' + postfix ).deltaBetaIsolationValueMap = cms.InputTag( 'elPFIsoValuePU03PFId' + postfix )
  getattr( process, 'pfElectrons' + postfix ).isolationValueMapsNeutral  = cms.VInputTag( cms.InputTag( 'elPFIsoValueNeutral03PFId' + postfix )
                                                                                            , cms.InputTag( 'elPFIsoValueGamma03PFId'   + postfix )
                                                                                            )
  getattr( process, 'patElectrons' + postfix ).isolationValues.pfNeutralHadrons   = cms.InputTag( 'elPFIsoValueNeutral03PFId' + postfix )
  getattr( process, 'patElectrons' + postfix ).isolationValues.pfChargedAll       = cms.InputTag( 'elPFIsoValueChargedAll03PFId' + postfix )
  getattr( process, 'patElectrons' + postfix ).isolationValues.pfPUChargedHadrons = cms.InputTag( 'elPFIsoValuePU03PFId' + postfix )
  getattr( process, 'patElectrons' + postfix ).isolationValues.pfPhotons          = cms.InputTag( 'elPFIsoValueGamma03PFId' + postfix )
  getattr( process, 'patElectrons' + postfix ).isolationValues.pfChargedHadrons   = cms.InputTag( 'elPFIsoValueCharged03PFId' + postfix )


from PhysicsTools.PatAlgos.tools.coreTools import *

from TopQuarkAnalysis.Configuration.patRefSel_refAllJets_cfi import *

# remove MC matching, object cleaning, photons and taus

if not runOnMC:
  runOnData( process
           , names = [ 'PFAll' ]
           , postfix = postfix
           )
removeSpecificPATObjects( process
                        , names = [ 'Photons', 'Taus' ]
                        , postfix = postfix
                        ) # includes 'removeCleaning'

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
                                , 'keep *_addPileupInfo_*_*'
                                ]


###
### Additional configuration
###

### Muons

### Jets

veryLoosePatJets.src = cms.InputTag( 'selectedPatJets' + postfix )
veryLoosePatJets.cut = veryLooseJetCut
setattr( process, 'veryLoosePatJets' + postfix, veryLoosePatJets )
loosePatJets.src = cms.InputTag( 'veryLoosePatJets' + postfix )
loosePatJets.cut = looseJetCut
setattr( process, 'loosePatJets' + postfix, loosePatJets )
tightPatJets.src = cms.InputTag( 'loosePatJets' + postfix )
tightPatJets.cut = tightJetCut
setattr( process, 'tightPatJets' + postfix, tightPatJets )

step3a.src = cms.InputTag( 'selectedPatJets' + postfix )
setattr( process, 'step3a' + postfix, step3a )
step3b_1.src = cms.InputTag( 'tightPatJets' + postfix )
setattr( process, 'step3b_1' + postfix, step3b_1 )
step3b_2.src = cms.InputTag( 'loosePatJets' + postfix )
setattr( process, 'step3b_2' + postfix, step3b_2 )
step3b_3.src = cms.InputTag( 'veryLoosePatJets' + postfix )
setattr( process, 'step3b_3' + postfix, step3b_3 )
step3bPF = cms.Sequence(
  getattr( process, 'step3b_1' + postfix )
* getattr( process, 'step3b_2' + postfix )
* getattr( process, 'step3b_3' + postfix )
)
setattr( process, 'step3b' + postfix, step3bPF )

### Electrons


process.out.outputCommands.append( 'keep *_veryLoosePatJets*_*_*' )
process.out.outputCommands.append( 'keep *_loosePatJets*_*_*' )
process.out.outputCommands.append( 'keep *_tightPatJets*_*_*' )


###
### Selection configuration
###

### Muons

getattr( process, 'selectedPatMuons' + postfix ).cut = muonCut

### Jets

getattr( process, 'selectedPatJets'  + postfix ).cut = jetCut
getattr( process, 'veryLoosePatJets' + postfix ).cut = veryLooseJetCut
getattr( process, 'loosePatJets'     + postfix ).cut = looseJetCut
getattr( process, 'tightPatJets'     + postfix ).cut = tightJetCut

### Electrons

getattr( process, 'patElectrons' + postfix ).electronIDSources = electronIDSources

getattr( process, 'selectedPatElectrons' + postfix ).cut = electronCut


###
### Trigger matching
###

if addTriggerMatching:

  if runOnMC:
    triggerObjectSelection = triggerObjectSelectionMC
  else:
    if useRelVals:
      triggerObjectSelection = triggerObjectSelectionDataRelVals
    else:
      triggerObjectSelection = triggerObjectSelectionData

  ### Trigger matching configuration
  from PhysicsTools.PatAlgos.triggerLayer1.triggerProducer_cfi import patTrigger
  from TopQuarkAnalysis.Configuration.patRefSel_triggerMatching_cfi import patJetTriggerMatch
  from PhysicsTools.PatAlgos.tools.trigTools import *
  triggerProducerPF = patTrigger.clone()
  setattr( process, 'patTrigger' + postfix, triggerProducerPF )
  triggerMatchPF = patJetTriggerMatch.clone( matchedCuts = triggerObjectSelection )
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
  getattr( process, 'veryLoosePatJets' + postfix ).src = cms.InputTag( 'selectedPatJets' + postfix + 'TriggerMatch' )


###
### Scheduling
###

# MVA electron ID

process.load( "EGamma.EGammaAnalysisTools.electronIdMVAProducer_cfi" )
process.eidMVASequence = cms.Sequence(
  process.mvaTrigV0
+ process.mvaNonTrigV0
)

# The additional sequence

patAddOnSequence = cms.Sequence(
  getattr( process, 'veryLoosePatJets' + postfix )
* getattr( process, 'loosePatJets'     + postfix )
* getattr( process, 'tightPatJets'     + postfix )
)
setattr( process, 'patAddOnSequence' + postfix, patAddOnSequence )

# The paths

process.p = cms.Path()
if useTrigger:
  process.p += process.step1
process.p += process.goodOfflinePrimaryVertices
if useGoodVertex:
  process.p += process.step2
process.p += process.eventCleaning
if runOnMC:
  process.p += process.eventCleaningMC
else:
  process.p += process.eventCleaningData
process.p += process.eidMVASequence
process.p += getattr( process, 'patPF2PATSequence' + postfix )
process.p += getattr( process, 'patAddOnSequence' + postfix )
if use6JetsLoose:
  process.p += getattr( process, 'step3a' + postfix )
if use6JetsTight:
  process.p += getattr( process, 'step3b' + postfix )
process.out.SelectEvents.SelectEvents.append( 'p' )
