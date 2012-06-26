import sys

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

# setup 'standard' options
options = VarParsing.VarParsing ('standard')
options.register('runOnMC', True, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, "decide if run on MC or data")
options.register('maxEvents', -1, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "maximum number of input events")

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

### Standard and PF reconstruction
runStandardPAT = True
runPF2PAT      = True

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
use6JetsTight   = False

addTriggerMatching = True

### Reference selection

from TopQuarkAnalysis.Configuration.patRefSel_refAllJets import *
#muonsUsePV             = False
#muonEmbedTrack         = True
#muonCutPF              = ''
#looseMuonCutPF         = ''
#tightMuonCutPF         = ''
#muonJetsDR             = 0.3
#jetCutPF               = ''
#jetMuonsDRPF           = 0.1
#electronCutPF          = ''
#jetCutMedium           = ''
#jetCutHard             = ''

# Trigger selection according to run range resp. MC sample:
# lower range limits for data available as suffix;
# available are: 160404 (default)

# Trigger and trigger object
#triggerSelectionData       = ''
#triggerObjectSelectionData = ''
#triggerSelectionMC       = ''
#triggerObjectSelectionMC = ''

### Particle flow
### takes effect only, if 'runPF2PAT' = True

postfix = 'PF' # needs to be a non-empty string and must not be 'AK5PF', if 'runStandardPAT' = True

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
#pfElectronIsoConeR03 = False
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

### Input

# list of input files
useRelVals = True # if 'False', "inputFiles" is used
inputFiles = []   # overwritten, if "useRelVals" is 'True'


# maximum number of events
maxEvents = options.maxEvents

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

### Event cleaning
process.load( 'TopQuarkAnalysis.Configuration.patRefSel_eventCleaning_cff' )
process.trackingFailureFilter.VertexSource = cms.InputTag( pfVertices )

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


###
### PAT/PF2PAT configuration
###

if runStandardPAT and runPF2PAT:
  if postfix == '':
    sys.exit( 'ERROR: running standard PAT and PF2PAT in parallel requires a defined "postfix" for PF2PAT' )
if not runStandardPAT and not runPF2PAT:
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
if useL2L3Residual and not runOnMC:
  jecLevels.append( 'L2L3Residual' )
if useL5Flavor:
  jecLevels.append( 'L5Flavor' )
if useL7Parton:
  jecLevels.append( 'L7Parton' )

### Switch configuration

if runPF2PAT:
  if useMuonCutBasePF:
    pfMuonSelectionCut += ' && %s'%( muonCutBase )
  if useElectronCutBasePF:
    pfElectronSelectionCut += ' && %s'%( electronCutBase )
  from PhysicsTools.PatAlgos.tools.pfTools import usePF2PAT
  usePF2PAT( process
           , runPF2PAT      = runPF2PAT
           , runOnMC        = runOnMC
           , jetAlgo        = jetAlgo
           , postfix        = postfix
           , jetCorrections = ( jecSetPF
                              , jecLevels
                              )
           , typeIMetCorrections = True
           , pvCollection   = cms.InputTag( pfVertices )
           )
  applyPostfix( process, 'pfNoPileUp'  , postfix ).enable = usePFnoPU
  applyPostfix( process, 'pfNoMuon'    , postfix ).enable = useNoMuon
  applyPostfix( process, 'pfNoElectron', postfix ).enable = useNoElectron
  applyPostfix( process, 'pfNoJet'     , postfix ).enable = useNoJet
  applyPostfix( process, 'pfNoTau'     , postfix ).enable = useNoTau
  if useL1FastJet:
    applyPostfix( process, 'pfPileUp'   , postfix ).checkClosestZVertex = False
    applyPostfix( process, 'pfPileUpIso', postfix ).checkClosestZVertex = usePfIsoLessCHS
    applyPostfix( process, 'pfJets', postfix ).doAreaFastjet = True
    applyPostfix( process, 'pfJets', postfix ).doRhoFastjet  = False
  applyPostfix( process, 'pfMuonsFromVertex'    , postfix ).d0Cut    = pfD0Cut
  applyPostfix( process, 'pfMuonsFromVertex'    , postfix ).dzCut    = pfDzCut
  applyPostfix( process, 'pfSelectedMuons'      , postfix ).cut = pfMuonSelectionCut
  applyPostfix( process, 'pfIsolatedMuons'      , postfix ).isolationCut = pfMuonCombIsoCut
  if pfMuonIsoConeR03:
    applyPostfix( process, 'pfIsolatedMuons', postfix ).isolationValueMapsCharged  = cms.VInputTag( cms.InputTag( 'muPFIsoValueCharged03' + postfix )
                                                                                                  )
    applyPostfix( process, 'pfIsolatedMuons', postfix ).deltaBetaIsolationValueMap = cms.InputTag( 'muPFIsoValuePU03' + postfix )
    applyPostfix( process, 'pfIsolatedMuons', postfix ).isolationValueMapsNeutral  = cms.VInputTag( cms.InputTag( 'muPFIsoValueNeutral03' + postfix )
                                                                                                  , cms.InputTag( 'muPFIsoValueGamma03' + postfix )
                                                                                                  )
    applyPostfix( process, 'pfMuons', postfix ).isolationValueMapsCharged  = cms.VInputTag( cms.InputTag( 'muPFIsoValueCharged03' + postfix )
                                                                                          )
    applyPostfix( process, 'pfMuons', postfix ).deltaBetaIsolationValueMap = cms.InputTag( 'muPFIsoValuePU03' + postfix )
    applyPostfix( process, 'pfMuons', postfix ).isolationValueMapsNeutral  = cms.VInputTag( cms.InputTag( 'muPFIsoValueNeutral03' + postfix )
                                                                                          , cms.InputTag( 'muPFIsoValueGamma03' + postfix )
                                                                                          )
    applyPostfix( process, 'patMuons', postfix ).isolationValues.pfNeutralHadrons   = cms.InputTag( 'muPFIsoValueNeutral03' + postfix )
    applyPostfix( process, 'patMuons', postfix ).isolationValues.pfChargedAll       = cms.InputTag( 'muPFIsoValueChargedAll03' + postfix )
    applyPostfix( process, 'patMuons', postfix ).isolationValues.pfPUChargedHadrons = cms.InputTag( 'muPFIsoValuePU03' + postfix )
    applyPostfix( process, 'patMuons', postfix ).isolationValues.pfPhotons          = cms.InputTag( 'muPFIsoValueGamma03' + postfix )
    applyPostfix( process, 'patMuons', postfix ).isolationValues.pfChargedHadrons   = cms.InputTag( 'muPFIsoValueCharged03' + postfix )
  applyPostfix( process, 'pfElectronsFromVertex'    , postfix ).d0Cut    = pfD0Cut
  applyPostfix( process, 'pfElectronsFromVertex'    , postfix ).dzCut    = pfDzCut
  applyPostfix( process, 'pfSelectedElectrons'      , postfix ).cut = pfElectronSelectionCut
  applyPostfix( process, 'pfIsolatedElectrons'      , postfix ).isolationCut = pfElectronCombIsoCut
  if pfElectronIsoConeR03:
    applyPostfix( process, 'pfIsolatedElectrons', postfix ).isolationValueMapsCharged  = cms.VInputTag( cms.InputTag( 'elPFIsoValueCharged03PFId' + postfix )
                                                                                                       )
    applyPostfix( process, 'pfIsolatedElectrons', postfix ).deltaBetaIsolationValueMap = cms.InputTag( 'elPFIsoValuePU03PFId' + postfix )
    applyPostfix( process, 'pfIsolatedElectrons', postfix ).isolationValueMapsNeutral  = cms.VInputTag( cms.InputTag( 'elPFIsoValueNeutral03PFId' + postfix )
                                                                                                      , cms.InputTag( 'elPFIsoValueGamma03PFId'   + postfix )
                                                                                                      )
    applyPostfix( process, 'pfElectrons', postfix ).isolationValueMapsCharged  = cms.VInputTag( cms.InputTag( 'elPFIsoValueCharged03PFId' + postfix )
                                                                                               )
    applyPostfix( process, 'pfElectrons', postfix ).deltaBetaIsolationValueMap = cms.InputTag( 'elPFIsoValuePU03PFId' + postfix )
    applyPostfix( process, 'pfElectrons', postfix ).isolationValueMapsNeutral  = cms.VInputTag( cms.InputTag( 'elPFIsoValueNeutral03PFId' + postfix )
                                                                                              , cms.InputTag( 'elPFIsoValueGamma03PFId'   + postfix )
                                                                                              )
    applyPostfix( process, 'patElectrons', postfix ).isolationValues.pfNeutralHadrons   = cms.InputTag( 'elPFIsoValueNeutral03PFId' + postfix )
    applyPostfix( process, 'patElectrons', postfix ).isolationValues.pfChargedAll       = cms.InputTag( 'elPFIsoValueChargedAll03PFId' + postfix )
    applyPostfix( process, 'patElectrons', postfix ).isolationValues.pfPUChargedHadrons = cms.InputTag( 'elPFIsoValuePU03PFId' + postfix )
    applyPostfix( process, 'patElectrons', postfix ).isolationValues.pfPhotons          = cms.InputTag( 'elPFIsoValueGamma03PFId' + postfix )
    applyPostfix( process, 'patElectrons', postfix ).isolationValues.pfChargedHadrons   = cms.InputTag( 'elPFIsoValueCharged03PFId' + postfix )


from TopQuarkAnalysis.Configuration.patRefSel_refAllJets_cfi import *

# remove MC matching, object cleaning, photons and taus
if runStandardPAT:
  if not runOnMC:
    runOnData( process )
  from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection
  jecSetPFNoCHS = jecSetPF.rstrip('chs')
  addJetCollection(process,cms.InputTag('ak5PFJets'),'AK5','PF',
                   doJTA        = True,
                   doBTagging   = True,
                   jetCorrLabel = (jecSetPFNoCHS, jecLevels),
                   doType1MET   = False,
                   doL1Cleaning = False,
                   doL1Counters = True,
                   genJetCollection=cms.InputTag('ak5GenJets'),
                   doJetID      = True
                   )
  from PhysicsTools.PatAlgos.tools.metTools import addPfMET
  addPfMET(process, 'AK5PF')
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

if runStandardPAT:

  ### Muons

  process.intermediatePatMuons = intermediatePatMuons.clone()
  process.loosePatMuons        = loosePatMuons.clone()
  process.tightPatMuons        = tightPatMuons.clone()
  process.step3a               = step3a.clone()

  ### Jets

  process.step3b_1 = step3b_1.clone()
  process.step3b_2 = step3b_2.clone()
  process.step3b_3 = step3b_3.clone()
  process.step3b   = cms.Sequence( process.step3b_1 * process.step3b_2 * process.step3b_3 )

  process.patJetCorrFactors.payload = jecSet
  process.patJetCorrFactors.levels  = jecLevels
  if useL1FastJet:
    process.patJetCorrFactors.useRho = True

  process.goodPatJets       = goodPatJets.clone()
  process.goodPatJetsMedium = process.goodPatJets.clone()
  process.goodPatJetsHard   = process.goodPatJets.clone()
  process.goodPatJetsAK5PF       = goodPatJets.clone()
  process.goodPatJetsMediumAK5PF = process.goodPatJets.clone()
  process.goodPatJetsHardAK5PF   = process.goodPatJets.clone()

  ### Electrons

if runPF2PAT:

  ### Muons

  intermediatePatMuonsPF = intermediatePatMuons.clone( src = cms.InputTag( 'selectedPatMuons' + postfix ) )
  setattr( process, 'intermediatePatMuons' + postfix, intermediatePatMuonsPF )

  loosePatMuonsPF = loosePatMuons.clone( src = cms.InputTag( 'intermediatePatMuons' + postfix ) )
  setattr( process, 'loosePatMuons' + postfix, loosePatMuonsPF )
  getattr( process, 'loosePatMuons' + postfix ).checkOverlaps.jets.src = cms.InputTag( 'goodPatJets' + postfix )

  tightPatMuonsPF = tightPatMuons.clone( src = cms.InputTag( 'loosePatMuons' + postfix ) )
  setattr( process, 'tightPatMuons' + postfix, tightPatMuonsPF )

  ### Jets

  goodPatJetsPF = goodPatJets.clone( src = cms.InputTag( 'selectedPatJets' + postfix ), checkOverlaps = cms.PSet() )
  setattr( process, 'goodPatJets' + postfix, goodPatJetsPF )

  goodPatJetsMediumPF = getattr( process, 'goodPatJets' + postfix ).clone()
  setattr( process, 'goodPatJetsMedium' + postfix, goodPatJetsMediumPF )
  goodPatJetsHardPF = getattr( process, 'goodPatJets' + postfix ).clone()
  setattr( process, 'goodPatJetsHard' + postfix, goodPatJetsHardPF )

  step3aPF = step3a.clone( src = cms.InputTag( 'goodPatJets' + postfix ) )
  setattr( process, 'step3a' + postfix, step3aPF )

  step3b_1PF = step3b_1.clone()
  setattr( process, 'step3b_1' + postfix, step3b_1PF )
  step3b_2PF = step3b_2.clone()
  setattr( process, 'step3b_2' + postfix, step3b_2PF )
  step3b_3PF = step3b_3.clone()
  setattr( process, 'step3b_3' + postfix, step3b_3PF )
  step3bPF = cms.Sequence( step3b_1PF * step3b_2PF * step3b_3PF )
  setattr( process, 'step3b' + postfix, step3bPF )

  ### Electrons


# keep produced collections in the PAT tuple
process.out.outputCommands.append( 'keep *_intermediatePatMuons*_*_*' )
process.out.outputCommands.append( 'keep *_loosePatMuons*_*_*' )
process.out.outputCommands.append( 'keep *_tightPatMuons*_*_*' )
process.out.outputCommands.append( 'keep *_goodPatJets*_*_*' )


###
### Selection configuration
###

if runStandardPAT:

  ### Muons

  process.patMuons.usePV      = muonsUsePV
  process.patMuons.embedTrack = muonEmbedTrack

  process.selectedPatMuons.cut = muonCut

  process.intermediatePatMuons.preselection = looseMuonCut

  process.loosePatMuons.checkOverlaps.jets.deltaR = muonJetsDR

  process.tightPatMuons.preselection = tightMuonCut

  ### Jets

  process.goodPatJets.preselection       = jetCut
  process.goodPatJetsMedium.preselection = jetCut + jetCutMedium
  process.goodPatJetsHard.preselection   = jetCut + jetCutHard

  process.goodPatJetsAK5PF.src       = 'selectedPatJetsAK5PF'
  process.goodPatJetsMediumAK5PF.src = 'selectedPatJetsAK5PF'
  process.goodPatJetsHardAK5PF.src   = 'selectedPatJetsAK5PF'

  process.goodPatJetsAK5PF.preselection       = jetCutPF
  process.goodPatJetsMediumAK5PF.preselection = jetCutPF + jetCutMedium
  process.goodPatJetsHardAK5PF.preselection   = jetCutPF + jetCutHard

  process.goodPatJetsAK5PF.checkOverlaps.muons.deltaR       = jetMuonsDRPF
  process.goodPatJetsMediumAK5PF.checkOverlaps.muons.deltaR = jetMuonsDRPF
  process.goodPatJetsHardAK5PF.checkOverlaps.muons.deltaR   = jetMuonsDRPF

  ### Electrons

  process.patElectrons.electronIDSources = electronIDSources

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

  getattr( process, 'goodPatJets'       + postfix ).preselection = jetCutPF
  getattr( process, 'goodPatJetsMedium' + postfix ).preselection = jetCutPF + jetCutMedium
  getattr( process, 'goodPatJetsHard'   + postfix ).preselection = jetCutPF + jetCutHard

  ### Electrons

  applyPostfix( process, 'patElectrons', postfix ).electronIDSources = electronIDSources

  applyPostfix( process, 'selectedPatElectrons', postfix ).cut = electronCutPF


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
  if runStandardPAT:
    triggerProducer = patTrigger.clone()
    setattr( process, 'patTrigger', triggerProducer )
    process.triggerMatch      = patJetTriggerMatch.clone( matchedCuts = triggerObjectSelection )
    process.triggerMatchAK5PF = patJetTriggerMatch.clone( matchedCuts = triggerObjectSelection, src = 'selectedPatJetsAK5PF' )
    switchOnTriggerMatchEmbedding( process
                                 , triggerMatchers = [ 'triggerMatch', 'triggerMatchAK5PF' ]
                                 )
    removeCleaningFromTriggerMatching( process )
    process.goodPatJets.src       = cms.InputTag( 'selectedPatJetsTriggerMatch' )
    process.goodPatJetsMedium.src = cms.InputTag( 'selectedPatJetsTriggerMatch' )
    process.goodPatJetsHard.src   = cms.InputTag( 'selectedPatJetsTriggerMatch' )
    process.goodPatJetsAK5PF.src       = cms.InputTag( 'selectedPatJetsAK5PFTriggerMatch' )
    process.goodPatJetsMediumAK5PF.src = cms.InputTag( 'selectedPatJetsAK5PFTriggerMatch' )
    process.goodPatJetsHardAK5PF.src   = cms.InputTag( 'selectedPatJetsAK5PFTriggerMatch' )
  if runPF2PAT:
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
    getattr( process, 'goodPatJets'       + postfix ).src = cms.InputTag( 'selectedPatJets' + postfix + 'TriggerMatch' )
    getattr( process, 'goodPatJetsMedium' + postfix ).src = cms.InputTag( 'selectedPatJets' + postfix + 'TriggerMatch' )
    getattr( process, 'goodPatJetsHard'   + postfix ).src = cms.InputTag( 'selectedPatJets' + postfix + 'TriggerMatch' )


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

if runStandardPAT:
  process.patAddOnSequence = cms.Sequence(
    process.intermediatePatMuons
  * process.goodPatJets
  * process.goodPatJetsMedium
  * process.goodPatJetsHard
  * process.goodPatJetsAK5PF
  * process.goodPatJetsMediumAK5PF
  * process.goodPatJetsHardAK5PF
  * process.loosePatMuons
  * process.tightPatMuons
  )
if runPF2PAT:
  patAddOnSequence = cms.Sequence(
    getattr( process, 'intermediatePatMuons' + postfix )
  * getattr( process, 'goodPatJets'          + postfix )
  * getattr( process, 'goodPatJetsMedium'    + postfix )
  * getattr( process, 'goodPatJetsHard'      + postfix )
  * getattr( process, 'loosePatMuons'        + postfix )
  * getattr( process, 'tightPatMuons'        + postfix )
  )
  setattr( process, 'patAddOnSequence' + postfix, patAddOnSequence )

# The paths
if runStandardPAT:
  process.p = cms.Path( process.goodOfflinePrimaryVertices )
  process.p += process.eventCleaning
  if runOnMC:
    process.p += process.eventCleaningMC
  else:
    process.p += process.eventCleaningData
  if useTrigger:
    process.p += process.step1
  if useGoodVertex:
    process.p += process.step2
  process.p += process.eidMVASequence
  process.p += process.patDefaultSequence
  process.p += process.patAddOnSequence
  if use6JetsLoose:
    process.p += process.step3a
  if use6JetsTight:
    process.p += process.step3b
  process.out.SelectEvents.SelectEvents.append( 'p' )

if runPF2PAT:
  pPF = cms.Path( process.goodOfflinePrimaryVertices )
  pPF += process.eventCleaning
  if runOnMC:
    pPF += process.eventCleaningMC
  else:
    pPF += process.eventCleaningData
  if useTrigger:
    pPF += process.step1
  if useGoodVertex:
    pPF += process.step2
  pPF += process.eidMVASequence
  pPF += getattr( process, 'patPF2PATSequence' + postfix )
  pPF += getattr( process, 'patAddOnSequence' + postfix )
  if use6JetsLoose:
    pPF += getattr( process, 'step3a' + postfix )
  if use6JetsTight:
    pPF += getattr( process, 'step3b' + postfix )
  setattr( process, 'p' + postfix, pPF )
  process.out.SelectEvents.SelectEvents.append( 'p' + postfix )
