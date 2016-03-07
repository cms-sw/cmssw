# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: l1NtupleRECO -s RAW2DIGI --era=Run2_2016 --customise=L1Trigger/Configuration/customiseReEmul.L1TReEmulFromRAW --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.L1NtupleEMU --customise=L1Trigger/Configuration/customiseUtils.L1TTurnOffUnpackStage2GtGmtAndCalo --conditions=auto:run2_data -n 200 --mc --no_exec --no_output --filein=/store/mc/RunIIFall15DR76/SingleNeutrino/GEN-SIM-RAW/25nsPUfixed20NzshcalRaw_76X_mcRun2_asymptotic_v12-v1/50000/02237A93-FCA7-E511-8A33-02163E0152AB.root --geometry=Extended2016,Extended2016Reco --customise=L1Trigger/Configuration/customiseReEmul.L1TEventSetupForHF1x1TPs
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('RAW2DIGI',eras.Run2_2016)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2016Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(100)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/mc/RunIIFall15DR76/SingleNeutrino/GEN-SIM-RAW/25nsPUfixed20NzshcalRaw_76X_mcRun2_asymptotic_v12-v1/50000/02237A93-FCA7-E511-8A33-02163E0152AB.root'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('l1NtupleRECO nevts:200'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.endjob_step = cms.EndPath(process.endOfProcess)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.endjob_step)

# customisation of the process.

# Automatic addition of the customisation function from L1Trigger.Configuration.customiseReEmul
from L1Trigger.Configuration.customiseReEmul import L1TReEmulFromRAW,L1TEventSetupForHF1x1TPs 

#call to customisation function L1TReEmulFromRAW imported from L1Trigger.Configuration.customiseReEmul
process = L1TReEmulFromRAW(process)

#call to customisation function L1TEventSetupForHF1x1TPs imported from L1Trigger.Configuration.customiseReEmul
process = L1TEventSetupForHF1x1TPs(process)

# Automatic addition of the customisation function from L1Trigger.L1TNtuples.customiseL1Ntuple
from L1Trigger.L1TNtuples.customiseL1Ntuple import L1NtupleEMU 

#call to customisation function L1NtupleEMU imported from L1Trigger.L1TNtuples.customiseL1Ntuple
process = L1NtupleEMU(process)

# Automatic addition of the customisation function from L1Trigger.Configuration.customiseUtils
from L1Trigger.Configuration.customiseUtils import L1TTurnOffUnpackStage2GtGmtAndCalo 

#call to customisation function L1TTurnOffUnpackStage2GtGmtAndCalo imported from L1Trigger.Configuration.customiseUtils
process = L1TTurnOffUnpackStage2GtGmtAndCalo(process)

# End of customisation functions

# uGMT customisations
process.l1MuonFilter = cms.EDFilter("SelectL1Muons",
                                    ugmtInput = cms.InputTag("simGmtStage2Digis"),
                                    gmtInput = cms.InputTag("gtDigis"),
                                   )

process.load("L1Trigger.L1TNtuples.l1MuonUpgradeTreeProducer_cfi")
process.l1MuonUpgradeTreeProducer.ugmtTag = cms.InputTag("simGmtStage2Digis")
process.l1MuonUpgradeTreeProducer.bmtfTag = process.simGmtStage2Digis.barrelTFInput
process.l1MuonUpgradeTreeProducer.omtfTag = process.simGmtStage2Digis.overlapTFInput
process.l1MuonUpgradeTreeProducer.emtfTag = process.simGmtStage2Digis.forwardTFInput
process.l1MuonUpgradeTreeProducer.calo2x2Tag = process.simGmtStage2Digis.triggerTowerInput
process.l1MuonUpgradeTreeProducer.caloTag = cms.InputTag("simGmtCaloSumDigis", "TriggerTowerSums")

process.load("L1Trigger.L1TNtuples.l1Tree_cfi")
process.l1Tree.generatorSource      = cms.InputTag("genParticles")
#process.l1Tree.generatorSource      = cms.InputTag("none")
process.l1Tree.simulationSource     = cms.InputTag("none")
process.l1Tree.hltSource            = cms.InputTag("none")
process.l1Tree.gmtSource            = cms.InputTag("gtDigis")
process.l1Tree.gtEvmSource          = cms.InputTag("none")
process.l1Tree.gtSource             = cms.InputTag("none")
process.l1Tree.gctCentralJetsSource = cms.InputTag("none")
process.l1Tree.gctNonIsoEmSource    = cms.InputTag("none")
process.l1Tree.gctForwardJetsSource = cms.InputTag("none")
process.l1Tree.gctIsoEmSource       = cms.InputTag("none")
process.l1Tree.gctETTSource         = cms.InputTag("none")
process.l1Tree.gctETMSource         = cms.InputTag("none")
process.l1Tree.gctHTTSource         = cms.InputTag("none")
process.l1Tree.gctHTMSource         = cms.InputTag("none")
process.l1Tree.gctHFSumsSource      = cms.InputTag("none")
process.l1Tree.gctHFBitsSource      = cms.InputTag("none")
process.l1Tree.gctTauJetsSource     = cms.InputTag("none")
process.l1Tree.gctIsoTauJetsSource  = cms.InputTag("none")
process.l1Tree.rctRgnSource         = cms.InputTag("none")
process.l1Tree.rctEmSource          = cms.InputTag("none")
process.l1Tree.dttfThSource         = cms.InputTag("none")
process.l1Tree.dttfPhSource         = cms.InputTag("none")
process.l1Tree.dttfTrkSource        = cms.InputTag("none")
process.l1Tree.ecalSource           = cms.InputTag("none")
process.l1Tree.hcalSource           = cms.InputTag("none")
process.l1Tree.csctfTrkSource       = cms.InputTag("none")
process.l1Tree.csctfLCTSource       = cms.InputTag("none")
process.l1Tree.csctfStatusSource    = cms.InputTag("none")
process.l1Tree.csctfDTStubsSource   = cms.InputTag("none")

process.ugmtExtraSeq = cms.Sequence(process.l1MuonFilter
                                  + process.l1Tree
                                  + process.l1MuonUpgradeTreeProducer
)

process.L1NtupleEMU.remove(process.l1CaloTowerEmuTree)
 
process.ugmtExtra_step = cms.Path(process.ugmtExtraSeq)
process.schedule.append(process.ugmtExtra_step)

print process.schedule

