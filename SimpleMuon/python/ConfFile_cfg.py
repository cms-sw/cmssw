import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2019_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff')
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load('Configuration.StandardSequences.Digi_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load("Configuration.StandardSequences.L1Emulator_cff")
process.load("Configuration.StandardSequences.L1Extra_cff")
process.load("RecoMuon.TrackingTools.MuonServiceProxy_cff")
process.load("SimMuon.CSCDigitizer.muonCSCDigis_cfi")
process.load('L1Trigger.CSCTrackFinder.csctfTrackDigisUngangedME1a_cfi')
process.simCsctfTrackDigis = process.csctfTrackDigisUngangedME1a.clone()
process.simCsctfTrackDigis.DTproducer = cms.untracked.InputTag("simDtTriggerPrimitiveDigis")
process.simCsctfTrackDigis.SectorReceiverInput = cms.untracked.InputTag("simCscTriggerPrimitiveDigis","MPCSORTED")
process.simCsctfTrackDigis.SectorProcessor.isCoreVerbose = cms.bool(True)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

inputFile = ['file:/afs/cern.ch/user/d/dildick/work/GEM/CMSSW_6_2_0_SLHC1/src/out_L1_MuonGun_neweta_PU100_Pt20_50k_digi_preTrig2.root']
#inputFile = ['file:/afs/cern.ch/user/d/dildick/work/GEM/CMSSW_6_2_0_SLHC1/src/out_sim_singleMuPt100Fwdv2.root']
process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
        #'file:myfile.root'
        *inputFile
    )
)

process.demo = cms.EDAnalyzer('SimpleMuon'
)

outputFileName = 'output.test.root'

process.TFileService = cms.Service("TFileService",
    fileName = cms.string(outputFileName)
)

process.p = cms.Path(process.demo)
