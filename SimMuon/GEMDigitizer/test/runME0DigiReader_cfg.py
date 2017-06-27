import FWCore.ParameterSet.Config as cms

process = cms.Process("Dump")

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023D9Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D9_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, '90X_upgrade2023_realistic_v1', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(	
#    'file:/afs/cern.ch/work/m/mileva/CMSSW_9_1_X_2017-03-29-2300/src/step2_newBkgTest.root'
    'file:/afs/cern.ch/work/m/mdyulend/work_cms1/CMSSW_9_2_X_2017-05-28-2300/src/20010.0_FourMuExtendedPt1_200+FourMuExtendedPt_1_200_pythia8_2023D7_GenSimHLBeamSpotFull+DigiFullTrigger_2023D7+RecoFullGlobal_2023D7+HARVESTFullGlobal_2023D7/step2.root'
    )
)

process.dumper = cms.EDAnalyzer("ME0DigiReader",
    simhitToken = cms.InputTag("g4SimHits","MuonME0Hits"), 
    me0DigiToken = cms.InputTag("simMuonME0Digis"), 
    me0StripDigiSimLinkToken = cms.InputTag("simMuonME0Digis","ME0"),
    debugFlag = cms.bool(False)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('testME0DigiReader.root')
)

process.p    = cms.Path(process.dumper)

