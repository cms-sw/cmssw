import FWCore.ParameterSet.Config as cms

process = cms.Process("Dump")

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')

process.load('SimGeneral.MixingModule.mixNoPU_cfi')

process.load('Configuration.Geometry.GeometryExtended2023D1Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D1_cff')

process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

#rumi#process.GlobalTag.globaltag = 'DES23_62_V1::All'
#process.GlobalTag.globaltag = 'DES19_62_V8::All'
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
	
    '/store/user/mileva/testFiles/out_local_reco.root'

    )
)

#process.Tracer = cms.Service("Tracer")

process.dumper = cms.EDAnalyzer("GEMDigiSimLinkReader",
    simhitToken = cms.InputTag("g4SimHits","MuonGEMHits"), 
    gemDigiToken = cms.InputTag("simMuonGEMDigis"), 
    gemDigiSimLinkToken = cms.InputTag("simMuonGEMDigis","GEM") ,
    debugFlag = cms.bool(False),	#for Milena: add a comma in the end of the line
#forMilena
    rechitToken = cms.InputTag("gemRecHits","")
#forMilena end

)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('testGemDigiSimLink.root')
)

process.p = cms.Path(process.dumper)
