import FWCore.ParameterSet.Config as cms

process = cms.Process("Dump")

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')

process.load('SimGeneral.MixingModule.mixNoPU_cfi')

process.load('Configuration.Geometry.GeometryExtended2023Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023_cff')

#process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
#process.load('Configuration.Geometry.GeometryExtended2019_cff')

process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = 'DES23_62_V1::All'
#process.GlobalTag.globaltag = 'DES19_62_V8::All'

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

'root://eoscms//eos/cms/store/user/mileva/gemTest/pt100_digi_withGemSimDigiLink.root'
    )
)

process.dumper = cms.EDAnalyzer("GEMDigiSimLinkReader")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('testGemDigiSimLink.root')
)

process.p = cms.Path(process.dumper)
