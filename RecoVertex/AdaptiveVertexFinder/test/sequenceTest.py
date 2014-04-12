import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO2')

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

# import of standard configurations

process.load('Configuration/StandardSequences/Geometry_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/StandardSequences/Reconstruction_cff')
process.load('Configuration/EventContent/EventContent_cff')

#process.load('RecoVertex/AdaptiveVertexFinder/inclusiveVertexing_cff')


process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    wantSummary = cms.untracked.bool(True) 
)
# Input source
process.source = cms.Source("PoolSource",
#    skipEvents = cms.untracked.uint32(51), 
    fileNames = cms.untracked.vstring( 
#'file:/data1/arizzi/CMSSW_3_5_6/src/MCQCD80120START_C07EAA09-2D2C-DF11-87E7-002618943886.root'
#'/store/relval/CMSSW_7_0_0/RelValTTbar_13/GEN-SIM-RECO/PU50ns_POSTLS170_V4-v2/00000/265B9219-FF98-E311-BF4A-02163E00EA95.root'
#'/store/mc/Summer12_DR53X/TTJets_MassiveBinDECAY_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S10_START53_V7A-v1/0000/001C868B-B2E1-E111-9BE3-003048D4DCD8.root'
#'/store/mc/Fall13dr/EWKZjj_mqq120_mll50_13TeV_madgraph-pythia8/AODSIM/tsg_PU20bx25_POSTLS162_V2-v1/00000/0087CB53-3576-E311-BB3D-848F69FD5027.root'
#'/store/mc/Fall13dr/EWKZjj_mqq120_mll50_13TeV_madgraph-pythia8/AODSIM/tsg_PU40bx25_POSTLS162_V2-v1/00000/00A356DA-0C76-E311-B789-7845C4FC364D.root'
'/store/mc/Fall13dr/EWKZjj_mqq120_mll50_13TeV_madgraph-pythia8/AODSIM/tsg_PU40bx50_POSTLS162_V2-v1/00000/16C169C2-C475-E311-8BEE-7845C4FC3A2B.root'
#'/store/mc/Summer13dr53X/QCD_Pt-80to120_TuneZ2star_13TeV-pythia6/AODSIM/PU25bx25_START53_V19D-v1/20000/006F508D-3AE4-E211-B654-90E6BA0D09D4.root'
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

# Output definition
process.FEVT = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('out.root')
)

# Other statements
process.GlobalTag.globaltag = 'POSTLS161_V15::All'
#process.GlobalTag.globaltag = 'POSTLS162_V5::All'
#process.GlobalTag.globaltag = 'START53_V26::All'


process.p = cms.Path(process.inclusiveVertexing)

process.out_step = cms.EndPath(process.FEVT)

