import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO2')

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
    skipEvents = cms.untracked.uint32(51), 
    fileNames = cms.untracked.vstring( 
#'file:/data1/arizzi/CMSSW_3_5_6/src/MCQCD80120START_C07EAA09-2D2C-DF11-87E7-002618943886.root'
'/store/relval/CMSSW_7_0_0/RelValTTbar_13/GEN-SIM-RECO/PU50ns_POSTLS170_V4-v2/00000/265B9219-FF98-E311-BF4A-02163E00EA95.root'
    )
)

# Output definition
process.FEVT = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('out.root')
)

# Other statements
process.GlobalTag.globaltag = 'POSTLS162_V1::All'


process.p = cms.Path(process.inclusiveVertexing)

process.out_step = cms.EndPath(process.FEVT)

