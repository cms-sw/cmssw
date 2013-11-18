import FWCore.ParameterSet.Config as cms

# make L1 ntuples from RAW+RECO

process = cms.Process("L1NTUPLE")

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/Geometry/GeometryIdeal_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/SimL1Emulator_cff')
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')
process.load('Configuration.StandardSequences.ReconstructionCosmics_cff')


process.mergedSuperClusters = cms.EDFilter("EgammaSuperClusterMerger",
#src = cms.VInputTag(cms.InputTag("correctedHybridSuperClusters"),cms.InputTag("correctedMulti5x5SuperClustersWithPreshower"))
src = cms.VInputTag(cms.InputTag("hybridSuperClusters"),cms.InputTag("multi5x5SuperClustersWithPreshower"))
)



# global tag
process.GlobalTag.globaltag = 'GR_P_V43::All'

# output file
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('L1Tree_2.root')
)

# analysis
process.load("L1Trigger.Configuration.L1Extra_cff")
process.load("L1TriggerDPG.L1Ntuples.l1NtupleProducer_cfi")
process.load("L1TriggerDPG.L1Ntuples.l1RecoTreeProducer_cfi")
process.load("L1TriggerDPG.L1Ntuples.l1ExtraTreeProducer_cfi")
process.load("L1TriggerDPG.L1Ntuples.l1MuonRecoTreeProducer_cfi")

process.load("L1TriggerDPG.L1Ntuples.l1MenuTreeProducer_cfi")
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtTriggerMenuLite_cfi")

process.p = cms.Path(
    process.gtDigis
    +process.gtEvmDigis
    +process.gctDigis
    +process.dttfDigis
    +process.csctfDigis
    +process.l1NtupleProducer
    +process.l1extraParticles
    +process.l1ExtraTreeProducer
    +process.l1GtTriggerMenuLite
    +process.l1MenuTreeProducer
    +process.l1RecoTreeProducer
    +process.l1MuonRecoTreeProducer
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(250) )

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",
                             fileNames = readFiles,
                             secondaryFileNames = secFiles
                             )

readFiles.extend( [
    '/store/data/Run2012D/SingleMu25ns/RECO/PromptReco-v1/000/209/151/0E5F8A5C-9249-E211-B109-5404A63886AA.root'
] )

secFiles.extend( [
    '/store/data/Run2012D/SingleMu25ns/RAW/v1/000/209/151/4607C9E6-BD47-E211-B249-003048D37694.root',
    '/store/data/Run2012D/SingleMu25ns/RAW/v1/000/209/151/688ADEDC-D547-E211-BFF1-BCAEC518FF80.root',
    '/store/data/Run2012D/SingleMu25ns/RAW/v1/000/209/151/9C02920D-B447-E211-A14E-003048F024FE.root',
    '/store/data/Run2012D/SingleMu25ns/RAW/v1/000/209/151/DA62B044-AC47-E211-8298-003048F118C2.root'
] )
