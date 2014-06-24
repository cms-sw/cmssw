import FWCore.ParameterSet.Config as cms

# make L1 ntuples from RAW+RECO

process = cms.Process("L1NTUPLE")

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/GeometryIdeal_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/SimL1Emulator_cff')
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')

# global tag
process.GlobalTag.globaltag = 'GR_P_V14::All'

# output file
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('L1Tree.root')
)

# analysis
process.load("L1Trigger.Configuration.L1Extra_cff")
process.load("L1TriggerDPG.L1Ntuples.l1NtupleProducer_cfi")
process.load("L1TriggerDPG.L1Ntuples.l1RecoTreeProducer_cfi")
process.load("L1TriggerDPG.L1Ntuples.l1ExtraTreeProducer_cfi")
process.load("L1TriggerDPG.L1Ntuples.l1MuonRecoTreeProducer_cfi")

process.gctDigis.numberOfGctSamplesToUnpack = cms.uint32(5)
process.l1extraParticles.centralBxOnly = cms.bool(False)

process.p = cms.Path(
    process.gtDigis
    +process.gtEvmDigis
    +process.gctDigis
    +process.dttfDigis
    +process.csctfDigis
    +process.l1NtupleProducer
    +process.l1extraParticles
    +process.l1ExtraTreeProducer
    #+process.l1RecoTreeProducer
    #+process.l1MuonRecoTreeProducer
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",
                             fileNames = readFiles,
                             secondaryFileNames = secFiles
                             )

readFiles.extend( [
# '/store/express/Commissioning10/ExpressPhysics/FEVT/v9/000/133/874/FEFC3201-644F-DF11-AED5-000423D98800.root'
#       '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/122/318/F23EF63B-7AD8-DE11-A6AC-0019B9F72F97.root',
#       '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/122/318/E0E8C7BF-7DD8-DE11-93F4-001617DC1F70.root',
#       '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/122/318/8E6024D2-7FD8-DE11-B2FD-001D09F295A1.root',
#       '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/122/318/8843C649-75D8-DE11-BAED-000423D6A6F4.root',
#       '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/122/318/24A1B63F-81D8-DE11-AFAD-003048D2C108.root',
#       '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/122/318/2222B70A-78D8-DE11-8E77-0019DB29C5FC.root'
] )

secFiles.extend( [
#       '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/122/318/94CEE17E-79D8-DE11-97D5-001D09F28F11.root',
#       '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/122/318/7EDBAEFA-70D8-DE11-ACFE-001617DBCF6A.root',
#       '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/122/318/648548EF-75D8-DE11-A26F-000423D94A04.root',
#       '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/122/318/2AEB364C-7CD8-DE11-A15C-001D09F241B9.root',
#       '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/122/318/16A68DCA-73D8-DE11-88FA-001617DBD224.root',
#       '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/122/318/0C749FE8-7AD8-DE11-B837-001D09F29114.root'
       ] )
