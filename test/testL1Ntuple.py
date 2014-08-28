
import FWCore.ParameterSet.Config as cms

process = cms.Process("L1NTUPLE")

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/GeometryIdeal_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/SimL1Emulator_cff')
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')
process.load('Configuration.StandardSequences.ReconstructionCosmics_cff')

# global tag
#process.GlobalTag.globaltag = 'GR09_31X_V5P::All'
process.GlobalTag.globaltag = 'GR09_R_35X_V3::All'

# output file
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('L1Tree.root')
)

# analysis
process.load("L1Trigger.Configuration.L1Extra_cff")
process.load("L1TriggerDPG.L1Ntuples.l1NtupleProducer_cfi")
process.load("L1TriggerDPG.L1Ntuples.l1RecoTreeProducer_cfi")
process.load("L1TriggerDPG.L1Ntuples.l1EgammaRecoTreeProducer_cfi")
process.load("L1TriggerDPG.L1Ntuples.l1ExtraTreeProducer_cfi")
process.load("L1TriggerDPG.L1Ntuples.l1MuonRecoTreeProducer_cfi")

process.p = cms.Path(
    process.gtDigis
    +process.gtEvmDigis
    +process.gctDigis
    +process.dttfDigis
    +process.csctfDigis
    +process.l1NtupleProducer
    +process.l1extraParticles
    +process.l1ExtraTreeProducer
    +process.l1RecoTreeProducer
    +process.l1MuonRecoTreeProducer
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(20) )

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",
                             fileNames = readFiles,
                             secondaryFileNames = secFiles
                             )

readFiles.extend( [
    #'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/615/FE6FE418-B8E2-DE11-A184-001D09F2423B.root'
     '/store/express/Commissioning10/ExpressPhysics/FEVT/v8/000/132/716/FEDD955C-2742-DF11-8024-001D09F23F2A.root'
#       '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/122/318/F23EF63B-7AD8-DE11-A6AC-0019B9F72F97.root',
#       '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/122/318/E0E8C7BF-7DD8-DE11-93F4-001617DC1F70.root',
#       '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/122/318/8E6024D2-7FD8-DE11-B2FD-001D09F295A1.root',
#       '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/122/318/8843C649-75D8-DE11-BAED-000423D6A6F4.root',
#       '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/122/318/24A1B63F-81D8-DE11-AFAD-003048D2C108.root',
#       '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/122/318/2222B70A-78D8-DE11-8E77-0019DB29C5FC.root'
] )

secFiles.extend( [
#       '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/122/318/94CEE17E-79D8-DE11-97D5-001D09F28F11.root',
#              '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/122/318/7EDBAEFA-70D8-DE11-ACFE-001617DBCF6A.root',
#              '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/122/318/648548EF-75D8-DE11-A26F-000423D94A04.root',
#              '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/122/318/2AEB364C-7CD8-DE11-A15C-001D09F241B9.root',
#              '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/122/318/16A68DCA-73D8-DE11-88FA-001617DBD224.root',
#              '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/122/318/0C749FE8-7AD8-DE11-B837-001D09F29114.root'
       ] )


process.L1MuTriggerScalesRcdSource = cms.ESSource("EmptyESSource",
recordName = cms.string('L1MuTriggerScalesRcd'),
iovIsRunNotTime = cms.bool(True),
firstValid = cms.vuint32(1)
)

process.L1MuTriggerScales = cms.ESProducer("L1MuTriggerScalesProducer",
signedPackingDTEta = cms.bool(True),
offsetDTEta = cms.int32(32),
nbinsDTEta = cms.int32(64),
offsetFwdRPCEta = cms.int32(16),
signedPackingBrlRPCEta = cms.bool(True),
maxDTEta = cms.double(1.2),
nbitPackingFwdRPCEta = cms.int32(6),
nbinsBrlRPCEta = cms.int32(33),
minCSCEta = cms.double(0.9),
nbitPackingGMTEta = cms.int32(6),
nbinsFwdRPCEta = cms.int32(33),
nbinsPhi = cms.int32(144),
nbitPackingPhi = cms.int32(8),
nbitPackingDTEta = cms.int32(6),
maxCSCEta = cms.double(2.5),
nbinsGMTEta = cms.int32(31),
minDTEta = cms.double(-1.2),
nbitPackingCSCEta = cms.int32(6),
signedPackingFwdRPCEta = cms.bool(True),
offsetBrlRPCEta = cms.int32(16),
scaleRPCEta = cms.vdouble(-2.1, -1.97, -1.85, -1.73, -1.61,
-1.48, -1.36, -1.24, -1.14, -1.04,
-0.93, -0.83, -0.72, -0.58, -0.44,
-0.27, -0.07, 0.07, 0.27, 0.44,
0.58, 0.72, 0.83, 0.93, 1.04,
1.14, 1.24, 1.36, 1.48, 1.61,
1.73, 1.85, 1.97, 2.1),
signedPackingPhi = cms.bool(False),
nbitPackingBrlRPCEta = cms.int32(6),
nbinsCSCEta = cms.int32(32),
maxPhi = cms.double(6.2831853),
minPhi = cms.double(0.0),
scaleGMTEta = cms.vdouble(0.0, 0.1, 0.2, 0.3, 0.4,
0.5, 0.6, 0.7, 0.8, 0.9,
1.0, 1.1, 1.2, 1.3, 1.4,
1.5, 1.6, 1.7, 1.75, 1.8,
1.85, 1.9, 1.95, 2.0, 2.05,
2.1, 2.15, 2.2, 2.25, 2.3,
2.35, 2.4)
)

process.L1MuTriggerPtScaleRcdSource = cms.ESSource("EmptyESSource",
recordName = cms.string('L1MuTriggerPtScaleRcd'),
iovIsRunNotTime = cms.bool(True),
firstValid = cms.vuint32(1)
)

process.L1MuTriggerPtScale = cms.ESProducer("L1MuTriggerPtScaleProducer",
nbitPackingPt = cms.int32(5),
scalePt = cms.vdouble(-1.0, 0.0, 1.5, 2.0, 2.5,
3.0, 3.5, 4.0, 4.5, 5.0,
6.0, 7.0, 8.0, 10.0, 12.0,
14.0, 16.0, 18.0, 20.0, 25.0,
30.0, 35.0, 40.0, 45.0, 50.0,
60.0, 70.0, 80.0, 90.0, 100.0,
120.0, 140.0, 1000000.0),
signedPackingPt = cms.bool(False),
nbinsPt = cms.int32(32)
)
