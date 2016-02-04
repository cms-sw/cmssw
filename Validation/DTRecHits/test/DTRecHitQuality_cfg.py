import FWCore.ParameterSet.Config as cms

process = cms.Process("DTRecHitQualityFromSimHit")
# process.load("FWCore.MessageService.MessageLogger_cfi")
# process.MessageLogger.cout.threshold = cms.untracked.string('ERROR')

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    destinations = cms.untracked.vstring('cout')
)

#include "Configuration/ReleaseValidation/data/Services.cfi"
process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load("SimMuon.Configuration.SimMuon_cff")

#     source = FlatRandomPtGunSource 
#     { 
#       untracked uint32 firstRun  =  1
#         untracked int32  maxEvents = 1000
#         untracked PSet  PGunParameters =
#         {
# # you can request more than 1 particle
# #untracked vint32  PartID = { 211, 11, -13 }
#           untracked vint32 PartID = {13, -13} 
#           untracked double MinEta = -1.3
#             untracked double MaxEta = 1.3
# #
# # phi must be given in radians
# #
#             untracked double MinPhi = -3.14159265358979323846
#             untracked double MaxPhi =  3.14159265358979323846
#             untracked double MinPt  =  10.
#             untracked double MaxPt  = 100.
#         }
#       untracked int32 Verbosity = 0 # set to 1 (or greater)  for printouts
#         untracked bool  AddAntiParticle = false   # if you turn it ON, for each particle
# # an anti-particle will be generated,
# # with 3-mom opposite to the particle's
#     }
#     include "IOMC/EventVertexGenerators/data/VtxSmearedGauss.cfi"
#     include "SimG4Core/Configuration/data/SimG4Core.cff"
#     include "SimMuon/Configuration/data/SimMuon.cff"
#     replace muonDTDigis.Smearing = 0.
#     
#     module mix = MixingModule {
#     int32 bunchspace = 25
#     }
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.DTGeometry.dtGeometry_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("RecoLocalMuon.Configuration.RecoLocalMuon_cff")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(13579),
        mix = cms.untracked.uint32(24680),
        VtxSmeared = cms.untracked.uint32(98765432)
    ),
    sourceSeed = cms.untracked.uint32(98765)
)

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(1),
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_2_0/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/946362E4-B4B9-DD11-A80E-001617E30F58.root')
    )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.mix = cms.EDFilter("MixingModule",
    bunchspace = cms.int32(25)
)

process.rechivalidation = cms.EDFilter("DTRecHitQuality",
    doStep2 = cms.untracked.bool(True),
    # Switches for analysis at various steps
    doStep1 = cms.untracked.bool(True),
    # Lable to retrieve RecHits from the event
    recHitLabel = cms.untracked.string('dt1DRecHits'),
    doStep3 = cms.untracked.bool(True),
    simHitLabel = cms.untracked.string('g4SimHits'),
    segment2DLabel = cms.untracked.string('dt2DSegments'),
    rootFileName = cms.untracked.string('DTRecHitQualityPlots.root'),
    debug = cms.untracked.bool(False),
    segment4DLabel = cms.untracked.string('dt4DSegments')
)

process.seg2dvalidation = cms.EDFilter("DTSegment2DQuality",
    sigmaResPos = cms.double(0.013),
    simHitLabel = cms.untracked.string('g4SimHits'),
    segment2DLabel = cms.untracked.string('dt2DSegments'),
    rootFileName = cms.untracked.string('DTSeg2DQualityPlots.root'),
    debug = cms.untracked.bool(False),
    sigmaResAngle = cms.double(0.008)
)

process.seg2dsuperphivalidation = cms.EDFilter("DTSegment2DSLPhiQuality",
    sigmaResPos = cms.double(0.013),
    simHitLabel = cms.untracked.string('g4SimHits'),
    sigmaResAngle = cms.double(0.008),
    rootFileName = cms.untracked.string('DTSeg2DSLPhiQualityPlots.root'),
    debug = cms.untracked.bool(False),
    segment4DLabel = cms.untracked.string('dt4DSegments')
)

process.seg4dvalidation = cms.EDFilter("DTSegment4DQuality",
    #resolution on angle
    sigmaResAlpha = cms.double(0.001),
    sigmaResBeta = cms.double(0.007),
    simHitLabel = cms.untracked.string('g4SimHits'),
    rootFileName = cms.untracked.string('DTSeg4DQualityPlots.root'),
    debug = cms.untracked.bool(False),
    #resolution on position
    sigmaResX = cms.double(0.01),
    sigmaResY = cms.double(0.05),
    segment4DLabel = cms.untracked.string('dt4DSegments')
)

process.p = cms.Path(
process.dtlocalreco_with_2DSegments*
process.rechivalidation*
process.seg2dvalidation*
process.seg2dsuperphivalidation*
process.seg4dvalidation
)
