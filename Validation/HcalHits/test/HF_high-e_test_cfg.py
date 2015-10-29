import FWCore.ParameterSet.Config as cms

process = cms.Process("HFTEST")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("SimGeneral.HepPDTESSource.pdt_cfi")

#--- Magnetic Field 		
process.load("Configuration.StandardSequences.MagneticField_cff")

#--- Full geometry or only HCAL+ECAL Geometry
#   include "Geometry/CMSCommonData/data/cmsIdealGeometryXML.cfi"
#   include "Geometry/TrackerNumberingBuilder/data/trackerNumberingGeometry.cfi"
process.load("Geometry.CMSCommonData.ecalhcalGeometryXML_cfi")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.Timing = cms.Service("Timing")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        generator = cms.untracked.uint32(456789),
        g4SimHits = cms.untracked.uint32(9876),
        VtxSmeared = cms.untracked.uint32(123456789)
    ),
    sourceSeed = cms.untracked.uint32(135799753)
)

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(11, 211),
        MinEta = cms.double(3.02),
        MaxEta = cms.double(5.10),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinE   = cms.double(1000.0),
        MaxE   = cms.double(5000.0)
    ),
    AddAntiParticle = cms.bool(False),
    Verbosity       = cms.untracked.int32(0),
    firstRun        = cms.untracked.uint32(1)
)

process.VtxSmeared = cms.EDProducer("GaussEvtVtxGenerator",
    MeanX = cms.double(0.0),
    MeanY = cms.double(0.0),
    MeanZ = cms.double(0.0),
    SigmaY = cms.double(0.0001),
    SigmaX = cms.double(0.0001),
    SigmaZ = cms.double(0.0001),
    TimeOffset = cms.double(0.0),
    src = cms.InputTag("generator","unsmeared")
)

process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('HF_test_ecalplushcalonly_nofield.root')
)

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.g4SimHits)
process.outpath = cms.EndPath(process.o1)
process.MessageLogger.cerr.default.limit = 100
process.g4SimHits.UseMagneticField = False


