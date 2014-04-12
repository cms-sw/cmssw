import FWCore.ParameterSet.Config as cms

process = cms.Process("Sim")
process.load("Configuration.Generator.QCD_Pt_3000_3500_cfi")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("SimG4Core.Application.g4SimHits_cfi")
process.load("SimG4CMS.Calo.CaloSimHitStudy_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('SimG4CoreSensitiveDetector', 
        'SimG4CoreGeometry', 'SimG4CoreApplication', 'MagneticField',
        'VolumeBasedMagneticFieldESProducer', 'TrackerSimInfo',
        'TrackerSimInfoNumbering', 'TrackerMapDDDtoID',
        'CaloSim', 'EcalGeom', 'EcalSim',
        'HCalGeom', 'HcalSim', 'HFShower', 'BscSim'),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        SimG4CoreSensitiveDetector = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        SimG4CoreApplication = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        SimG4CoreGeometry = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        MagneticField = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        VolumeBasedMagneticFieldESProducer = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        TrackerSimInfo = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        TrackerSimInfoNumbering = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        TrackerMapDDDtoID = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        CaloSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EcalGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EcalSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HCalGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HcalSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HFShower = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        BscSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    )
)

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    oncePerEventMode = cms.untracked.bool(True),
    showMallocInfo = cms.untracked.bool(True),
    dump = cms.untracked.bool(True),
    ignoreTotal = cms.untracked.int32(1)
)

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789
process.rndmStore = cms.EDProducer("RandomEngineStateProducer")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('qcdpt_3000_3500_QGSP_BERT_EML.root')
)

# Event output
process.load("Configuration.EventContent.EventContent_cff")

process.o1 = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('simevent_qcdpt_3000_3500_QGSP_BERT_EML.root')
)

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.g4SimHits*process.caloSimHitStudy*process.rndmStore)
process.outpath = cms.EndPath(process.o1)
process.generator.pythiaHepMCVerbosity = False
process.generator.pythiaPylistVerbosity = 0
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_BERT_EML'

# process.g4SimHits.ECalSD.IgnoreTrackID      = True
# process.g4SimHits.HCalSD.IgnoreTrackID      = True
# process.g4SimHits.ECalSD.TimeSliceUnit      = 5
# process.g4SimHits.HCalSD.TimeSliceUnit      = 5

# process.g4SimHits.HCalSD.UseShowerLibrary   = True
# process.g4SimHits.HCalSD.UseParametrize     = False
# process.g4SimHits.HCalSD.UsePMTHits         = False
# process.g4SimHits.HCalSD.UseFibreBundleHits = False
# process.g4SimHits.HFShower.UseShowerLibrary = False
# process.g4SimHits.HFShower.UseHFGflash      = True
# process.g4SimHits.HFShower.TrackEM          = False
# process.g4SimHits.HFShower.OnlyLong         = True
# process.g4SimHits.HFShower.EminLibrary      = 0.0
