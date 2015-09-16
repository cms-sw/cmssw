import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load("Geometry.CMSCommonData.cmsExtendedGeometryHFLibraryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Geometry.HcalCommonData.hcalParameters_cfi")
process.load("Geometry.HcalCommonData.hcalDDDSimConstants_cfi")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("SimG4Core.Application.g4SimHits_cfi")
process.load("SimG4CMS.Calo.HFPMTHitAnalyzer_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('CaloSim', 
        'EcalSim', 'G4cerr', 'G4cout',
        'HcalSim', 'HFShower'),
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        CaloSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EcalSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        G4cerr = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        G4cout = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        HcalSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HFShower = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    )
)

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run1_mc', '')

process.Timing = cms.Service("Timing")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)

process.source = cms.Source("EmptySource",
    firstRun        = cms.untracked.uint32(1),
    firstEvent      = cms.untracked.uint32(1)
)

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(211),
        MinEta = cms.double(3.25),
        MaxEta = cms.double(3.50),
        MinPhi = cms.double(-3.1415926),
        MaxPhi = cms.double(3.1415926),
        MinE   = cms.double(1000.00),
        MaxE   = cms.double(1000.00)
    ),
    Verbosity       = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(False)
)

process.o1 = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('simevent.root')
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('HFLibrary.root')
)

process.common_maximum_timex = cms.PSet(
    MaxTrackTime  = cms.double(500.0),
    MaxTimeNames  = cms.vstring(),
    MaxTrackTimes = cms.vdouble()
)

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.g4SimHits*process.hfPMTHitAnalyzer)
process.outpath = cms.EndPath(process.o1)
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_FTFP_BERT_EML'
process.g4SimHits.Physics.DefaultCutValue   = 0.1
process.g4SimHits.HCalSD.UseShowerLibrary   = True
process.g4SimHits.HCalSD.UseParametrize     = False
process.g4SimHits.HCalSD.UsePMTHits         = False
process.g4SimHits.HCalSD.UseFibreBundleHits = False
process.g4SimHits.HFShower.UseShowerLibrary = False
process.g4SimHits.HFShower.UseHFGflash      = False
process.g4SimHits.HFShower.TrackEM          = False
process.g4SimHits.HFShower.OnlyLong         = False
process.g4SimHits.HFShower.EminLibrary      = 0.0
process.g4SimHits.HFShower.ApplyFiducialCut = True
process.g4SimHits.HFShowerLibrary.FileName  = 'SimG4CMS/Calo/data/HFShowerLibrary_npmt_eta4_16en.root'
process.g4SimHits.HFShowerLibrary.BranchPost= ''
process.g4SimHits.HFShowerLibrary.BranchPre = ''
process.g4SimHits.HFShowerLibrary.BranchEvt = ''
