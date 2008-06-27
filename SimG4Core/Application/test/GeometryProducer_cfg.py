import FWCore.ParameterSet.Config as cms

process = cms.Process("GEOM")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

#Geometry
#
process.load("Configuration.StandardSequences.Geometry_cff")

#Magnetic Field
process.load("Configuration.StandardSequences.MagneticField_cff")

process.Timing = cms.Service("Timing")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(0)
)
process.source = cms.Source("EmptySource")

process.common_heavy_suppression = cms.PSet(
    NeutronThreshold = cms.double(30.0),
    ProtonThreshold = cms.double(30.0),
    IonThreshold = cms.double(30.0)
)
process.m = cms.EDProducer("GeometryProducer",
    TrackerSD = cms.PSet(
        ZeroEnergyLoss = cms.bool(False),
        PrintHits = cms.bool(False),
        ElectronicSigmaInNanoSeconds = cms.double(12.06),
        NeverAccumulate = cms.bool(False),
        EnergyThresholdForPersistencyInGeV = cms.double(0.2),
        EnergyThresholdForHistoryInGeV = cms.double(0.05)
    ),
    HcalTB06BeamSD = cms.PSet(
        UseBirkLaw = cms.bool(False),
        BirkC1 = cms.double(0.013),
        BirkC3 = cms.double(1.75),
        BirkC2 = cms.double(0.0568)
    ),
    FP420SD = cms.PSet(
        Verbosity = cms.untracked.int32(2)
    ),
    HcalTB02SD = cms.PSet(
        UseBirkLaw = cms.untracked.bool(False),
        BirkC1 = cms.untracked.double(0.013),
        BirkC3 = cms.untracked.double(1.75),
        BirkC2 = cms.untracked.double(0.0568)
    ),
    HFShower = cms.PSet(
        TrackEM = cms.bool(False),
        CFibre = cms.double(0.5),
        PEPerGeV = cms.double(0.25),
        ProbMax = cms.double(0.7268),
        PEPerGeVPMT = cms.double(1.0)
    ),
    MagneticField = cms.PSet(
        delta = cms.double(1.0)
    ),
    ECalSD = cms.PSet(
        TestBeam = cms.untracked.bool(False),
        BirkL3Parametrization = cms.bool(True),
        BirkCut = cms.double(0.1),
        BirkC1 = cms.double(0.03333),
        BirkC3 = cms.double(1.0),
        BirkC2 = cms.double(0.0),
        SlopeLightYield = cms.double(0.02),
        UseBirkLaw = cms.bool(True),
        BirkSlope = cms.double(0.253694)
    ),
    ZdcSD = cms.PSet(
        Verbosity = cms.int32(0),
        FiberDirection = cms.double(0.0)
    ),
    UseSensitiveDetectors = cms.bool(True),
    EcalTBH4BeamSD = cms.PSet(
        UseBirkLaw = cms.bool(False),
        BirkC1 = cms.double(0.013),
        BirkC3 = cms.double(1.75),
        BirkC2 = cms.double(0.0568)
    ),
    CastorSD = cms.PSet(
        Verbosity = cms.untracked.int32(0)
    ),
    BscSD = cms.PSet(
        Verbosity = cms.untracked.int32(0)
    ),
    TotemSD = cms.PSet(
        Verbosity = cms.untracked.int32(0)
    ),
    MuonSD = cms.PSet(
        EnergyThresholdForPersistency = cms.double(1.0),
        PrintHits = cms.bool(False),
        AllMuonsPersistent = cms.bool(False)
    ),
    HCalSD = cms.PSet(
        BetaThreshold = cms.double(0.7),
        TestNumberingScheme = cms.bool(False),
        UsePMTHits = cms.bool(False),
        UseParametrize = cms.bool(False),
        ForTBH2 = cms.untracked.bool(False),
        WtFile = cms.untracked.string('None'),
        UseHF = cms.untracked.bool(True),
        UseLayerWt = cms.untracked.bool(False),
        UseShowerLibrary = cms.bool(True),
        BirkC3 = cms.double(1.75),
        BirkC2 = cms.double(0.142),
        BirkC1 = cms.double(0.0052),
        UseBirkLaw = cms.bool(True)
    ),
    CaloTrkProcessing = cms.PSet(
        TestBeam = cms.bool(False),
        EminTrack = cms.double(0.01)
    ),
    HFShowerLibrary = cms.PSet(
        BranchPost = cms.untracked.string('_R.obj'),
        BranchEvt = cms.untracked.string('HFShowerLibraryEventInfos_hfshowerlib_HFShowerLibraryEventInfo'),
        TreeHadID = cms.string('hadParticles'),
        Verbosity = cms.untracked.bool(False),
        BackProbability = cms.double(0.2),
        FileName = cms.FileInPath('SimG4CMS/Calo/data/hfshowerlibrary_lhep_140_edm.root'),
        TreeEMID = cms.string('emParticles'),
        BranchPre = cms.untracked.string('HFShowerPhotons_hfshowerlib_')
    ),
    CaloSD = cms.PSet(
        process.common_heavy_suppression,
        SuppressHeavy = cms.bool(False),
        DetailedTiming = cms.untracked.bool(False),
        Verbosity = cms.untracked.int32(0),
        CheckHits = cms.untracked.int32(25),
        BeamPosition = cms.untracked.double(0.0),
        CorrectTOFBeam = cms.untracked.bool(False),
        UseMap = cms.untracked.bool(True),
        EminTrack = cms.double(1.0),
        TmaxHit = cms.double(1000.0)
    ),
    UseMagneticField = cms.bool(True),
    HFCherenkov = cms.PSet(
        RefIndex = cms.double(1.459),
        Gain = cms.double(0.33),
        Aperture = cms.double(0.33),
        CheckSurvive = cms.bool(False),
        Lambda1 = cms.double(280.0),
        Lambda2 = cms.double(700.0),
        ApertureTrapped = cms.double(0.22)
    )
)

process.p1 = cms.Path(process.m)


