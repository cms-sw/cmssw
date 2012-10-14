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
        ProbMax         = cms.double(0.7268),
        CFibre          = cms.double(0.5),
        PEPerGeV        = cms.double(0.25),
        TrackEM         = cms.bool(False),
        UseShowerLibrary= cms.bool(False),
        UseR7600UPMT    = cms.bool(False),
        EminLibrary     = cms.double(5.0),
        RefIndex        = cms.double(1.459),
        Lambda1         = cms.double(280.0),
        Lambda2         = cms.double(700.0),
        Aperture        = cms.double(0.33),
        ApertureTrapped = cms.double(0.22),
        Gain            = cms.double(0.33),
        CheckSurvive    = cms.bool(False)
    ),
    HFShowerLibrary = cms.PSet(
        FileName        = cms.FileInPath('SimG4CMS/Calo/data/hfshowerlibrary_lhep_140_edm.root'),
        BackProbability = cms.double(0.2),
        TreeEMID        = cms.string('emParticles'),
        TreeHadID       = cms.string('hadParticles'),
        Verbosity       = cms.untracked.bool(False),
        BranchPost      = cms.untracked.string('_R.obj'),
        BranchEvt       = cms.untracked.string('HFShowerLibraryEventInfos_hfshowerlib_HFShowerLibraryEventInfo'),
        BranchPre       = cms.untracked.string('HFShowerPhotons_hfshowerlib_')
    ),
    HFShowerPMT = cms.PSet(
        PEPerGeVPMT     = cms.double(1.0),
        RefIndex        = cms.double(1.52),
        Lambda1         = cms.double(280.0),
        Lambda2         = cms.double(700.0),
        Aperture        = cms.double(0.66),
        ApertureTrapped = cms.double(0.22),
        Gain            = cms.double(0.33),
        CheckSurvive    = cms.bool(False)
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
        EminHitHB = cms.double(0.0),
        EminHitHE = cms.double(0.0),
        EminHitHO = cms.double(0.0),
        EminHitHF = cms.double(0.0),
        BirkC3 = cms.double(1.75),
        BirkC2 = cms.double(0.142),
        BirkC1 = cms.double(0.0052),
        UseBirkLaw = cms.bool(True)
    ),
    CaloTrkProcessing = cms.PSet(
        TestBeam = cms.bool(False),
        EminTrack = cms.double(0.01)
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
        TmaxHit   = cms.double(1000.0),
        TmaxHits  = cms.vdouble(1000.0,1000.0,1000.0,1000.0),
	HCNames   = cms.vstring('EcalHitsEB','EcalHitsEE','EcalHitsES','HcalHits'),
        EminHits  = cms.vdouble(0.0,0.0,0.0,0.0)
    ),
    UseMagneticField = cms.bool(True)
)

process.p1 = cms.Path(process.m)


