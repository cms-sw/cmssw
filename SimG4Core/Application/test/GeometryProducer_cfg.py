import FWCore.ParameterSet.Config as cms

process = cms.Process("GEOM")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load('Configuration.Geometry.GeometryExtended2015_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")

process.Tracer = cms.Service("Tracer")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.MessageLogger.destinations = cms.untracked.vstring("geomprod.txt")

common_heavy_suppression = cms.PSet(
    NeutronThreshold = cms.double(30.0),
    ProtonThreshold = cms.double(30.0),
    IonThreshold = cms.double(30.0)
)

common_maximum_time = cms.PSet(
    MaxTrackTime  = cms.double(500.0),
    MaxTimeNames  = cms.vstring('ZDCRegion'),
    MaxTrackTimes = cms.vdouble(2000.0),
    DeadRegions   = cms.vstring('QuadRegion','InterimRegion'),
    CriticalEnergyForVacuum = cms.double(2.0),
    CriticalDensity         = cms.double(1e-15)
)

common_UsePMT = cms.PSet(
    UseR7600UPMT  = cms.bool(False)
)

common_UseHF = cms.PSet(
    Lambda1       = cms.double(280.0),
    Lambda2       = cms.double(700.0),
    Gain          = cms.double(0.33),
    CheckSurvive  = cms.bool(False),
    FibreR        = cms.untracked.double(0.3)
)

common_UseLuminosity = cms.PSet(
    InstLuminosity  = cms.double(0.),   
    DelivLuminosity = cms.double(5000.)
)

process.m = cms.EDProducer("GeometryProducer",
    UseSensitiveDetectors = cms.bool(True),
    UseMagneticField = cms.bool(True),
    MagneticField = cms.PSet(
        UseLocalMagFieldManager = cms.bool(False),
        Verbosity = cms.untracked.bool(False),
        ConfGlobalMFM = cms.PSet(
            Volume = cms.string('OCMS'),
            OCMS = cms.PSet(
                Stepper = cms.string('G4ClassicalRK4'),
                Type = cms.string('CMSIMField'),
                StepperParam = cms.PSet(
                    MaximumEpsilonStep = cms.untracked.double(0.01), ## in mm
                    DeltaOneStep = cms.double(0.001), ## in mm
                    MaximumLoopCounts = cms.untracked.double(1000.0),
                    DeltaChord = cms.double(0.001), ## in mm
                    MinStep = cms.double(0.1), ## in mm
                    DeltaIntersectionAndOneStep = cms.untracked.double(-1.0),
                    DeltaIntersection = cms.double(0.0001), ## in mm
                    MinimumEpsilonStep = cms.untracked.double(1e-05) ## in mm
                )
            )
        ),
        delta = cms.double(1.0)
    ),
    TrackerSD = cms.PSet(
        ZeroEnergyLoss = cms.bool(False),
        PrintHits = cms.bool(False),
        ElectronicSigmaInNanoSeconds = cms.double(12.06),
        NeverAccumulate = cms.bool(False),
        EnergyThresholdForPersistencyInGeV = cms.double(0.2),
        EnergyThresholdForHistoryInGeV = cms.double(0.05)
    ),
    MuonSD = cms.PSet(
        EnergyThresholdForPersistency = cms.double(1.0),
        PrintHits = cms.bool(False),
        AllMuonsPersistent = cms.bool(True)
    ),
    CaloSD = cms.PSet(
        common_heavy_suppression,
        SuppressHeavy = cms.bool(False),
        EminTrack = cms.double(1.0),
        TmaxHit   = cms.double(1000.0),
        HCNames   = cms.vstring('EcalHitsEB','EcalHitsEE','EcalHitsES','HcalHits','ZDCHITS'),
        EminHits  = cms.vdouble(0.015,0.010,0.0,0.0,0.0),
        EminHitsDepth = cms.vdouble(0.0,0.0,0.0,0.0,0.0),
        TmaxHits  = cms.vdouble(500.0,500.0,500.0,500.0,2000.0),
        UseResponseTables = cms.vint32(0,0,0,0,0),
        BeamPosition      = cms.double(0.0),
        CorrectTOFBeam    = cms.bool(False),
        DetailedTiming    = cms.untracked.bool(False),
        UseMap            = cms.untracked.bool(False),
        Verbosity         = cms.untracked.int32(0),
        CheckHits         = cms.untracked.int32(25)
    ),
    CaloResponse = cms.PSet(
        UseResponseTable  = cms.bool(True),
        ResponseScale     = cms.double(1.0),
        ResponseFile      = cms.FileInPath('SimG4CMS/Calo/data/responsTBpim50.dat')
    ),
    ECalSD = cms.PSet(
        common_UseLuminosity,
        UseBirkLaw      = cms.bool(True),
        BirkL3Parametrization = cms.bool(True),
        BirkSlope       = cms.double(0.253694),
        BirkCut         = cms.double(0.1),
        BirkC1          = cms.double(0.03333),
        BirkC3          = cms.double(1.0),
        BirkC2          = cms.double(0.0),
        SlopeLightYield = cms.double(0.02),
        StoreSecondary  = cms.bool(False),
        TimeSliceUnit   = cms.double(1),
        IgnoreTrackID   = cms.bool(False),
        XtalMat         = cms.untracked.string('E_PbWO4'),
        TestBeam        = cms.untracked.bool(False),
        NullNumbering   = cms.untracked.bool(False),
        StoreRadLength  = cms.untracked.bool(False),
        AgeingWithSlopeLY  = cms.untracked.bool(False)
    ),
    HCalSD = cms.PSet(
        common_UseLuminosity,
        UseBirkLaw          = cms.bool(True),
        BirkC3              = cms.double(1.75),
        BirkC2              = cms.double(0.142),
        BirkC1              = cms.double(0.0052),
        UseShowerLibrary    = cms.bool(True),
        UseParametrize      = cms.bool(False),
        UsePMTHits          = cms.bool(False),
        UseFibreBundleHits  = cms.bool(False),
        TestNumberingScheme = cms.bool(False),
        EminHitHB           = cms.double(0.0),
        EminHitHE           = cms.double(0.0),
        EminHitHO           = cms.double(0.0),
        EminHitHF           = cms.double(0.0),
        BetaThreshold       = cms.double(0.7),
        TimeSliceUnit       = cms.double(1),
        IgnoreTrackID       = cms.bool(False),
        HEDarkening         = cms.bool(False),
        HFDarkening         = cms.bool(False),
        UseHF               = cms.untracked.bool(True),
        ForTBH2             = cms.untracked.bool(False),
        UseLayerWt          = cms.untracked.bool(False),
        WtFile              = cms.untracked.string('None')
    ),
    CaloTrkProcessing = cms.PSet(
        TestBeam   = cms.bool(False),
        EminTrack  = cms.double(0.01),
        PutHistory = cms.bool(False)
    ),
    HFShower = cms.PSet(
        common_UsePMT,
        common_UseHF,
        ProbMax           = cms.double(1.0),
        CFibre            = cms.double(0.5),
        PEPerGeV          = cms.double(0.31),
        TrackEM           = cms.bool(False),
        UseShowerLibrary  = cms.bool(True),
        UseHFGflash       = cms.bool(False),
        EminLibrary       = cms.double(0.0),
        OnlyLong          = cms.bool(True),
        LambdaMean        = cms.double(350.0),
        ApplyFiducialCut  = cms.bool(True),
        RefIndex          = cms.double(1.459),
        Aperture          = cms.double(0.33),
        ApertureTrapped   = cms.double(0.22),
        CosApertureTrapped= cms.double(0.5),
        SinPsiMax         = cms.untracked.double(0.5),
        ParametrizeLast   = cms.untracked.bool(False)
    ),
    HFShowerLibrary = cms.PSet(
        FileName        = cms.FileInPath('SimG4CMS/Calo/data/HFShowerLibrary_oldpmt_noatt_eta4_16en_v3.root'),
        BackProbability = cms.double(0.2),
        TreeEMID        = cms.string('emParticles'),
        TreeHadID       = cms.string('hadParticles'),
        Verbosity       = cms.untracked.bool(False),
        ApplyFiducialCut= cms.bool(True),
        BranchPost      = cms.untracked.string(''),
        BranchEvt       = cms.untracked.string(''),
        BranchPre       = cms.untracked.string('')
    ),
    HFShowerPMT = cms.PSet(
        common_UsePMT,
        common_UseHF,
        PEPerGeVPMT       = cms.double(1.0),
        RefIndex          = cms.double(1.52),
        Aperture          = cms.double(0.99),
        ApertureTrapped   = cms.double(0.22),
        CosApertureTrapped= cms.double(0.5),
        SinPsiMax         = cms.untracked.double(0.5)
    ),
    HFShowerStraightBundle = cms.PSet(
        common_UsePMT,
        common_UseHF,
        FactorBundle      = cms.double(1.0),
        RefIndex          = cms.double(1.459),
        Aperture          = cms.double(0.33),
        ApertureTrapped   = cms.double(0.22),
        CosApertureTrapped= cms.double(0.5),
        SinPsiMax         = cms.untracked.double(0.5)
    ),
    HFShowerConicalBundle = cms.PSet(
        common_UsePMT,
        common_UseHF,
        FactorBundle      = cms.double(1.0),
        RefIndex          = cms.double(1.459),
        Aperture          = cms.double(0.33),
        ApertureTrapped   = cms.double(0.22),
        CosApertureTrapped= cms.double(0.5),
        SinPsiMax         = cms.untracked.double(0.5)
    ),
    HFGflash = cms.PSet(
        BField          = cms.untracked.double(3.8),
        WatcherOn       = cms.untracked.bool(True),
        FillHisto       = cms.untracked.bool(True)
    ),
    CastorSD = cms.PSet(
        useShowerLibrary               = cms.bool(True),
        minEnergyInGeVforUsingSLibrary = cms.double(1.0),
        nonCompensationFactor          = cms.double(0.817),
        Verbosity                      = cms.untracked.int32(0)
    ),
    CastorShowerLibrary =  cms.PSet(
        FileName  = cms.FileInPath('SimG4CMS/Forward/data/CastorShowerLibrary_CMSSW500_Standard.root'),
        BranchEvt = cms.untracked.string('hadShowerLibInfo.'),
        BranchEM  = cms.untracked.string('emParticles.'),
        BranchHAD = cms.untracked.string('hadParticles.'),
        Verbosity = cms.untracked.bool(False)
    ),
    BHMSD = cms.PSet(
         Verbosity = cms.untracked.int32(0)
    ),
    FastTimerSD = cms.PSet(
        Verbosity = cms.untracked.int32(0)
    ),
    HGCSD = cms.PSet(
        Verbosity        = cms.untracked.int32(0),
        TimeSliceUnit    = cms.double(1),
        IgnoreTrackID    = cms.bool(False),
        EminHit          = cms.double(0.0),
        CheckID          = cms.untracked.bool(True),
    ),
    TotemSD = cms.PSet(
        Verbosity = cms.untracked.int32(0)
    ),
    ZdcSD = cms.PSet(
        Verbosity = cms.int32(0),
        UseShowerLibrary = cms.bool(True),
        UseShowerHits = cms.bool(False),
        FiberDirection = cms.double(45.0),
        ZdcHitEnergyCut = cms.double(10.0)
    ),
    ZdcShowerLibrary = cms.PSet(
        Verbosity = cms.untracked.int32(0)
    ),
    FP420SD = cms.PSet(
        Verbosity = cms.untracked.int32(2)
    ),
    BscSD = cms.PSet(
        Verbosity = cms.untracked.int32(0)
    ),
    PltSD = cms.PSet(
        EnergyThresholdForPersistencyInGeV = cms.double(0.2),
        EnergyThresholdForHistoryInGeV = cms.double(0.05)
    ),
    Bcm1fSD = cms.PSet(
        EnergyThresholdForPersistencyInGeV = cms.double(0.010),
        EnergyThresholdForHistoryInGeV = cms.double(0.005)
    ),
    HcalTB02SD = cms.PSet(
        UseBirkLaw = cms.untracked.bool(False),
        BirkC1 = cms.untracked.double(0.013),
        BirkC3 = cms.untracked.double(1.75),
        BirkC2 = cms.untracked.double(0.0568)
    ),
    EcalTBH4BeamSD = cms.PSet(
        UseBirkLaw = cms.bool(False),
        BirkC1 = cms.double(0.013),
        BirkC3 = cms.double(1.75),
        BirkC2 = cms.double(0.0568)
    ),
    HcalTB06BeamSD = cms.PSet(
        UseBirkLaw = cms.bool(False),
        BirkC1 = cms.double(0.013),
        BirkC3 = cms.double(1.75),
        BirkC2 = cms.double(0.0568)
    )
)

process.p1 = cms.Path(process.m)

