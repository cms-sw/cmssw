import FWCore.ParameterSet.Config as cms

from SimG4Core.Application.hectorParameter_cfi import *

## HF Raddam Dose Class in /SimG4CMS/Calo
from SimG4CMS.Calo.HFDarkeningParams_cff import *

## This object is used to customise g4SimHits for different running scenarios

common_heavy_suppression = cms.PSet(
    NeutronThreshold = cms.double(30.0),
    ProtonThreshold = cms.double(30.0),
    IonThreshold = cms.double(30.0)
)

common_maximum_time = cms.PSet(
    MaxTrackTime  = cms.double(500.0), # ns
    MaxTrackTimeForward = cms.double(2000.0), # ns
    MaxTimeNames  = cms.vstring(),
    MaxTrackTimes = cms.vdouble(),     # ns
    MaxZCentralCMS = cms.double(50.0), # m
    DeadRegions   = cms.vstring('QuadRegion','InterimRegion'),
    CriticalEnergyForVacuum = cms.double(2.0),   # MeV
    CriticalDensity         = cms.double(1e-15)  # g/cm3
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

g4SimHits = cms.EDProducer("OscarMTProducer",
    g4GeometryDD4hepSource = cms.bool(False),
    NonBeamEvent = cms.bool(False),
    G4EventManagerVerbosity = cms.untracked.int32(0),
    UseMagneticField = cms.bool(True),
    StoreRndmSeeds = cms.bool(False),
    RestoreRndmSeeds = cms.bool(False),
    PhysicsTablesDirectory = cms.untracked.string('PhysicsTables'),
    StorePhysicsTables = cms.untracked.bool(False),
    RestorePhysicsTables = cms.untracked.bool(False),
    UseParametrisedEMPhysics = cms.untracked.bool(True),
    CheckGeometry = cms.untracked.bool(False),
    G4CheckOverlap = cms.untracked.PSet(
        OutputBaseName = cms.string('2017'),
        MaterialFlag = cms.bool(True),
        GeomFlag = cms.bool(True),
        OverlapFlag = cms.bool(False),
        RegionFlag = cms.bool(True),  # if true - selection by G4Region name
        gdmlFlag = cms.bool(False),   # if true - dump gdml file
        Verbose = cms.bool(True),
        Tolerance = cms.double(0.0),
        Resolution = cms.int32(10000),
        ErrorThreshold = cms.int32(1),
        Level = cms.int32(1),
        Depth = cms.int32(3),        # -1 means check whatever depth
        PVname = cms.string(''),
        LVname = cms.string(''),
        NodeNames = cms.vstring('World')
    ),
    G4Commands = cms.vstring(),
    #G4Commands = cms.vstring('/process/em/UseGeneralProcess true'), # eneble G4 general process
    SteppingVerbosity = cms.untracked.int32(0),
    StepVerboseThreshold = cms.untracked.double(0.1), # in GeV
    VerboseEvents = cms.untracked.vint32(),
    VertexNumber  = cms.untracked.vint32(),
    VerboseTracks = cms.untracked.vint32(),
    FileNameField = cms.untracked.string(''),
    FileNameGDML = cms.untracked.string(''),
    FileNameRegions = cms.untracked.string(''),
    Watchers = cms.VPSet(),
    HepMCProductLabel = cms.InputTag("generatorSmeared"),
    theLHCTlinkTag = cms.InputTag("LHCTransport"),
    LHCTransport = cms.bool(False),
    CustomUIsession = cms.untracked.PSet(
        Type = cms.untracked.string("MessageLogger"), # alternatives: MessageLoggerThreadPrefix, FilePerThread
        ThreadPrefix = cms.untracked.string("W"),     # for MessageLoggerThreadPrefix
        ThreadFile = cms.untracked.string("sim_output_thread"), # for FilePerThread
    ),
    MagneticField = cms.PSet(
        UseLocalMagFieldManager = cms.bool(False),
        Verbosity = cms.bool(False),
        ConfGlobalMFM = cms.PSet(
            Volume = cms.string('OCMS'),
            OCMS = cms.PSet(
                Stepper = cms.string('G4DormandPrince745'),
                Type = cms.string('CMSIMField'),
                StepperParam = cms.PSet(
                    VacRegions = cms.vstring(),
#                   VacRegions = cms.vstring('DefaultRegionForTheWorld','BeamPipeVacuum','BeamPipeOutside'),
                    EnergyThTracker = cms.double(100000),  ## in GeV
                    RmaxTracker = cms.double(8000),        ## in mm
                    ZmaxTracker = cms.double(11000),       ## in mm
                    MaximumEpsilonStep = cms.untracked.double(0.01),
                    DeltaOneStep = cms.double(0.001),      ## in mm
                    DeltaOneStepTracker = cms.double(1e-4),## in mm
                    MaximumLoopCounts = cms.untracked.double(1000.0),
                    DeltaChord = cms.double(0.001),        ## in mm
                    DeltaChordTracker = cms.double(0.001), ## in mm
                    MinStep = cms.double(0.1),             ## in mm
                    DeltaIntersectionAndOneStep = cms.untracked.double(-1.0),
                    DeltaIntersection = cms.double(0.0001),     ## in mm
                    DeltaIntersectionTracker = cms.double(1e-6),## in mm
                    MaxStep = cms.double(150.),            ## in cm
                    MinimumEpsilonStep = cms.untracked.double(1e-05),
                    EnergyThSimple = cms.double(0.015),    ## in GeV
                    DeltaChordSimple = cms.double(0.1),    ## in mm
                    DeltaOneStepSimple = cms.double(0.1),  ## in mm
                    DeltaIntersectionSimple = cms.double(0.01),       ## in mm
                    MaxStepSimple = cms.double(50.),       ## in cm
                )
            )
        ),
        delta = cms.double(1.0)
    ),
    Physics = cms.PSet(
        common_maximum_time,
        # NOTE : if you want EM Physics only,
        #        please select "SimG4Core/Physics/DummyPhysics" for type
        #        and turn ON DummyEMPhysics
        #
        type = cms.string('SimG4Core/Physics/FTFP_BERT_EMM'),
        DummyEMPhysics = cms.bool(False),
        CutsPerRegion = cms.bool(True),
        CutsOnProton  = cms.bool(True),
        DefaultCutValue = cms.double(1.0), ## cuts in cm
        G4BremsstrahlungThreshold = cms.double(0.5), ## cut in GeV
        Verbosity = cms.untracked.int32(0),
        # 1 will print cuts as they get set from DD
        # 2 will do as 1 + will dump Geant4 table of cuts
        MonopoleCharge       = cms.untracked.int32(1),
        MonopoleDeltaRay     = cms.untracked.bool(True),
        MonopoleMultiScatter = cms.untracked.bool(False),
        MonopoleTransport    = cms.untracked.bool(True),
        MonopoleMass         = cms.untracked.double(0),
        ExoticaTransport     = cms.untracked.bool(False),
        ExoticaPhysicsSS     = cms.untracked.bool(False),
        RhadronPhysics       = cms.bool(False),
        DarkMPFactor         = cms.double(1.0),
        Region      = cms.string(''),
        TrackingCut = cms.bool(False),
        SRType      = cms.bool(True),
        FlagMuNucl  = cms.bool(False),
        FlagFluo    = cms.bool(False),
        EMPhysics   = cms.untracked.bool(True),
        HadPhysics  = cms.untracked.bool(True),
        FlagBERT    = cms.untracked.bool(False),
        EminFTFP    = cms.double(3.), # in GeV
        EmaxBERT    = cms.double(6.), # in GeV
        EminQGSP    = cms.double(12.), # in GeV
        EmaxFTFP    = cms.double(25.), # in GeV
        EmaxBERTpi  = cms.double(12.), # in GeV
        LowEnergyGflashEcal = cms.bool(False),
        LowEnergyGflashEcalEmax = cms.double(100),
        GflashEcal    = cms.bool(False),
        GflashHcal    = cms.bool(False),
        GflashEcalHad = cms.bool(False),
        GflashHcalHad = cms.bool(False),
        bField        = cms.double(3.8),
        energyScaleEB = cms.double(1.032),
        energyScaleEE = cms.double(1.024),
        ThermalNeutrons  = cms.untracked.bool(False),
        RusRoElectronEnergyLimit  = cms.double(0.0),
        RusRoEcalElectron         = cms.double(1.0),
        RusRoHcalElectron         = cms.double(1.0),
        RusRoMuonIronElectron     = cms.double(1.0),
        RusRoPreShowerElectron    = cms.double(1.0),
        RusRoCastorElectron       = cms.double(1.0),
        RusRoWorldElectron        = cms.double(1.0),
        ElectronStepLimit         = cms.bool(False),
        ElectronRangeTest         = cms.bool(False),
        PositronStepLimit         = cms.bool(False),
        ProtonRegionLimit         = cms.bool(False),
        PionRegionLimit           = cms.bool(False),
        LimitsPerRegion = cms.vstring('EcalRegion','HcalRegion'),
        EnergyLimitsE   = cms.vdouble(0.,0.0),
        EnergyLimitsH   = cms.vdouble(0.,0.0),
        EnergyFactorsE  = cms.vdouble(1.,0.0),
        EnergyRMSE      = cms.vdouble(0.0,0.0),
        MinStepLimit              = cms.double(1.0),
        ModifyTransportation      = cms.bool(False),
        ThresholdWarningEnergy    = cms.untracked.double(100.0),
        ThresholdImportantEnergy  = cms.untracked.double(250.0),
        ThresholdTrials           = cms.untracked.int32(10)
    ),
    Generator = cms.PSet(
        common_maximum_time,
        HectorEtaCut,
#        HepMCProductLabel = cms.InputTag('LHCTransport'),
        HepMCProductLabel = cms.InputTag('generatorSmeared'),
        ApplyPCuts = cms.bool(True),
        ApplyPtransCut = cms.bool(False),
        MinPCut = cms.double(0.04), ## the cut is in GeV 
        MaxPCut = cms.double(99999.0), ## the pmax=99.TeV 
        ApplyEtaCuts = cms.bool(True),
        MinEtaCut = cms.double(-5.5),
        MaxEtaCut = cms.double(5.5),
        RDecLenCut = cms.double(2.9), ## (cm) the cut on vertex radius
        LDecLenCut = cms.double(30.0), ## (cm) decay volume length
        ApplyPhiCuts = cms.bool(False),
        MinPhiCut = cms.double(-3.14159265359), ## (radians)
        MaxPhiCut = cms.double(3.14159265359),  ## according to CMS conventions
        ApplyLumiMonitorCuts = cms.bool(False), ## primary for lumi monitors
        Verbosity = cms.untracked.int32(0),
        PDGselection = cms.PSet(
            PDGfilterSel = cms.bool(False),        ## filter out unwanted particles
            PDGfilter = cms.vint32(21,1,2,3,4,5,6) ## list of unwanted particles (gluons and quarks)
        )
    ),
    RunAction = cms.PSet(
        StopFile = cms.string('')
    ),
    EventAction = cms.PSet(
        debug = cms.untracked.bool(False),
        StopFile = cms.string(''),
        PrintRandomSeed = cms.bool(False),
        CollapsePrimaryVertices = cms.bool(False)
    ),
    StackingAction = cms.PSet(
        common_heavy_suppression,
        common_maximum_time,
        KillDeltaRay  = cms.bool(False),
        TrackNeutrino = cms.bool(False),
        KillHeavy     = cms.bool(False),
        KillGamma     = cms.bool(True),
        GammaThreshold = cms.double(0.0001), ## (MeV)
        SaveFirstLevelSecondary = cms.untracked.bool(False),
        SavePrimaryDecayProductsAndConversionsInTracker = cms.untracked.bool(False),
        SavePrimaryDecayProductsAndConversionsInCalo = cms.untracked.bool(False),
        SavePrimaryDecayProductsAndConversionsInMuon = cms.untracked.bool(False),
        SaveAllPrimaryDecayProductsAndConversions = cms.untracked.bool(True),
        RusRoGammaEnergyLimit  = cms.double(5.0), ## (MeV)
        RusRoEcalGamma         = cms.double(0.3),
        RusRoHcalGamma         = cms.double(0.3),
        RusRoMuonIronGamma     = cms.double(0.3),
        RusRoPreShowerGamma    = cms.double(0.3),
        RusRoCastorGamma       = cms.double(0.3),
        RusRoWorldGamma        = cms.double(0.3),
        RusRoNeutronEnergyLimit  = cms.double(10.0), ## (MeV)
        RusRoEcalNeutron         = cms.double(0.1),
        RusRoHcalNeutron         = cms.double(0.1),
        RusRoMuonIronNeutron     = cms.double(0.1),
        RusRoPreShowerNeutron    = cms.double(0.1),
        RusRoCastorNeutron       = cms.double(0.1),
        RusRoWorldNeutron        = cms.double(0.1),
        RusRoProtonEnergyLimit  = cms.double(0.0),
        RusRoEcalProton         = cms.double(1.0),
        RusRoHcalProton         = cms.double(1.0),
        RusRoMuonIronProton     = cms.double(1.0),
        RusRoPreShowerProton    = cms.double(1.0),
        RusRoCastorProton       = cms.double(1.0),
        RusRoWorldProton        = cms.double(1.0)
    ),
    TrackingAction = cms.PSet(
        DetailedTiming = cms.untracked.bool(False),
        CheckTrack = cms.untracked.bool(False)
    ),
    SteppingAction = cms.PSet(
        common_maximum_time,
        EkinNames               = cms.vstring(),
        EkinThresholds          = cms.vdouble(),
        EkinParticles           = cms.vstring()
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
        UseFineCaloID     = cms.bool(False),
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
        ScaleRadLength  = cms.untracked.double(1.0),
        StoreLayerTimeSim = cms.untracked.bool(False),
        AgeingWithSlopeLY = cms.untracked.bool(False)
        ),
    HCalSD = cms.PSet(
        common_UseLuminosity,
        UseBirkLaw                = cms.bool(True),
        BirkC3                    = cms.double(1.75),
        BirkC2                    = cms.double(0.142),
        BirkC1                    = cms.double(0.0060),
        UseShowerLibrary          = cms.bool(True),
        UseParametrize            = cms.bool(False),
        UsePMTHits                = cms.bool(False),
        UseFibreBundleHits        = cms.bool(False),
        TestNumberingScheme       = cms.bool(False),
        doNeutralDensityFilter    = cms.bool(False),
        EminHitHB                 = cms.double(0.0),
        EminHitHE                 = cms.double(0.0),
        EminHitHO                 = cms.double(0.0),
        EminHitHF                 = cms.double(0.0),
        BetaThreshold             = cms.double(0.7),
        TimeSliceUnit             = cms.double(1),
        IgnoreTrackID             = cms.bool(False),
        HBDarkening               = cms.bool(False),
        HEDarkening               = cms.bool(False),
        HFDarkening               = cms.bool(False),
        UseHF                     = cms.untracked.bool(True),
        ForTBH2                   = cms.untracked.bool(False),
        ForTBHCAL                 = cms.untracked.bool(False),
        UseLayerWt                = cms.untracked.bool(False),
        WtFile                    = cms.untracked.string('None'),
        TestNS                    = cms.untracked.bool(False),
        HFDarkeningParameterBlock = HFDarkeningParameterBlock
    ),
    CaloTrkProcessing = cms.PSet(
        TestBeam   = cms.bool(False),
        EminTrack  = cms.double(0.01),
        PutHistory = cms.bool(False),
        DoFineCalo = cms.bool(False),
        EminFineTrack = cms.double(10000.0),
        EminFinePhoton = cms.double(5000.0)
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
    MtdSD = cms.PSet(
        Verbosity = cms.untracked.int32(0),
        TimeSliceUnit    = cms.double(0.01), #stepping = 10 ps (for timing)
        IgnoreTrackID    = cms.bool(False),
        EminHit          = cms.double(0.0),
        CheckID          = cms.untracked.bool(True),
    ),
    HGCSD = cms.PSet(
        Verbosity        = cms.untracked.int32(0),
        TimeSliceUnit    = cms.double(0.001), #stepping = 1 ps (for timing)
        IgnoreTrackID    = cms.bool(False),
        EminHit          = cms.double(0.0),
        FiducialCut      = cms.bool(False),
        DistanceFromEdge = cms.double(1.0),
        StoreAllG4Hits   = cms.bool(False),
        RejectMouseBite  = cms.bool(False),
        RotatedWafer     = cms.bool(False),
        CornerMinMask    = cms.int32(0),
        WaferAngles      = cms.untracked.vdouble(90.0,30.0),
        WaferSize        = cms.untracked.double(123.7),
        MouseBite        = cms.untracked.double(2.5),
        CheckID          = cms.untracked.bool(True),
    ),
    HGCScintSD = cms.PSet(
        Verbosity        = cms.untracked.int32(0),
        EminHit          = cms.double(0.0),
        UseBirkLaw       = cms.bool(True),
        BirkC3           = cms.double(1.75),
        BirkC2           = cms.double(0.142),
        BirkC1           = cms.double(0.0052),
        FiducialCut      = cms.bool(False),
        DistanceFromEdge = cms.double(1.0),
        StoreAllG4Hits   = cms.bool(False),
    ),
    HFNoseSD = cms.PSet(
        Verbosity        = cms.untracked.int32(0),
        TimeSliceUnit    = cms.double(0.001), #stepping = 1 ps (for timing)
        IgnoreTrackID    = cms.bool(False),
        EminHit          = cms.double(0.0),
        FiducialCut      = cms.bool(False),
        DistanceFromEdge = cms.double(1.0),
        StoreAllG4Hits   = cms.bool(False),
        RejectMouseBite  = cms.bool(False),
        RotatedWafer     = cms.bool(False),
        CornerMinMask    = cms.int32(0),
        WaferAngles      = cms.untracked.vdouble(90.0,30.0),
        CheckID          = cms.untracked.bool(True),
    ),
    TotemRPSD = cms.PSet(
        Verbosity = cms.int32(0)
    ),
    TotemSD = cms.PSet(
        Verbosity = cms.untracked.int32(0)
    ),
    TotemT2ScintSD = cms.PSet(
        UseBirkLaw    = cms.bool(True),
        BirkC3        = cms.double(1.75),
        BirkC2        = cms.double(0.142),
        BirkC1        = cms.double(0.006),
        TimeSliceUnit = cms.double(1),
        IgnoreTrackID = cms.bool(False),
    ),
    PPSDiamondSD = cms.PSet(
        Verbosity = cms.int32(0)
    ),
    PPSPixelSD = cms.PSet(
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
    HGCalTestBeamSD = cms.PSet(
        Material   = cms.string('Scintillator'),
        UseBirkLaw = cms.bool(False),
        BirkC1 = cms.double(0.013),
        BirkC3 = cms.double(1.75),
        BirkC2 = cms.double(0.0568),
    ),
    HcalTB06BeamSD = cms.PSet(
        UseBirkLaw = cms.bool(False),
        BirkC1 = cms.double(0.013),
        BirkC3 = cms.double(1.75),
        BirkC2 = cms.double(0.0568)
    ),
    AHCalSD = cms.PSet(
        UseBirkLaw      = cms.bool(True),
        BirkC3          = cms.double(1.75),
        BirkC2          = cms.double(0.142),
        BirkC1          = cms.double(0.0052),
        EminHit         = cms.double(0.0),
        TimeSliceUnit   = cms.double(1),
        IgnoreTrackID   = cms.bool(False),
    ),
)
##
## Change the HFShowerLibrary file from Run 2
##
from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toModify( g4SimHits.HFShowerLibrary, FileName = 'SimG4CMS/Calo/data/HFShowerLibrary_npmt_noatt_eta4_16en_v4.root' )
run2_common.toModify( g4SimHits.HFShower, ProbMax = 0.5)

##
## Change HCAL numbering scheme in 2017
##
from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
run2_HCAL_2017.toModify( g4SimHits, HCalSD = dict( TestNumberingScheme = True ) )

##
## Disable Castor from Run 3
##
from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify( g4SimHits, CastorSD = dict( useShowerLibrary = False ) ) 

##
## Change ECAL time slices
##
from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toModify( g4SimHits.ECalSD,
                             StoreLayerTimeSim = cms.untracked.bool(True),
                             TimeSliceUnit = cms.double(0.001) )
##
## DD4Hep migration
##
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
dd4hep.toModify( g4SimHits, g4GeometryDD4hepSource = True )
