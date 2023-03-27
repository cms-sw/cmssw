import FWCore.ParameterSet.Config as cms

from SimG4Core.Application.hectorParameter_cfi import *

## HF Raddam Dose Class in /SimG4CMS/Calo
from SimG4CMS.Calo.HFDarkeningParams_cff import *

## HF shower parameters
from Geometry.HcalSimData.HFParameters_cff import *

## Modification needed for H2 TestBeam studies
from Configuration.Eras.Modifier_h2tb_cff import h2tb

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

h2tb.toModify(common_maximum_time,
    MaxTrackTime = cms.double(1000.0),
    DeadRegions  = cms.vstring()
)

common_UsePMT = cms.PSet(
    UseR7600UPMT  = cms.bool(False)
)

common_UseHF = cms.PSet(
    Lambda1       = cms.double(280.0),
    Lambda2       = cms.double(700.0),
    Gain          = cms.double(0.33),
    CheckSurvive  = cms.bool(False),
    FibreR        = cms.double(0.3)
)

common_UseLuminosity = cms.PSet(
    InstLuminosity  = cms.double(0.),
    DelivLuminosity = cms.double(5000.)
)

common_MCtruth = cms.PSet(
    DoFineCalo = cms.bool(False),
    SaveCaloBoundaryInformation = cms.bool(False),
    # currently unused; left in place for future studies
    EminFineTrack = cms.double(10000.0),
    FineCaloNames = cms.vstring('ECAL', 'HCal', 'HGCal', 'HFNoseVol', 'VCAL'),
    FineCaloLevels = cms.vint32(4, 4, 8, 3, 3),
    UseFineCalo = cms.vint32(2, 3),
)

## enable fine calorimeter functionality: must occur *before* common PSet is used below
from Configuration.ProcessModifiers.fineCalo_cff import fineCalo
fineCalo.toModify(common_MCtruth,
    DoFineCalo = True,
    UseFineCalo = [2],
    EminFineTrack = 0.0,
)

## enable CaloBoundary information for all Phase2 workflows
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify(common_MCtruth,
        SaveCaloBoundaryInformation = True
)

g4SimHits = cms.EDProducer("OscarMTProducer",
    g4GeometryDD4hepSource = cms.bool(False),
    NonBeamEvent = cms.bool(False),
    EventVerbose = cms.int32(0),
    UseG4EventManager = cms.bool(True),
    UseMagneticField = cms.bool(True),
    UseCommandBaseScorer = cms.bool(False),
    StoreRndmSeeds = cms.bool(False),
    RestoreRndmSeeds = cms.bool(False),
    PhysicsTablesDirectory = cms.untracked.string('PhysicsTables'),
    StorePhysicsTables = cms.untracked.bool(False),
    RestorePhysicsTables = cms.untracked.bool(False),
    UseParametrisedEMPhysics = cms.untracked.bool(True),
    ThresholdForGeometryExceptions = cms.double(0.1), ## in GeV
    TraceExceptions = cms.bool(False),
    DefaultVoxelDensity = cms.double(2.0),
    VoxelRegions = cms.vstring(),
    VoxelDensityPerRegion = cms.vdouble(),
    CheckGeometry = cms.untracked.bool(False),
    OnlySDs = cms.vstring('ZdcSensitiveDetector', 'TotemT2ScintSensitiveDetector', 'TotemSensitiveDetector', 'RomanPotSensitiveDetector', 'PLTSensitiveDetector', 'MuonSensitiveDetector', 'MtdSensitiveDetector', 'BCM1FSensitiveDetector', 'EcalSensitiveDetector', 'CTPPSSensitiveDetector', 'BSCSensitiveDetector', 'CTPPSDiamondSensitiveDetector', 'FP420SensitiveDetector', 'BHMSensitiveDetector', 'CastorSensitiveDetector', 'CaloTrkProcessing', 'HcalSensitiveDetector', 'TkAccumulatingSensitiveDetector'),
    G4CheckOverlap = cms.untracked.PSet(
        OutputBaseName = cms.string('2022'),
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
    G4CommandsEndRun = cms.vstring(),
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
                Stepper = cms.string('CMSTDormandPrince45'),
                Type = cms.string('CMSIMField'),
                StepperParam = cms.PSet(
                    VacRegions = cms.vstring(),
#                   VacRegions = cms.vstring('DefaultRegionForTheWorld','BeamPipeVacuum','BeamPipeOutside'),
                    EnergyThTracker = cms.double(0.2),     ## in GeV
                    RmaxTracker = cms.double(8000),        ## in mm
                    ZmaxTracker = cms.double(11000),       ## in mm
                    MaximumEpsilonStep = cms.untracked.double(0.01),
                    DeltaOneStep = cms.double(0.001),      ## in mm
                    DeltaOneStepTracker = cms.double(1e-4),## in mm
                    MaximumLoopCounts = cms.untracked.double(1000.0),
                    DeltaChord = cms.double(0.002),        ## in mm
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
                    DeltaIntersectionSimple = cms.double(0.01), ## in mm
                    MaxStepSimple = cms.double(50.),       ## in cm
                )
            )
        ),
        delta = cms.double(1.0) ## in mm
    ),
    Physics = cms.PSet(
        common_maximum_time,
        # NOTE : if you want EM Physics only,
        #        please select "SimG4Core/Physics/DummyPhysics" for type
        #        and turn ON DummyEMPhysics
        #
        type = cms.string('SimG4Core/Physics/FTFP_BERT_EMM'),
        DummyEMPhysics = cms.bool(False),
        # 1 will print cuts as they get set from DD
        # 2 will do as 1 + will dump Geant4 table of cuts
        Verbosity = cms.untracked.int32(0),
        # EM physics options
        CutsPerRegion = cms.bool(True),
        CutsOnProton  = cms.bool(True),
        DefaultCutValue = cms.double(1.0), ## cuts in cm
        G4BremsstrahlungThreshold = cms.double(0.5), ## cut in GeV
        G4MuonBremsstrahlungThreshold = cms.double(10000.), ## cut in GeV
        G4TrackingCut = cms.double(0.025), ## cut in MeV
        G4MscRangeFactor = cms.double(0.04),
        G4MscGeomFactor = cms.double(2.5), 
        G4MscSafetyFactor = cms.double(0.6), 
        G4MscLambdaLimit = cms.double(1.0), # in mm 
        G4MscStepLimit = cms.string("UseSafety"),
        G4GammaGeneralProcess = cms.bool(True),
        G4ElectronGeneralProcess = cms.bool(False),
        G4TransportWithMSC = cms.int32(0),  # 1 - fEnabled, 2 - fMultipleSteps
        PhotoeffectBelowKShell = cms.bool(True),
        G4HepEmActive = cms.bool(False),
        G4MuonPairProductionByMuon = cms.bool(False),
        ReadMuonData = cms.bool(False), 
        Region      = cms.string(''),
        TrackingCut = cms.bool(False),
        SRType      = cms.bool(True),
        FlagMuNucl  = cms.bool(False),
        FlagFluo    = cms.bool(False),
        EMPhysics   = cms.untracked.bool(True),
        # Hadronic physics options
        HadPhysics  = cms.untracked.bool(True),
        FlagBERT    = cms.untracked.bool(False),
        EminFTFP    = cms.double(3.), # in GeV
        EmaxBERT    = cms.double(6.), # in GeV
        EminQGSP    = cms.double(12.), # in GeV
        EmaxFTFP    = cms.double(25.), # in GeV
        EmaxBERTpi  = cms.double(12.), # in GeV
        G4NeutronGeneralProcess = cms.bool(False),
        G4BCHadronicProcess = cms.bool(False),
        G4LightHyperNucleiTracking = cms.bool(False),
        ThermalNeutrons = cms.untracked.bool(False),
        # Exotica
        MonopoleCharge       = cms.untracked.int32(1),
        MonopoleDeltaRay     = cms.untracked.bool(True),
        MonopoleMultiScatter = cms.untracked.bool(False),
        MonopoleTransport    = cms.untracked.bool(True),
        MonopoleMass         = cms.untracked.double(0),
        ExoticaTransport     = cms.untracked.bool(False),
        ExoticaPhysicsSS     = cms.untracked.bool(False),
        RhadronPhysics       = cms.bool(False),
        DarkMPFactor         = cms.double(1.0),
        # GFlash methods
        LowEnergyGflashEcal = cms.bool(False),
        LowEnergyGflashEcalEmax = cms.double(0.02), # in GeV
        GflashEcal    = cms.bool(False),
        GflashHcal    = cms.bool(False),
        GflashEcalHad = cms.bool(False),
        GflashHcalHad = cms.bool(False),
        bField        = cms.double(3.8),
        energyScaleEB = cms.double(1.032),
        energyScaleEE = cms.double(1.024),
        # Russian roulette
        RusRoElectronEnergyLimit  = cms.double(0.0),
        RusRoEcalElectron         = cms.double(1.0),
        RusRoHcalElectron         = cms.double(1.0),
        RusRoMuonIronElectron     = cms.double(1.0),
        RusRoPreShowerElectron    = cms.double(1.0),
        RusRoCastorElectron       = cms.double(1.0),
        RusRoWorldElectron        = cms.double(1.0),
        # Tracking and step limiters
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
        ThresholdWarningEnergy    = cms.untracked.double(100.0), #in MeV
        ThresholdImportantEnergy  = cms.untracked.double(250.0), #in MeV
        ThresholdTrials           = cms.untracked.int32(10)
    ),
    Generator = cms.PSet(
        common_maximum_time,
        HectorEtaCut,
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
            PDGfilterSel = cms.bool(False), ## filter out unwanted particles
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
        common_MCtruth,
        DetailedTiming = cms.untracked.bool(False),
        CheckTrack = cms.untracked.bool(False),
        EndPrintTrackID = cms.int32(0)
    ),
    SteppingAction = cms.PSet(
        common_maximum_time,
        MaxNumberOfSteps        = cms.int32(20000),
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
        AllMuonsPersistent = cms.bool(True),
        UseDemoHitRPC = cms.bool(True),
        UseDemoHitGEM = cms.bool(True),
        HaveDemoChambers = cms.bool(True)
    ),
    CaloSD = cms.PSet(
        common_heavy_suppression,
        common_MCtruth,
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
        ScaleRadLength  = cms.untracked.double(1.0),
        StoreLayerTimeSim = cms.untracked.bool(False),
        AgeingWithSlopeLY = cms.untracked.bool(False),
        Detectors         = cms.untracked.int32(3),
        DumpGeometry      = cms.untracked.int32(0)
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
        DumpGeometry              = cms.untracked.bool(False),
        HFDarkeningParameterBlock = HFDarkeningParameterBlock
    ),
    CaloTrkProcessing = cms.PSet(
        common_MCtruth,
        TestBeam   = cms.bool(False),
        EminTrack  = cms.double(0.01),
        PutHistory = cms.bool(False),
    ),
    HFShower = cms.PSet(
        common_UsePMT,
        common_UseHF,
        PEPerGeV          = cms.double(0.31),
        TrackEM           = cms.bool(False),
        UseShowerLibrary  = cms.bool(True),
        UseHFGflash       = cms.bool(False),
        EminLibrary       = cms.double(0.0),
        LambdaMean        = cms.double(350.0),
        ApplyFiducialCut  = cms.bool(True),
        RefIndex          = cms.double(1.459),
        Aperture          = cms.double(0.33),
        ApertureTrapped   = cms.double(0.22),
        CosApertureTrapped= cms.double(0.5),
        SinPsiMax         = cms.untracked.double(0.5),
        ParametrizeLast   = cms.untracked.bool(False),
        HFShowerBlock     = cms.PSet(refToPSet_ = cms.string("HFShowerBlock"))
    ),
    HFShowerLibrary = cms.PSet(
        HFLibraryFileBlock = cms.PSet(refToPSet_ = cms.string("HFLibraryFileBlock"))
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
        CheckID          = cms.untracked.bool(False),
        UseDetector      = cms.untracked.int32(3),
        Detectors        = cms.untracked.int32(2),
        MissingWaferFile = cms.untracked.string("")
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
        CheckID          = cms.untracked.bool(False),
        TileFileName     = cms.untracked.string("")
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

##
## Change HCAL numbering scheme in 2017
##
from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
run2_HCAL_2017.toModify( g4SimHits, HCalSD = dict( TestNumberingScheme = True ) )

##
## Disable Castor from Run 3, enable PPS (***temporarily disable PPS***)
##
from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify( g4SimHits, CastorSD = dict( useShowerLibrary = False ) )

## Disable PPS from Run 3 PbPb runs
##
from Configuration.Eras.Modifier_pp_on_PbPb_run3_cff import pp_on_PbPb_run3
pp_on_PbPb_run3.toModify( g4SimHits, LHCTransport = False )

##
## Change ECAL time slices
##
from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toModify( g4SimHits, ECalSD = dict(
                             StoreLayerTimeSim = True,
                             TimeSliceUnit = 0.001 )
)

##
## Change CALO Thresholds
##
from Configuration.Eras.Modifier_h2tb_cff import h2tb
h2tb.toModify(g4SimHits,
              OnlySDs = ['EcalSensitiveDetector', 'CaloTrkProcessing', 'HcalTB06BeamDetector', 'HcalSensitiveDetector'],
              CaloSD = dict(
                  EminHits  = [0.0, 0.0, 0.0, 0.0, 0.0],
                  TmaxHits  = [1000.0, 1000.0, 1000.0, 1000.0, 2000.0] ),
              CaloTrkProcessing = dict(
                  TestBeam = True ),
              HCalSD = dict(
                  ForTBHCAL = True )
)

##
## DD4hep migration
##
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
dd4hep.toModify( g4SimHits, g4GeometryDD4hepSource = True )

##
## Selection of SD's for Phase2, exclude PPS
##

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(g4SimHits,
                       OnlySDs = ['ZdcSensitiveDetector', 'TotemT2ScintSensitiveDetector', 'TotemSensitiveDetector', 'RomanPotSensitiveDetector', 'PLTSensitiveDetector', 'MuonSensitiveDetector', 'MtdSensitiveDetector', 'BCM1FSensitiveDetector', 'EcalSensitiveDetector', 'CTPPSSensitiveDetector', 'HGCalSensitiveDetector', 'BSCSensitiveDetector', 'CTPPSDiamondSensitiveDetector', 'FP420SensitiveDetector', 'BHMSensitiveDetector', 'HFNoseSensitiveDetector', 'HGCScintillatorSensitiveDetector', 'CastorSensitiveDetector', 'CaloTrkProcessing', 'HcalSensitiveDetector', 'TkAccumulatingSensitiveDetector'],
                       LHCTransport = False, 
                       MuonSD = dict( 
                       HaveDemoChambers = False ) 
)

from Configuration.Eras.Modifier_hgcaltb_cff import hgcaltb
hgcaltb.toModify(g4SimHits,
                 OnlySDs = ['AHcalSensitiveDetector', 'HGCSensitiveDetector', 'HGCalTB1601SensitiveDetector', 'HcalTB06BeamDetector']
)
