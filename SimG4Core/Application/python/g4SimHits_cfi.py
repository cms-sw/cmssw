import FWCore.ParameterSet.Config as cms

from SimG4Core.Application.hectorParameter_cfi import *
common_heavy_suppression = cms.PSet(
    NeutronThreshold = cms.double(30.0),
    ProtonThreshold = cms.double(30.0),
    IonThreshold = cms.double(30.0)
)
g4SimHits = cms.EDProducer("OscarProducer",
    SteppingAction = cms.PSet(
        CriticalDensity = cms.double(1e-15),
        Verbosity = cms.untracked.int32(0),
        CriticalEnergyForVacuum = cms.double(2.0),
        KillBeamPipe = cms.bool(True)
    ),
    G4StackManagerVerbosity = cms.untracked.int32(0),
    OverrideUserStackingAction = cms.bool(True),
    G4TrackingManagerVerbosity = cms.untracked.int32(0),
    EventAction = cms.PSet(
        debug = cms.untracked.bool(False),
        StopFile = cms.string('StopRun'),
        CollapsePrimaryVertices = cms.bool(False)
    ),
    G4EventManagerVerbosity = cms.untracked.int32(0),
    StoreRndmSeeds = cms.bool(False),
    StorePhysicsTables = cms.bool(False),
    CheckOverlap = cms.untracked.bool(False),
    RestorePhysicsTables = cms.bool(False),
    Generator = cms.PSet(
        HectorEtaCut,
        # string HepMCProductLabel = "VtxSmeared"
        ApplyPtCuts = cms.bool(True),
        MaxPhiCut = cms.double(3.14159265359), ## according to CMS conventions

        ApplyEtaCuts = cms.bool(True),
        MaxPtCut = cms.double(99999.0), ## the ptmax=99.TeV in this case

        MinPtCut = cms.double(0.04), ## the pt-cut is in GeV (CMS conventions)

        ApplyPhiCuts = cms.bool(False),
        Verbosity = cms.untracked.int32(0),
        MinPhiCut = cms.double(-3.14159265359), ## in radians

        MaxEtaCut = cms.double(5.5),
        HepMCProductLabel = cms.string('source'),
        MinEtaCut = cms.double(-5.5),
        #Temporary fix we are investigating more in detail the inpact of this selection 
        DecLenCut = cms.double(2.9)
    ),
    HcalTB02SD = cms.PSet(
        UseBirkLaw = cms.untracked.bool(False),
        BirkC1 = cms.untracked.double(0.013),
        BirkC2 = cms.untracked.double(0.0568),
        BirkC3 = cms.untracked.double(1.75)
    ),
    MagneticField = cms.PSet(
        UseLocalMagFieldManager = cms.bool(False),
        Verbosity = cms.untracked.bool(False),
        ConfGlobalMFM = cms.PSet(
            Volume = cms.string('OCMS'),
            OCMS = cms.PSet(
                Stepper = cms.string('G4ClassicalRK4'),
                Type = cms.string('CMSIMField'),
                G4ClassicalRK4 = cms.PSet(
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
    ECalSD = cms.PSet(
        BirkL3Parametrization = cms.bool(True),
        BirkCut = cms.double(0.1),
        BirkC1 = cms.double(0.03333),
        BirkC2 = cms.double(0.0),
        BirkC3 = cms.double(1.0),
        TestBeam = cms.untracked.bool(False),
        SlopeLightYield = cms.double(0.02),
        UseBirkLaw = cms.bool(True),
        BirkSlope = cms.double(0.253694)
    ),
    UseMagneticField = cms.bool(True),
    NonBeamEvent = cms.bool(False),
    BscSD = cms.PSet(
        Verbosity = cms.untracked.int32(0)
    ),
    TotemSD = cms.PSet(
        Verbosity = cms.untracked.int32(0)
    ),
    HCalSD = cms.PSet(
        BetaThreshold = cms.double(0.9),
        TestNumberingScheme = cms.bool(False),
        UsePMTHits = cms.bool(False),
        UseParametrize = cms.bool(False),
        ForTBH2 = cms.untracked.bool(False),
        WtFile = cms.untracked.string('None'),
        UseHF = cms.untracked.bool(True),
        UseLayerWt = cms.untracked.bool(False),
        UseShowerLibrary = cms.bool(True),
        BirkC1 = cms.double(0.013),
        BirkC2 = cms.double(0.0568),
        BirkC3 = cms.double(1.75),
        UseBirkLaw = cms.bool(True)
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
        common_heavy_suppression,
        SuppressHeavy = cms.bool(False),
        DetailedTiming = cms.untracked.bool(False),
        Verbosity = cms.untracked.int32(0),
        CheckHits = cms.untracked.int32(25),
        BeamPosition = cms.untracked.double(0.0),
        CorrectTOFBeam = cms.untracked.bool(False),
        UseMap = cms.untracked.bool(True),
        EminTrack = cms.double(1.0)
    ),
    HFCherenkov = cms.PSet(
        RefIndex = cms.double(1.459),
        Gain = cms.double(0.33),
        Aperture = cms.double(0.33),
        CheckSurvive = cms.bool(False),
        Lambda1 = cms.double(280.0),
        Lambda2 = cms.double(700.0),
        ApertureTrapped = cms.double(0.22)
    ),
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
        BirkC2 = cms.double(0.0568),
        BirkC3 = cms.double(1.75)
    ),
    Watchers = cms.VPSet(),
    EcalTBH4BeamSD = cms.PSet(
        UseBirkLaw = cms.bool(False),
        BirkC1 = cms.double(0.013),
        BirkC2 = cms.double(0.0568),
        BirkC3 = cms.double(1.75)
    ),
    RunAction = cms.PSet(
        StopFile = cms.string('StopRun')
    ),
    ZdcSD = cms.PSet(
        Verbosity = cms.int32(0),
        FiberDirection = cms.double(0.0)
    ),
    RestoreRndmSeeds = cms.bool(False),
    FP420SD = cms.PSet(
        Verbosity = cms.untracked.int32(2)
    ),
    Physics = cms.PSet(
        type = cms.string('SimG4Core/Physics/QGSP_EMV'),
        G4BremsstrahlungThreshold = cms.double(0.5), ## cut in GeV
        DefaultCutValue = cms.double(1.0), ## cuts in cm
        Verbosity = cms.untracked.int32(0),
        # 2 will do as 1 + will dump Geant4 table of cuts
        CutsPerRegion = cms.bool(True),

        FlagBERT = cms.untracked.bool(False),
        FlagCHIPS   = cms.untracked.bool(False),
        FlagFTF     = cms.untracked.bool(False),
        FlagGlauber = cms.untracked.bool(False),
        FlagHP      = cms.untracked.bool(False),
        EMPhysics = cms.untracked.bool(True),
        HadPhysics = cms.untracked.bool(True),
        DummyEMPhysics = cms.bool(False)
    ),
    GFlash = cms.PSet(
        GflashEMShowerModel = cms.bool(False),
        GflashHadronShowerModel = cms.bool(False),
        GflashHistogram = cms.bool(False),
        GflashHadronPhysics = cms.string('QGSP')
    ),
    G4Commands = cms.vstring(),
    StackingAction = cms.PSet(
        common_heavy_suppression,
        TrackNeutrino = cms.bool(False),
        KillHeavy = cms.bool(False),
        SavePrimaryDecayProductsAndConversions = cms.untracked.bool(True)
    ),
    CastorSD = cms.PSet(
        Verbosity = cms.untracked.int32(0)
    ),
    HFShower = cms.PSet(
        TrackEM = cms.bool(False),
        CFibre = cms.double(0.5),
        PEPerGeV = cms.double(0.25),
        ProbMax = cms.double(0.7268),
        PEPerGeVPMT = cms.double(1.0)
    ),
    MuonSD = cms.PSet(
        EnergyThresholdForPersistency = cms.double(1.0),
        PrintHits = cms.bool(False),
        AllMuonsPersistent = cms.bool(False)
    ),
    CaloTrkProcessing = cms.PSet(
        TestBeam = cms.bool(False),
        EminTrack = cms.double(0.01)
    ),
    PhysicsTablesDirectory = cms.string('PhysicsTables'),
    TrackingAction = cms.PSet(
        DetailedTiming = cms.untracked.bool(False)
    )
)


