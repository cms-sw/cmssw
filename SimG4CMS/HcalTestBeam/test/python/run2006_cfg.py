import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load('SimG4CMS.HcalTestBeam.TB2006GeometryXML_cfi')
process.load('SimGeneral.HepPDTESSource.pdt_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Geometry.HcalTestBeamData.hcalDDDSimConstants_cff')
process.load("Geometry.EcalCommonData.ecalSimulationParameters_cff")
process.load('Configuration.EventContent.EventContent_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedFlat_cfi')
process.load('GeneratorInterface.Core.generatorSmeared_cfi')
process.load('SimG4Core.Application.g4SimHits_cfi')
process.load('IOMC.RandomEngine.IOMC_cff')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.categories.append('HCalGeom')
    process.MessageLogger.categories.append('HcalSim')

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('hcaltb06.root')
                                   )

process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.common_beam_direction_parameters = cms.PSet(
    MinE   = cms.double(50.0),
    MaxE   = cms.double(50.0),
    PartID = cms.vint32(-211),
    MinEta       = cms.double(0.5655),
    MaxEta       = cms.double(0.5655),
    MinPhi       = cms.double(-0.1309),
    MaxPhi       = cms.double(-0.1309),
    BeamPosition = cms.double(-800.0)
    )

process.source = cms.Source("EmptySource",
                            firstRun        = cms.untracked.uint32(1),
                            firstEvent      = cms.untracked.uint32(1)
                            )

process.generator = cms.EDProducer("FlatRandomEGunProducer",
                                   PGunParameters = cms.PSet(
        process.common_beam_direction_parameters,
        ),
                                   Verbosity       = cms.untracked.int32(0),
                                   AddAntiParticle = cms.bool(False)
                                   )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
    )

process.o1 = cms.OutputModule("PoolOutputModule",
                              process.FEVTSIMEventContent,
                              fileName = cms.untracked.string('sim2006.root')
                              )

process.common_heavy_suppression1 = cms.PSet(
    NeutronThreshold = cms.double(30.0),
    ProtonThreshold = cms.double(30.0),
    IonThreshold = cms.double(30.0)
)

process.Timing = cms.Service("Timing")

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
process.VtxSmeared = cms.EDProducer("BeamProfileVtxGenerator",
    process.common_beam_direction_parameters,
                                    VtxSmearedCommon,
                                    BeamMeanX       = cms.double(0.0),
                                    BeamMeanY       = cms.double(0.0),
                                    BeamSigmaX      = cms.double(0.0001),
                                    BeamSigmaY      = cms.double(0.0001),
                                    Psi             = cms.double(999.9),
                                    GaussianProfile = cms.bool(False),
                                    BinX            = cms.int32(50),
                                    BinY            = cms.int32(50),
                                    File            = cms.string('beam.profile'),
                                    UseFile         = cms.bool(False),
                                    TimeOffset      = cms.double(0.)
                                    )

process.testbeam = cms.EDAnalyzer("HcalTB06Analysis",
                                  process.common_beam_direction_parameters,
                                  ECAL = cms.bool(True),
                                  TestBeamAnalysis = cms.PSet(
        EHCalMax   = cms.untracked.double(400.0),
        ETtotMax   = cms.untracked.double(400.0),
        beamEnergy = cms.untracked.double(50.),
        TimeLimit  = cms.double(180.0),
        EcalWidth  = cms.double(0.362),
        HcalWidth  = cms.double(0.640),
        EcalFactor = cms.double(1.0),
        HcalFactor = cms.double(100.0),
        MIP        = cms.double(0.8),
        Verbose    = cms.untracked.bool(True),
        MakeTree   = cms.untracked.bool(True)
        )
                                  )

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.generatorSmeared*process.g4SimHits*process.testbeam)
process.outpath = cms.EndPath(process.o1)

process.common_maximum_timex = cms.PSet(
    MaxTrackTime  = cms.double(1000.0),
    MaxTimeNames  = cms.vstring(),
    MaxTrackTimes = cms.vdouble(),
    DeadRegions   = cms.vstring(),
    CriticalEnergyForVacuum = cms.double(2.0),
    CriticalDensity         = cms.double(1e-15)
    )
process.g4SimHits.NonBeamEvent = True
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_FTFP_BERT_EML'
process.g4SimHits.Physics.Region = 'HcalRegion'
process.g4SimHits.Physics.DefaultCutValue = 1.

process.g4SimHits.ECalSD.UseBirkLaw = True
process.g4SimHits.ECalSD.BirkL3Parametrization = True
process.g4SimHits.ECalSD.BirkC1 = 0.033
process.g4SimHits.ECalSD.BirkC2 = 0.0
process.g4SimHits.ECalSD.SlopeLightYield = 0.02
process.g4SimHits.HCalSD.UseBirkLaw = True
process.g4SimHits.HCalSD.BirkC1 = 0.0052
process.g4SimHits.HCalSD.BirkC2 = 0.142
process.g4SimHits.HCalSD.BirkC3 = 1.75
process.g4SimHits.HCalSD.UseLayerWt = False
process.g4SimHits.HCalSD.WtFile     = ' '
process.g4SimHits.HCalSD.UseShowerLibrary    = False
process.g4SimHits.HCalSD.TestNumberingScheme = False
process.g4SimHits.HCalSD.UseHF   = False
process.g4SimHits.HCalSD.ForTBHCAL = True
process.g4SimHits.HCalSD.ForTBH2 = True
process.g4SimHits.StackingAction = cms.PSet(
    process.common_heavy_suppression1,
    process.common_maximum_timex,
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
    )

process.g4SimHits.SteppingAction = cms.PSet(
    process.common_maximum_timex,
    EkinNames               = cms.vstring(),
    EkinThresholds          = cms.vdouble(),
    EkinParticles           = cms.vstring()
    )

process.g4SimHits.CaloSD = cms.PSet(
    process.common_beam_direction_parameters,
    process.common_heavy_suppression1,
    EminTrack      = cms.double(1.0),
    TmaxHit        = cms.double(1000.0),
    EminHits       = cms.vdouble(0.0,0.0,0.0,0.0),
    EminHitsDepth  = cms.vdouble(0.0,0.0,0.0,0.0),
    TmaxHits       = cms.vdouble(1000.0,1000.0,1000.0,1000.0),
    HCNames        = cms.vstring('EcalHitsEB','EcalHitsEE','EcalHitsES','HcalHits'),
    UseResponseTables = cms.vint32(0,0,0,0),
    SuppressHeavy  = cms.bool(False),
    CheckHits      = cms.untracked.int32(25),
    UseMap         = cms.untracked.bool(True),
    Verbosity      = cms.untracked.int32(0),
    DetailedTiming = cms.untracked.bool(False),
    CorrectTOFBeam = cms.bool(False)
    )

process.g4SimHits.CaloTrkProcessing.TestBeam = True
