import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_h2tb_cff import h2tb

process = cms.Process("PROD", h2tb)

process = cms.Process("PROD")

process.load('SimG4CMS.HcalTestBeam.TB2006Geometry77XML_cfi')
process.load('SimGeneral.HepPDTESSource.pdt_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("Geometry.EcalCommonData.ecalSimulationParameters_cff")
process.load('Geometry.HcalTestBeamData.hcalDDDSimConstants_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedFlat_cfi')
process.load('GeneratorInterface.Core.generatorSmeared_cfi')
process.load('SimG4Core.Application.g4SimHits_cfi')
process.load('IOMC.RandomEngine.IOMC_cff')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HCalGeom=dict()
    process.MessageLogger.HcalSim=dict()

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('hcaltb06_77.root')
                                   )

process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

beamPosition = -800.0

process.common_beam_direction_parameters = cms.PSet(
    MinE   = cms.double(50.0),
    MaxE   = cms.double(50.0),
    PartID = cms.vint32(-211),
    MinEta       = cms.double(0.5655),
    MaxEta       = cms.double(0.5655),
    MinPhi       = cms.double(0.15708),
    MaxPhi       = cms.double(0.15708),
    BeamPosition = cms.double(beamPosition)
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
                              fileName = cms.untracked.string('sim2006_77.root')
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

process.g4SimHits.NonBeamEvent = True
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/FTFP_BERT_EMM'
process.g4SimHits.Physics.Region = 'HcalRegion'
process.g4SimHits.Physics.DefaultCutValue = 1.

process.g4SimHits.StackingAction.KillGamma = False
process.g4SimHits.CaloTrkProcessing.TestBeam = True
process.g4SimHits.CaloSD.BeamPosition = beamPosition
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
process.g4SimHits.OnlySDs = ['CaloTrkProcessing',
                             'EcalTBH4BeamDetector',
                             'HcalTB02SensitiveDetector',
                             'HcalTB06BeamDetector',
                             'EcalSensitiveDetector',
                             'HcalSensitiveDetector']
