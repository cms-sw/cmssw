import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *

def testbeam2006(process):

    process.load('FWCore.MessageService.MessageLogger_cfi')
    process.load('Configuration.StandardSequences.Services_cff')
    process.load('SimGeneral.HepPDTESSource.pdt_cfi')
    process.load('Configuration.EventContent.EventContent_cff')
    process.load('Geometry.HcalCommonData.hcalParameters_cfi')
    process.load('Geometry.HcalCommonData.hcalDDDSimConstants_cfi')
    process.load('IOMC.EventVertexGenerators.VtxSmearedFlat_cfi')
    process.load('GeneratorInterface.Core.generatorSmeared_cfi')
    process.load('SimG4Core.Application.g4SimHits_cfi')
    process.load('IOMC.RandomEngine.IOMC_cff')

    process.RandomNumberGeneratorService.generator.initialSeed = 456789
    process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
    process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

    process.source = cms.Source("EmptySource",
        firstRun        = cms.untracked.uint32(1),
        firstEvent      = cms.untracked.uint32(1)
    )

    process.common_beam_parameters = cms.PSet(
        MinE   = cms.double(50.0),
        MaxE   = cms.double(50.0),
        PartID = cms.vint32(-211),
        MinEta = cms.double(0.2175),
        MaxEta = cms.double(0.2175),
        MinPhi = cms.double(-0.1309),
        MaxPhi = cms.double(-0.1309),
        BeamPosition = cms.double(-800.0)
    )

    process.generator = cms.EDProducer("FlatRandomEGunProducer",
        PGunParameters = cms.PSet(
            process.common_beam_parameters
        ),
        Verbosity       = cms.untracked.int32(0),
        AddAntiParticle = cms.bool(False)
    )

    process.VtxSmeared = cms.EDProducer("BeamProfileVtxGenerator",
        process.common_beam_parameters,
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
        process.common_beam_parameters,
        ECAL = cms.bool(True),
        TestBeamAnalysis = cms.PSet(
            Verbose = cms.untracked.bool(False),
            ETtotMax = cms.untracked.double(400.),
            EHCalMax = cms.untracked.double(400.),
            beamEnergy = cms.untracked.double(50.),
            TimeLimit  = cms.double(180.),
            EcalWidth  = cms.double(0.362),
            HcalWidth  = cms.double(0.640),
            EcalFactor = cms.double(1.),
            HcalFactor = cms.double(100.)
        )
    )

    process.p1 = cms.Path(process.generator*process.VtxSmeared*process.generatorSmeared*process.g4SimHits*process.testbeam)

    process.common_maximum_time.MaxTrackTime = cms.double(1000.0)
    process.common_maximum_time.MaxTimeNames = cms.vstring()
    process.common_maximum_time.MaxTrackTimes = cms.vstring()
    process.common_maximum_time.DeadRegions = cms.vstring()

    process.g4SimHits.NonBeamEvent = True
    process.g4SimHits.UseMagneticField = False

    process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_FTFP_BERT_EML'
    process.g4SimHits.Physics.Region = 'HcalRegion'
    process.g4SimHits.Physics.MaxTrackTime = cms.double(1000.0)

    process.g4SimHits.Generator.ApplyEtaCuts = cms.bool(False)

    process.g4SimHits.StackingAction.MaxTrackTime = cms.double(1000.0)
    process.g4SimHits.StackingAction.MaxTimeNames = cms.vstring()
    process.g4SimHits.StackingAction.MaxTrackTimes = cms.vdouble()
    process.g4SimHits.StackingAction.DeadRegions = cms.vstring()

    process.g4SimHits.SteppingAction.MaxTrackTime = cms.double(1000.0)
    process.g4SimHits.SteppingAction.MaxTimeNames = cms.vstring()
    process.g4SimHits.SteppingAction.MaxTrackTimes = cms.vdouble()
    process.g4SimHits.SteppingAction.DeadRegions = cms.vstring()

    process.g4SimHits.CaloSD.EminHits = cms.vdouble(0.0,0.0,0.0,0.0)
    process.g4SimHits.CaloSD.TmaxHits = cms.vdouble(1000.0,1000.0,1000.0,1000.0)

    process.g4SimHits.HCalSD.UseShowerLibrary    = False
    process.g4SimHits.HCalSD.UseHF   = False
    process.g4SimHits.HCalSD.ForTBH2 = False

    return(process)
