import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *

def testbeam2006(process):

    process.load('FWCore.MessageService.MessageLogger_cfi')
    process.load('Configuration.StandardSequences.Services_cff')
    process.load('SimGeneral.HepPDTESSource.pdt_cfi')
    process.load('Configuration.EventContent.EventContent_cff')
    process.load("Geometry.EcalCommonData.ecalSimulationParameters_cff")
    process.load('Geometry.HcalTestBeamData.hcalDDDSimConstants_cff')
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

    process.load('SimG4CMS.HcalTestBeam.TBDirectionParameters_cfi')
    process.load('SimG4CMS.HcalTestBeam.TBVtxSmeared_cfi')
    process.load('SimG4CMS.HcalTestBeam.TB06Analysis_cfi')

    process.generator = cms.EDProducer("FlatRandomEGunProducer",
        PGunParameters = cms.PSet(
            process.common_beam_direction_parameters
        ),
        Verbosity       = cms.untracked.int32(0),
        AddAntiParticle = cms.bool(False)
    )

    process.p1 = cms.Path(process.generator*process.VtxSmeared*process.generatorSmeared*process.g4SimHits*process.testbeam)

    process.common_maximum_time.MaxTrackTime = cms.double(1000.0)
#    process.common_maximum_time.MaxTimeNames = cms.vstring()
#    process.common_maximum_time.MaxTrackTimes = cms.vdouble()
#    process.common_maximum_time.DeadRegions = cms.vstring()

    process.g4SimHits.NonBeamEvent = True
    process.g4SimHits.UseMagneticField = False

    process.g4SimHits.Physics.type = 'SimG4Core/Physics/FTFP_BERT_EMM'
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
    process.g4SimHits.CaloTrkProcessing.TestBeam = True

    process.g4SimHits.HCalSD.UseShowerLibrary    = False
    process.g4SimHits.HCalSD.UseHF   = False
    process.g4SimHits.HCalSD.ForTBHCAL = True
    process.g4SimHits.HCalSD.ForTBH2 = False

    return(process)
