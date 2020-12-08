import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load("Configuration.Geometry.GeometryExtended2017_cff")
process.load("Configuration.Geometry.GeometryExtended2017Reco_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load("SimG4CMS.Calo.EcalSimHitStudy_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['run2_mc']

#
#    


process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

process.source = cms.Source("EmptySource",
    firstRun        = cms.untracked.uint32(1),
    firstEvent      = cms.untracked.uint32(1)
)

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(11),
        MinEta = cms.double(-2.9),
        MaxEta = cms.double(2.9),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinE   = cms.double(10.0),
        MaxE   = cms.double(10.0)
    ),
    Verbosity       = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(False)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('electron10.root')
)

process.g4SimHits.ECalSD.StoreRadLength  = True

process.EcalSimHitStudy.MaxEnergy = 15.0

process.EcalSimHitStudyNormal = process.EcalSimHitStudy.clone(
	MaxEnergy  = 15.0,
	SelectX    = 0)

process.EcalSimHitStudyReflect = process.EcalSimHitStudy.clone(
	MaxEnergy  = 15.0,
	SelectX    = 1)


process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.analysis_step   = cms.Path(process.EcalSimHitStudy*
				   process.EcalSimHitStudyNormal*
				   process.EcalSimHitStudyReflect)


# Schedule definition
process.schedule = cms.Schedule(process.generation_step,
                                process.simulation_step,
                                process.analysis_step,
                                )

# filter all path with the production filter sequence
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq
