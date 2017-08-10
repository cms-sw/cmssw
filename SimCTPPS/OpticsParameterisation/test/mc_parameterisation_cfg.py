import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
process = cms.Process('HLT', eras.ctpps_2016)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealisticCrossingAngleCollision2016_cfi')
process.load('Configuration.StandardSequences.Generator_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50000),
)

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32( 1000 )

process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring(
#'/store/group/phys_pps/diphoton/GammaGammaToGammaGamma_13TeV_fpmc/RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1_MINIAODSIM/170324_062301/0000/GammaGammaGammaToGammaGamma-RunIISummer16MiniAODv2_1.root', # miniAOD-level
'/store/group/phys_pps/diphoton/GammaGammaToGammaGamma_13TeV_fpmc/GammaGammaToGammaGamma_13TeV_fpmc_GEN-SIM/170319_191338/0000/GammaGammaGammaGamma_Tune4C_13TeV_pythia8_cff_py_GEN_1.root', # GEN-level
    )
)

# load the geometry
process.load('SimCTPPS.OpticsParameterisation.simGeometryRP_cfi')

process.VtxSmeared.src = cms.InputTag('source')

process.load('SimCTPPS.OpticsParameterisation.ctppsFastProtonSimulation_cfi')
process.ctppsFastProtonSimulation.beamParticlesTag = cms.InputTag('source')
#process.ctppsFastProtonSimulation.beamParticlesTag = cms.InputTag('generatorSmeared')
#process.ctppsFastProtonSimulation.beamParticlesTag = cms.InputTag('prunedGenParticles') # miniAOD
#process.ctppsFastProtonSimulation.yOffsetSector45 = cms.double(0.0)
#process.ctppsFastProtonSimulation.yOffsetSector56 = cms.double(0.0)
#process.ctppsFastProtonSimulation.beamConditions.halfCrossingAngleSector45 = cms.double(0.0)
#process.ctppsFastProtonSimulation.beamConditions.halfCrossingAngleSector56 = cms.double(0.0)

# load the reconstruction part
process.load('RecoCTPPS.TotemRPLocal.totemRPUVPatternFinder_cfi')
process.load('RecoCTPPS.TotemRPLocal.totemRPLocalTrackFitter_cfi')
process.load('RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cfi')

process.totemRPUVPatternFinder.tagRecHit = cms.InputTag('ctppsFastProtonSimulation')

process.out = cms.OutputModule('PoolOutputModule',
    fileName = cms.untracked.string('ctppsSim.root')
)

# for detectors resolution smearing
process.RandomNumberGeneratorService.ctppsFastProtonSimulation = cms.PSet( initialSeed = cms.untracked.uint32(1), )

process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(
    process.ctppsFastProtonSimulation
    * process.totemRPUVPatternFinder
    * process.totemRPLocalTrackFitter
    * process.ctppsLocalTrackLiteProducer
)
process.outpath = cms.EndPath(process.out)

process.schedule = cms.Schedule(
    process.generation_step,
    process.simulation_step,
    process.outpath
)

