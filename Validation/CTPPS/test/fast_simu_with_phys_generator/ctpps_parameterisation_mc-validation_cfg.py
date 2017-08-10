import FWCore.ParameterSet.Config as cms

process = cms.Process('test')

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("Configuration.StandardSequences.Services_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50000),
)
#process.MessageLogger.cerr.threshold = 'INFO'
#process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32( 1000 )

process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring(
#'/store/group/phys_pps/diphoton/GammaGammaToGammaGamma_13TeV_fpmc/RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1_MINIAODSIM/170324_062301/0000/GammaGammaGammaToGammaGamma-RunIISummer16MiniAODv2_1.root', # miniAOD-level
'/store/group/phys_pps/diphoton/GammaGammaToGammaGamma_13TeV_fpmc/GammaGammaToGammaGamma_13TeV_fpmc_GEN-SIM/170319_191338/0000/GammaGammaGammaGamma_Tune4C_13TeV_pythia8_cff_py_GEN_1.root', # GEN-level
    )
)
process.load('SimCTPPS.OpticsParameterisation.ctppsFastProtonSimulation_cfi')

# load the geometry
process.load('SimCTPPS.OpticsParameterisation.simGeometryRP_cfi')

# load the reconstruction
process.load('RecoCTPPS.TotemRPLocal.totemRPUVPatternFinder_cfi')
process.load('RecoCTPPS.TotemRPLocal.totemRPLocalTrackFitter_cfi')
process.load('RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cfi')

process.load('RecoCTPPS.ProtonReconstruction.ctppsOpticsReconstruction_cfi')
process.ctppsFastProtonSimulation.beamParticlesTag = cms.InputTag('source')
#process.ctppsFastProtonSimulation.beamParticlesTag = cms.InputTag('generatorSmeared')
#process.ctppsFastProtonSimulation.beamParticlesTag = cms.InputTag('prunedGenParticles') # miniAOD
#process.ctppsFastProtonSimulation.beamConditions.yOffsetSector45 = cms.double(0.0)
#process.ctppsFastProtonSimulation.beamConditions.yOffsetSector56 = cms.double(0.0)
#process.ctppsFastProtonSimulation.beamConditions.halfCrossingAngleSector45 = cms.double(0.0)
#process.ctppsFastProtonSimulation.beamConditions.halfCrossingAngleSector56 = cms.double(0.0)
process.totemRPUVPatternFinder.tagRecHit = cms.InputTag("ctppsFastProtonSimulation")
process.ctppsLocalTrackLiteProducer.doNothing = cms.bool(False)

process.out = cms.OutputModule('PoolOutputModule',
    fileName = cms.untracked.string('ctppsSim.root'),
    outputCommands = cms.untracked.vstring('keep *')
)

# for detectors resolution smearing
process.RandomNumberGeneratorService.ctppsFastProtonSimulation = cms.PSet( initialSeed = cms.untracked.uint32(1), )

process.load('Validation.CTPPS.ctppsParameterisationValidation_cfi')
process.paramValidation.genProtonsTag = cms.InputTag('source')

# prepare the output file
process.TFileService = cms.Service('TFileService',
    fileName = cms.string('output.root'),
    closeFileFast = cms.untracked.bool(True),
)

process.p = cms.Path(
    process.ctppsFastProtonSimulation
    * process.ctppsFastProtonSimulation
    * process.totemRPUVPatternFinder
    * process.totemRPLocalTrackFitter
    * process.ctppsLocalTrackLiteProducer
    * process.ctppsOpticsReconstruction
)

process.e = cms.EndPath(
    process.out
    * process.paramValidation
)
