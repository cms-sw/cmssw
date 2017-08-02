import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
process = cms.Process('test', eras.ctpps_2016)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("Configuration.StandardSequences.Services_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50000),
)
#process.MessageLogger.cerr.threshold = 'INFO'
#process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32( 1000 )

process.source = cms.Source('EmptySource')

# load the geometry
process.load('SimCTPPS.OpticsParameterisation.simGeometryRP_cfi')

process.generator = cms.EDFilter('Pythia8GeneratorFilter',
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(13000.),
    PythiaParameters = cms.PSet(
        pythia8_proc = cms.vstring(
            'SoftQCD:nonDiffractive = off',
            'SoftQCD:singleDiffractive = on',
            'SoftQCD:doubleDiffractive = off',
            'Tune:pp 2',
        ),
        parameterSets = cms.vstring('pythia8_proc')
    )
)

process.load('SimCTPPS.OpticsParameterisation.ctppsFastProtonSimulation_cfi')

# load the reconstruction part
process.load('RecoCTPPS.TotemRPLocal.totemRPUVPatternFinder_cfi')
process.load('RecoCTPPS.TotemRPLocal.totemRPLocalTrackFitter_cfi')
process.load('RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cfi')

process.load('RecoCTPPS.ProtonReconstruction.ctppsProtonReconstruction_cfi')
#process.ctppsFastProtonSimulation.beamConditions.yOffsetSector45 = cms.double(0.0)
#process.ctppsFastProtonSimulation.beamConditions.yOffsetSector56 = cms.double(0.0)
#process.ctppsFastProtonSimulation.beamConditions.halfCrossingAngleSector45 = cms.double(0.0)
#process.ctppsFastProtonSimulation.beamConditions.halfCrossingAngleSector56 = cms.double(0.0)
process.totemRPUVPatternFinder.tagRecHit = cms.InputTag("ctppsFastProtonSimulation")

process.out = cms.OutputModule('PoolOutputModule',
    fileName = cms.untracked.string('ctppsSim.root'),
    outputCommands = cms.untracked.vstring('keep *')
)

# for detectors resolution smearing
process.RandomNumberGeneratorService.ctppsFastProtonSimulation = cms.PSet( initialSeed = cms.untracked.uint32(1), )

process.load('Validation.CTPPS.ctppsParameterisationValidation_cfi')

# prepare the output file
process.TFileService = cms.Service('TFileService',
    fileName = cms.string('output.root'),
    closeFileFast = cms.untracked.bool(True),
)

process.p = cms.Path(
    process.generator
    * process.ctppsFastProtonSimulation
    * process.totemRPUVPatternFinder
    * process.totemRPLocalTrackFitter
    * process.ctppsLocalTrackLiteProducer
    * process.ctppsProtonReconstruction
)

process.e = cms.EndPath(
    process.out
    * process.paramValidation
)
