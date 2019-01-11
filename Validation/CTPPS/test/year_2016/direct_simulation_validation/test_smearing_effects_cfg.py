import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
process = cms.Process('CTPPSTestAcceptance', eras.ctpps_2016)

# minimal logger settings
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cerr'),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
)

# number of events
process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(280000)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

# particle-data table
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# supply LHC info
process.load("Validation.CTPPS.year_2016.ctppsLHCInfoESSource_cfi")

# supply optics
process.load("Validation.CTPPS.year_2016.ctppsOpticalFunctionsESSource_cfi")

# supply beam parameters
process.load("Validation.CTPPS.year_2016.ctppsBeamParametersESSource_cfi")

# particle generator
process.load("Validation.CTPPS.year_2016.randomXiThetaGunProducer_cfi")

# random seeds
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    sourceSeed = cms.PSet(initialSeed =cms.untracked.uint32(98765)),
    generator = cms.PSet(initialSeed = cms.untracked.uint32(98766)),
    beamDivergenceVtxGenerator = cms.PSet(initialSeed =cms.untracked.uint32(3849))
)

# geometry
process.load("Geometry.VeryForwardGeometry.geometryRPFromDD_2017_cfi")
del(process.XMLIdealGeometryESSource_CTPPS.geomXMLFiles[-1])
process.XMLIdealGeometryESSource_CTPPS.geomXMLFiles.append("Validation/CTPPS/test/year_2016/RP_Dist_Beam_Cent.xml")

# beam-smearing
process.load("IOMC.EventVertexGenerators.beamDivergenceVtxGenerator_cfi")

# fast simulation
process.load('Validation.CTPPS.ctppsDirectProtonSimulation_cfi')

process.ctppsDirectProtonSimulationSm = process.ctppsDirectProtonSimulation.clone(
  verbosity = 0,
  hepMCTag = cms.InputTag('beamDivergenceVtxGenerator'),
  useEmpiricalApertures = False,
  roundToPitch = True,
  pitchStrips = 66E-3 * 12 / 19, # effective value to reproduce real RP resolution
  produceHitsRelativeToBeam = True,
  produceScoringPlaneHits = True,
  produceRecHits = True,
)

process.ctppsDirectProtonSimulationNoSm = process.ctppsDirectProtonSimulationSm.clone(
  hepMCTag = cms.InputTag("generator", "unsmeared"),
  produceScoringPlaneHits = True,
  produceRecHits = False,
)

# strips reco
process.load('RecoCTPPS.TotemRPLocal.totemRPUVPatternFinder_cfi')
process.totemRPUVPatternFinder.tagRecHit = cms.InputTag('ctppsDirectProtonSimulationSm')

process.load('RecoCTPPS.TotemRPLocal.totemRPLocalTrackFitter_cfi')

# common reco: lite track production
process.load('RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cff')
process.ctppsLocalTrackLiteProducer.includeDiamonds = False
process.ctppsLocalTrackLiteProducer.includePixels = False

# plotters
process.ctppsDirectProtonSimulationValidatorBeamSm = cms.EDAnalyzer("CTPPSDirectProtonSimulationValidator",
  simuTracksTag = cms.InputTag("ctppsDirectProtonSimulationNoSm"),
  recoTracksTag = cms.InputTag("ctppsDirectProtonSimulationSm"),
  outputFile = cms.string("test_smearing_effects_beam.root")
)

process.ctppsDirectProtonSimulationValidatorSensorSm = cms.EDAnalyzer("CTPPSDirectProtonSimulationValidator",
  simuTracksTag = cms.InputTag("ctppsDirectProtonSimulationSm"),
  recoTracksTag = cms.InputTag("ctppsLocalTrackLiteProducer"),
  outputFile = cms.string("test_smearing_effects_sensor.root")
)

# processing path
process.p = cms.Path(
  process.generator
  * process.beamDivergenceVtxGenerator
  * process.ctppsDirectProtonSimulationNoSm
  * process.ctppsDirectProtonSimulationSm

  * process.totemRPUVPatternFinder
  * process.totemRPLocalTrackFitter
  * process.ctppsLocalTrackLiteProducer

  * process.ctppsDirectProtonSimulationValidatorBeamSm
  * process.ctppsDirectProtonSimulationValidatorSensorSm
)
