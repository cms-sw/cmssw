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
process.ctppsDirectProtonSimulation.verbosity = 0
process.ctppsDirectProtonSimulation.hepMCTag = cms.InputTag('beamDivergenceVtxGenerator')
process.ctppsDirectProtonSimulation.useEmpiricalApertures = False
process.ctppsDirectProtonSimulation.roundToPitch = True
process.ctppsDirectProtonSimulation.pitchStrips = 66E-3 * 12 / 19 # effective value to reproduce real RP resolution
process.ctppsDirectProtonSimulation.produceHitsRelativeToBeam = True
process.ctppsDirectProtonSimulation.produceScoringPlaneHits = False
process.ctppsDirectProtonSimulation.produceRecHits = True

# strips reco
process.load('RecoCTPPS.TotemRPLocal.totemRPUVPatternFinder_cfi')
process.totemRPUVPatternFinder.tagRecHit = cms.InputTag('ctppsDirectProtonSimulation')

process.load('RecoCTPPS.TotemRPLocal.totemRPLocalTrackFitter_cfi')

# common reco: lite track production
process.load('RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cff')
process.ctppsLocalTrackLiteProducer.includeDiamonds = False
process.ctppsLocalTrackLiteProducer.includePixels = False

# acceptance plotter
process.ctppsAcceptancePlotter = cms.EDAnalyzer("CTPPSAcceptancePlotter",
  tagHepMC = cms.InputTag("generator", "unsmeared"),
  tagTracks = cms.InputTag("ctppsLocalTrackLiteProducer"),

  rpId_45_F = cms.uint32(3),
  rpId_45_N = cms.uint32(2),
  rpId_56_N = cms.uint32(102),
  rpId_56_F = cms.uint32(103),

  outputFile = cms.string("test_acceptance.root")
)

# processing path
process.p = cms.Path(
  process.generator
  * process.beamDivergenceVtxGenerator
  * process.ctppsDirectProtonSimulation

  * process.totemRPUVPatternFinder
  * process.totemRPLocalTrackFitter
  * process.ctppsLocalTrackLiteProducer

  * process.ctppsAcceptancePlotter
)
