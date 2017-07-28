import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
process = cms.Process('CTPPSFastSimulation', eras.ctpps_2016)

# minimal logger settings
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cerr'),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
)

# number of events
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

# particle-data table
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# particle generator
process.generator = cms.EDProducer("RandomtXiGunProducer",
  Verbosity = cms.untracked.int32(0),

  FireBackward = cms.bool(True),
  FireForward = cms.bool(True),

  PGunParameters = cms.PSet(
    PartID = cms.vint32(2212),
    ECMS = cms.double(13E3),

    Mint = cms.double(0),
    Maxt = cms.double(1),
    MinXi = cms.double(0.0),
    MaxXi = cms.double(0.1),

    MinPhi = cms.double(-3.14159265359),
    MaxPhi = cms.double(+3.14159265359)
  )
)

# random seeds
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    sourceSeed = cms.PSet(initialSeed =cms.untracked.uint32(98765)),
    generator = cms.PSet(initialSeed = cms.untracked.uint32(98766)),
    SmearingGenerator = cms.PSet(initialSeed =cms.untracked.uint32(3849))
)

# geometry
from Geometry.VeryForwardGeometry.geometryRP_cfi import totemGeomXMLFiles, ctppsDiamondGeomXMLFiles

process.XMLIdealGeometryESSource_CTPPS = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = totemGeomXMLFiles+ctppsDiamondGeomXMLFiles+["Validation/CTPPS/test/fast_simu_with_phys_generator/qgsjet/global/RP_Dist_Beam_Cent.xml"],
    rootNodeName = cms.string('cms:CMSE'),
)

process.TotemRPGeometryESModule = cms.ESProducer("TotemRPGeometryESModule")

# alternative association between RPs and optics approximators
from SimCTPPS.OpticsParameterisation.ctppsDetectorPackages_cff import genericStripsPackage
detectorPackagesLong = cms.VPSet(
    #----- sector 45
    genericStripsPackage.clone(
        potId = cms.uint32(0x76100000), # 002
        interpolatorName = cms.string('ip5_to_station_150_h_1_lhcb2'),
        zPosition = cms.double(-203.826),
    ),
    genericStripsPackage.clone(
        potId = cms.uint32(0x76180000), # 003
        interpolatorName = cms.string('ip5_to_station_150_h_1_lhcb2'),
        zPosition = cms.double(-203.826),
    ),

    #----- sector 56
    genericStripsPackage.clone(
        potId = cms.uint32(0x77100000), # 102
        interpolatorName = cms.string('ip5_to_station_150_h_1_lhcb1'),
        zPosition = cms.double(+203.826),
    ),
    genericStripsPackage.clone(
        potId = cms.uint32(0x77180000), # 103
        interpolatorName = cms.string('ip5_to_station_150_h_1_lhcb1'),
        zPosition = cms.double(+203.826),
    ),
)

# fast simulation
from SimCTPPS.OpticsParameterisation.ctppsFastProtonSimulation_cfi import ctppsFastProtonSimulation
ctppsFastProtonSimulation.checkApertures = True
ctppsFastProtonSimulation.produceHitsRelativeToBeam = False
ctppsFastProtonSimulation.roundToPitch = False

process.ctppsFastProtonSimulationStd = ctppsFastProtonSimulation.clone(
    produceScoringPlaneHits = cms.bool(True),
    produceRecHits = cms.bool(False)
)

process.ctppsFastProtonSimulationLong = ctppsFastProtonSimulation.clone(
    produceScoringPlaneHits = cms.bool(False),
    produceRecHits = cms.bool(True),
    detectorPackages = detectorPackagesLong
)

# strips reco: pattern recognition
process.load('RecoCTPPS.TotemRPLocal.totemRPUVPatternFinder_cfi')
process.totemRPUVPatternFinder.tagRecHit = cms.InputTag('ctppsFastProtonSimulationLong')

# strips reco: track fitting
process.load('RecoCTPPS.TotemRPLocal.totemRPLocalTrackFitter_cfi')

# common reco: lite track production
process.load('RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cfi')

# distribution plotters
process.ctppsFastSimulationValidator = cms.EDAnalyzer("CTPPSFastSimulationValidator",
  simuTracksTag = cms.InputTag("ctppsFastProtonSimulationStd"),
  recoTracksTag = cms.InputTag("ctppsLocalTrackLiteProducer", ""),
  outputFile = cms.string("output_long_extrapolation.root")
)

# processing path
process.p = cms.Path(
    process.generator

    * process.ctppsFastProtonSimulationStd

    * process.ctppsFastProtonSimulationLong
    * process.totemRPUVPatternFinder
    * process.totemRPLocalTrackFitter
    * process.ctppsLocalTrackLiteProducer

    * process.ctppsFastSimulationValidator
)
