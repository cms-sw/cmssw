import FWCore.ParameterSet.Config as cms
process = cms.Process("CTPPSFastSimulationTest")

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cerr'),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
)


# event source
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

# random seeds
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
  generator = cms.PSet(
    initialSeed = cms.untracked.uint32(36)
  )
)


# particle-data table
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")


# particle generator
process.generator = cms.EDProducer("RandomtXiGunProducer",
  Verbosity = cms.untracked.int32(10),

  FireBackward = cms.bool(True),
  FireForward = cms.bool(True),

  PGunParameters = cms.PSet(
    PartID = cms.vint32(2212),
    ECMS = cms.double(13E3),

    Mint = cms.double(0),
    Maxt = cms.double(0.1),
    MinXi = cms.double(0.05),
    MaxXi = cms.double(0.010),

    MinPhi = cms.double(-3.14159265359),
    MaxPhi = cms.double(+3.14159265359)
  )
)


# geometry
process.load("Geometry.VeryForwardGeometry.geometryRP_cfi")

# RP simulation
process.ctppsFastProtonSimulation = cms.EDProducer("CTPPSFastProtonSimulation",
  verbosity = cms.untracked.uint32(10),

  tagHepMC = cms.InputTag("generator", "unsmeared"),

  thetaLimit = cms.double(1E-3),  # rad

  opticsFile_45 = cms.string("../../../RecoCTPPS/OpticsParametrization/data/2016_preTS2/version4-vale1/beam2/parametrization_6500GeV_0p4_185_reco.root"),
  opticsObject_45 = cms.string("ip5_to_station_220_h_1_lhcb2"),
  opticsZ0_45 = cms.double(-215077), # mm

  opticsFile_56 = cms.string("../../../RecoCTPPS/OpticsParametrization/data/2016_preTS2/version4-vale1/beam1/parametrization_6500GeV_0p4_185_reco.root"),
  opticsObject_56 = cms.string("ip5_to_station_220_h_1_lhcb1"),
  opticsZ0_56 = cms.double(+215077), # mm

  # in m
  vtx0_y_45 = cms.double(300E-6),
  vtx0_y_56 = cms.double(200E-6),

  # in rad
  half_crossing_angle_45 = cms.double(+179.394E-6),
  half_crossing_angle_56 = cms.double(+191.541E-6),

  roundToPitch = cms.bool(False),
  pitch = cms.double(66E-3), # mm

  insensitiveMargin = cms.double(34E-3)  # mm
)

process.eca = cms.EDAnalyzer("EventContentAnalyzer")

# processing sequence
process.p = cms.Path(
    process.generator
    #* process.eca
    * process.ctppsFastProtonSimulation
)
