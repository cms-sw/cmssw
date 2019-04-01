import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process("CTPPSTestProtonReconstruction", eras.ctpps_2016)

# declare global tag
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, "auto:run2_data")

# TODO: these lines can be useful before all necessary data available in DB with an auto GT
#process.GlobalTag = GlobalTag(process.GlobalTag, "105X_dataRun2_relval_v2")
#process.alignmentEsPrefer = cms.ESPrefer("PoolDBESSource", "GlobalTag")

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
  statistics = cms.untracked.vstring(),
  destinations = cms.untracked.vstring('cout'),
  cout = cms.untracked.PSet(
    threshold = cms.untracked.string('WARNING')
  )
)

# data source
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    # more at: https://cmsweb.cern.ch/das/request?view=list&limit=50&instance=prod%2Fglobal&input=file+dataset%3D%2FBTagCSV%2FRun2016C-07Aug17-v1%2FAOD
    "/store/data/Run2016C/BTagCSV/AOD/07Aug17-v1/110000/0026FCD2-369A-E711-920C-0025905A607E.root"
  ),
  inputCommands = cms.untracked.vstring(
    "keep *",
    "drop CTPPSLocalTrackLites_*_*_*"
  )
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

# load reconstruction sequences
process.load("RecoCTPPS.Configuration.recoCTPPS_cff")
process.ctppsLocalTrackLiteProducer.includeDiamonds = False
process.ctppsLocalTrackLiteProducer.includePixels = False

process.ctppsProtons.verbosity = 0

# reconstruction validator
process.ctppsProtonReconstructionValidator = cms.EDAnalyzer("CTPPSProtonReconstructionValidator",
    tagTracks = cms.InputTag("ctppsLocalTrackLiteProducer"),
    tagRecoProtons = cms.InputTag("ctppsProtons", "multiRP"),

    chiSqCut = cms.double(2.),

    outputFile = cms.string("test_reconstruction_validation.root")
)

# reconstruction plotter
process.ctppsProtonReconstructionPlotter = cms.EDAnalyzer("CTPPSProtonReconstructionPlotter",
    tagTracks = cms.InputTag("ctppsLocalTrackLiteProducer"),
    tagRecoProtonsSingleRP = cms.InputTag("ctppsProtons", "singleRP"),
    tagRecoProtonsMultiRP = cms.InputTag("ctppsProtons", "multiRP"),

    rpId_45_F = cms.uint32(3),
    rpId_45_N = cms.uint32(2),
    rpId_56_N = cms.uint32(102),
    rpId_56_F = cms.uint32(103),

    outputFile = cms.string("test_reconstruction_plots.root")
)

# processing sequence
process.p = cms.Path(
    process.totemRPLocalReconstruction
    * process.ctppsDiamondLocalReconstruction
    #* process.totemTimingLocalReconstruction
    #* process.ctppsPixelLocalReconstruction
    * process.ctppsLocalTrackLiteProducer
    * process.ctppsProtons

    * process.ctppsProtonReconstructionValidator
    * process.ctppsProtonReconstructionPlotter
)
