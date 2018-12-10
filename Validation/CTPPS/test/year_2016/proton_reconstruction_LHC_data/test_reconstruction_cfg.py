import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process("CTPPSTestProtonReconstruction", eras.ctpps_2016)

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
  )
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

# load LHCInfo
process.load("Validation.CTPPS.year_2016.ctppsLHCInfoESSource_cfi")

# load optics
process.load("Validation.CTPPS.year_2016.ctppsOpticalFunctionsESSource_cfi")

# load reconstruction sequences
process.load("RecoCTPPS.Configuration.recoCTPPS_sequences_cff")

process.ctppsLocalTrackLiteProducer.includeStrips = True
process.ctppsLocalTrackLiteProducer.includeDiamonds = False
process.ctppsLocalTrackLiteProducer.includePixels = False 

process.ctppsProtonReconstruction.verbosity = 0

# load geometry
process.load("Geometry.VeryForwardGeometry.geometryRPFromDD_2017_cfi") # 2017 is OK here

# load alignment corrections
process.ctppsIncludeAlignmentsFromXML.RealFiles += cms.vstring("Validation/CTPPS/test/year_2016/alignment_export_2018_12_07.1.xml")

# reconstruction validator
process.ctppsProtonReconstructionValidator = cms.EDAnalyzer("CTPPSProtonReconstructionValidator",
    tagTracks = cms.InputTag("ctppsLocalTrackLiteProducer"),
    tagRecoProtons = cms.InputTag("ctppsProtonReconstruction", "multiRP"),

    chiSqCut = cms.double(2.),

    outputFile = cms.string("test_recontruction_validation.root")
)

# reconstruction plotter
process.ctppsProtonReconstructionPlotter = cms.EDAnalyzer("CTPPSProtonReconstructionPlotter",
    tagTracks = cms.InputTag("ctppsLocalTrackLiteProducer"),
    tagRecoProtonsSingleRP = cms.InputTag("ctppsProtonReconstruction", "singleRP"),
    tagRecoProtonsMultiRP = cms.InputTag("ctppsProtonReconstruction", "multiRP"),

    rpId_45_F = cms.uint32(3),
    rpId_45_N = cms.uint32(2),
    rpId_56_N = cms.uint32(102),
    rpId_56_F = cms.uint32(103),

    outputFile = cms.string("test_reconstruction_plots.root")
)

# processing sequence
process.p = cms.Path(
    process.totemRPUVPatternFinder
    * process.totemRPLocalTrackFitter

    * process.ctppsLocalTrackLiteProducer
    * process.ctppsProtonReconstruction

    * process.ctppsProtonReconstructionValidator
    * process.ctppsProtonReconstructionPlotter
)
