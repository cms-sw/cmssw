# HGCAL digis from noise only: EmptySource events, no sim hits, so every digi comes from the noise table.
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
from Configuration.Eras.Era_Phase2C26I13M9_cff import Phase2C26I13M9

options = VarParsing.VarParsing()
options.register("streams", 1, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,
                 "number of streams and threads")
options.register("events", 8, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,
                 "number of events")
options.register("seedingMode", "", VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string,
                 "seedingMode of the RandomNumberGeneratorService; empty keeps the default of the era")
options.register("noiseTableFromJobSeed", True, VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool, "fill the noise table from the job seed")
options.register("digest", "hgcalNoiseDigest.txt", VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string, "digest file")
options.parseArguments()

process = cms.Process("DIGI", Phase2C26I13M9)
process.load("Configuration.StandardSequences.Services_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.Geometry.GeometryExtendedRun4D128Reco_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, "auto:phase2_realistic_T35", "")

process.options.numberOfThreads = options.streams
process.options.numberOfStreams = options.streams
# 8 events in 2 lumis of 4: with 4 streams, the streams start with different events.
process.source = cms.Source("EmptySource", numberEventsInLuminosityBlock=cms.untracked.uint32(4))
process.maxEvents.input = options.events
# The digitizers report the missing sim hits of every event.
process.MessageLogger.cerr.threshold = "WARNING"
process.MessageLogger.suppressError = cms.untracked.vstring("mix")

process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.mix.digitizers = cms.PSet(hgceeDigitizer=process.mix.digitizers.hgceeDigitizer,
                                  hgchefrontDigitizer=process.mix.digitizers.hgchefrontDigitizer,
                                  hgchebackDigitizer=process.mix.digitizers.hgchebackDigitizer)
process.mix.mixObjects = cms.PSet()
# The aliases of the other digitizers have no products.
for alias in list(process.aliases_()):
    delattr(process, alias)

if options.seedingMode:
    process.RandomNumberGeneratorService.seedingMode = cms.untracked.string(options.seedingMode)
for digitizer in (process.mix.digitizers.hgceeDigitizer, process.mix.digitizers.hgchefrontDigitizer,
                  process.mix.digitizers.hgchebackDigitizer):
    digitizer.noiseTableFromJobSeed = options.noiseTableFromJobSeed

process.hgcalDigiDigest = cms.EDAnalyzer("HGCalDigiDigest",
                                         digis=cms.VInputTag(cms.InputTag("mix", "HGCDigisEE"),
                                                             cms.InputTag("mix", "HGCDigisHEfront"),
                                                             cms.InputTag("mix", "HGCDigisHEback")),
                                         fileName=cms.untracked.string(options.digest))
process.p = cms.Path(process.mix + process.hgcalDigiDigest)
