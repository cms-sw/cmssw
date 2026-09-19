# Truth dump of the ionization loss along signal tracks (BLElossTruthDump) on the gate's TTbar PU200 input.
#   cmsRun blElossTruthDump_cfg.py maxEvents=200 outputFile=truth.root
import glob
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9

options = VarParsing('analysis')
options.inputFiles = ["file:" + f for f in sorted(glob.glob("/scratch/TTbar_20_X/*.root"))]
options.outputFile = "truth.root"
options.maxEvents = 200
options.parseArguments()

process = cms.Process("TRUTH", Phase2C22I13M9)
process.load('Configuration.Geometry.GeometryExtendedRun4D121Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T35', '')
process.MessageLogger.cerr.FwkReport.reportEvery = 20

process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring(options.inputFiles))
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(options.maxEvents))
process.TFileService = cms.Service("TFileService", fileName=cms.string(options.outputFile))
process.dump = cms.EDAnalyzer("BLElossTruthDump",
    simTracks=cms.InputTag("g4SimHits"), simVertices=cms.InputTag("g4SimHits"), minPt=cms.double(0.5),
    simHits=cms.VInputTag(*[cms.InputTag("g4SimHits", "TrackerHits%s%sTof" % (d, t))
                            for d in ("PixelBarrel", "PixelEndcap", "TOB", "TID", "TIB", "TEC") for t in ("Low", "High")]))
process.p = cms.Path(process.dump)
