import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("MCvertices")

#prepare options

options = VarParsing.VarParsing()

options.register ('globalTag',
                  "DONOTEXIST::All",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")
#options.globalTag = "DONOTEXIST::All"

options.parseArguments()

#
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    fileMode = cms.untracked.string("FULLMERGE")
    )

process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string("INFO")
process.MessageLogger.cout.default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    )
process.MessageLogger.cout.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(10000)
    )

process.MessageLogger.cerr.placeholder = cms.untracked.bool(False)
process.MessageLogger.cerr.threshold = cms.untracked.string("WARNING")
process.MessageLogger.cerr.default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    )
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(100000)
    )

#----Remove too verbose PrimaryVertexProducer

process.MessageLogger.suppressInfo.append("pixelVerticesAdaptive")
process.MessageLogger.suppressInfo.append("pixelVerticesAdaptiveNoBS")

#----Remove too verbose BeamSpotOnlineProducer

process.MessageLogger.suppressInfo.append("testBeamSpot")
process.MessageLogger.suppressInfo.append("onlineBeamSpot")
process.MessageLogger.suppressWarning.append("testBeamSpot")
process.MessageLogger.suppressWarning.append("onlineBeamSpot")

#----Remove too verbose TrackRefitter

process.MessageLogger.suppressInfo.append("newTracksFromV0")
process.MessageLogger.suppressInfo.append("newTracksFromOtobV0")


#------------------------------------------------------------------

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
                    fileNames = cms.untracked.vstring(),
#                    skipBadFiles = cms.untracked.bool(True),
                    inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")
                    )


process.source.fileNames = cms.untracked.vstring(
"/store/relval/CMSSW_4_3_0_pre1/RelValZTT/GEN-SIM-RECO/MC_42_V7-v1/0053/1640096C-9E59-E011-BBE5-001A92971B8A.root",
"/store/relval/CMSSW_4_3_0_pre1/RelValZTT/GEN-SIM-RECO/MC_42_V7-v1/0048/F0BF09EC-0559-E011-AFD4-0018F3D0966C.root",
"/store/relval/CMSSW_4_3_0_pre1/RelValZTT/GEN-SIM-RECO/MC_42_V7-v1/0048/C85057F9-0A59-E011-936D-0030486792B6.root"
)

process.load("Validation.RecoVertex.bspvanalyzer_cfi")
process.bspvanalyzer.pvCollection = cms.InputTag("goodVertices")
process.bspvanalyzer.bspvHistogramMakerPSet.histoParameters = cms.untracked.PSet(
    nBinX = cms.untracked.uint32(2000), xMin=cms.untracked.double(-0.2), xMax=cms.untracked.double(0.2),
    nBinY = cms.untracked.uint32(2000), yMin=cms.untracked.double(-0.2), yMax=cms.untracked.double(0.2),
    nBinZ = cms.untracked.uint32(200), zMin=cms.untracked.double(-30.), zMax=cms.untracked.double(30.),
    nBinZProfile = cms.untracked.uint32(60), zMinProfile=cms.untracked.double(-30.), zMaxProfile=cms.untracked.double(30.)
    )
process.bspvanalyzer.bspvHistogramMakerPSet.runHisto = cms.untracked.bool(False)

process.bspvnoslope = process.bspvanalyzer.clone()
process.bspvnoslope.bspvHistogramMakerPSet.useSlope = cms.bool(False)


process.load("Validation.RecoVertex.pvSelectionSequence_cff")


process.p0 = cms.Path(process.goodVertices + process.bspvanalyzer + process.bspvnoslope)

#----GlobalTag ------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = options.globalTag


process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('bspvanalyzer.root')
                                   )

