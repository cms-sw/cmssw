import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("MCvertices")

#prepare options

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "DONOTEXIST",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")
#options.globalTag = "DONOTEXIST"

options.parseArguments()

#
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    fileMode = cms.untracked.string("FULLMERGE")
    )

process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string("INFO")
process.MessageLogger.cout.default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    )
process.MessageLogger.cout.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(10000)
    )

process.MessageLogger.cerr.enable = cms.untracked.bool(True)
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


process.source.fileNames = cms.untracked.vstring(options.inputFiles)

#process.load("Validation.RecoVertex.pvRecoSequence_cff")
process.load("Validation.RecoVertex.pvSelectionSequence_cff")

#process.load("Validation.RecoVertex.mcvertexweight_cfi")
process.load("Validation.RecoVertex.mcverticesanalyzer_cfi")
process.load("Validation.RecoVertex.mcvsrecoverticesanalyzer_cfi")

#process.mcvertexweight45mm = process.mcvertexweight.clone()
#process.mcvertexweight45mm.weighterConfig.finalSigma = cms.double(4.5)

process.mcvsreconofakeverticesanalyzer = process.mcvsrecoverticesanalyzer.clone(pvCollection = cms.InputTag("noFakeVertices"))
process.mcvsrecogoodverticesanalyzer = process.mcvsrecoverticesanalyzer.clone(pvCollection = cms.InputTag("goodVertices"))

process.mcvsrecoverticesanalyzervisible = process.mcvsrecoverticesanalyzer.clone(useVisibleVertices = cms.bool(True))
process.mcvsreconofakeverticesanalyzervisible = process.mcvsreconofakeverticesanalyzer.clone(useVisibleVertices = cms.bool(True))
process.mcvsrecogoodverticesanalyzervisible = process.mcvsrecogoodverticesanalyzer.clone(useVisibleVertices = cms.bool(True))
#process.mcvsrecogoodverticesD0s5 = process.mcvsrecoverticesanalyzer.clone(pvCollection = cms.InputTag("goodVerticesD0s5"))
#process.mcvsrecogoodverticesD0s51mm = process.mcvsrecoverticesanalyzer.clone(pvCollection = cms.InputTag("goodVerticesD0s51mm"))
#process.mcvsrecogoodverticesDA100um = process.mcvsrecoverticesanalyzer.clone(pvCollection = cms.InputTag("goodVerticesDA100um"))
#process.mcvsrecogoodverticesDA100umV7 = process.mcvsrecoverticesanalyzer.clone(pvCollection = cms.InputTag("goodVerticesDA100umV7"))
#process.mcvsrecogoodverticesDA100umV8 = process.mcvsrecoverticesanalyzer.clone(pvCollection = cms.InputTag("goodVerticesDA100umV8"))


#process.weightedmcverticesanalyzer = process.mcverticesanalyzer.clone(useWeight = cms.bool(True))
#process.weightedmcvsrecoverticesanalyzer = process.mcvsrecoverticesanalyzer.clone(useWeight = cms.bool(True))
#process.weightedmcvsreconofakeverticesanalyzer = process.mcvsreconofakeverticesanalyzer.clone(useWeight = cms.bool(True))
#process.weightedmcvsrecogoodverticesanalyzer = process.mcvsrecogoodverticesanalyzer.clone(useWeight = cms.bool(True))
#process.weightedmcvsrecogoodverticesD0s5 = process.mcvsrecogoodverticesD0s5.clone(useWeight = cms.bool(True))
#process.weightedmcvsrecogoodverticesD0s51mm = process.mcvsrecogoodverticesD0s51mm.clone(useWeight = cms.bool(True))
#process.weightedmcvsrecogoodverticesDA100um = process.mcvsrecogoodverticesDA100um.clone(useWeight = cms.bool(True))
#process.weightedmcvsrecogoodverticesDA100umV7 = process.mcvsrecogoodverticesDA100umV7.clone(useWeight = cms.bool(True))
#process.weightedmcvsrecogoodverticesDA100umV8 = process.mcvsrecogoodverticesDA100umV8.clone(useWeight = cms.bool(True))


#process.weighted45mmmcverticesanalyzer = process.mcverticesanalyzer.clone(useWeight = cms.bool(True), weightProduct=cms.InputTag("mcvertexweight45mm"))
#process.weighted45mmmcvsrecoverticesanalyzer = process.mcvsrecoverticesanalyzer.clone(useWeight = cms.bool(True), weightProduct=cms.InputTag("mcvertexweight45mm"))
#process.weighted45mmmcvsreconofakeverticesanalyzer = process.mcvsreconofakeverticesanalyzer.clone(useWeight = cms.bool(True), weightProduct=cms.InputTag("mcvertexweight45mm"))
#process.weighted45mmmcvsrecogoodverticesanalyzer = process.mcvsrecogoodverticesanalyzer.clone(useWeight = cms.bool(True), weightProduct=cms.InputTag("mcvertexweight45mm"))
#process.weighted45mmmcvsrecogoodverticesD0s5 = process.mcvsrecogoodverticesD0s5.clone(useWeight = cms.bool(True), weightProduct=cms.InputTag("mcvertexweight45mm"))
#process.weighted45mmmcvsrecogoodverticesD0s51mm = process.mcvsrecogoodverticesD0s51mm.clone(useWeight = cms.bool(True), weightProduct=cms.InputTag("mcvertexweight45mm"))
#process.weighted45mmmcvsrecogoodverticesDA100um = process.mcvsrecogoodverticesDA100um.clone(useWeight = cms.bool(True), weightProduct=cms.InputTag("mcvertexweight45mm"))
#process.weighted45mmmcvsrecogoodverticesDA100umV7 = process.mcvsrecogoodverticesDA100umV7.clone(useWeight = cms.bool(True), weightProduct=cms.InputTag("mcvertexweight45mm"))
#process.weighted45mmmcvsrecogoodverticesDA100umV8 = process.mcvsrecogoodverticesDA100umV8.clone(useWeight = cms.bool(True), weightProduct=cms.InputTag("mcvertexweight45mm"))

process.load("Validation.RecoVertex.anotherprimaryvertexanalyzer_cfi")

process.primaryvertexanalyzer.vHistogramMakerPSet.runHisto=cms.untracked.bool(False)
process.primaryvertexanalyzer.vHistogramMakerPSet.histoParameters = cms.untracked.PSet(
    nBinX = cms.untracked.uint32(2000), xMin=cms.untracked.double(-0.5), xMax=cms.untracked.double(0.5),
    nBinY = cms.untracked.uint32(2000), yMin=cms.untracked.double(-0.5), yMax=cms.untracked.double(0.5),
    nBinZ = cms.untracked.uint32(300), zMin=cms.untracked.double(-30.), zMax=cms.untracked.double(30.)
    )

process.pvnofakevertices = process.primaryvertexanalyzer.clone(pvCollection=cms.InputTag("noFakeVertices"))
process.pvgoodvertices = process.primaryvertexanalyzer.clone(pvCollection=cms.InputTag("goodVertices"))
#process.pvgoodverticesD0s5 = process.primaryvertexanalyzer.clone(pvCollection=cms.InputTag("goodVerticesD0s5"))
#process.pvgoodverticesD0s51mm = process.primaryvertexanalyzer.clone(pvCollection=cms.InputTag("goodVerticesD0s51mm"))
#process.pvgoodverticesDA100um = process.primaryvertexanalyzer.clone(pvCollection=cms.InputTag("goodVerticesDA100um"))
#process.pvgoodverticesDA100umV7 = process.primaryvertexanalyzer.clone(pvCollection=cms.InputTag("goodVerticesDA100umV7"))
#process.pvgoodverticesDA100umV8 = process.primaryvertexanalyzer.clone(pvCollection=cms.InputTag("goodVerticesDA100umV8"))


process.p626 = cms.Path(
#    process.seqPVReco +
    process.seqSimplePVSelection 
    + process.mcverticesanalyzer
    + process.mcvsrecoverticesanalyzer + process.mcvsreconofakeverticesanalyzer + process.mcvsrecogoodverticesanalyzer
    + process.mcvsrecoverticesanalyzervisible + process.mcvsreconofakeverticesanalyzervisible + process.mcvsrecogoodverticesanalyzervisible
#    +  process.mcvsrecogoodverticesD0s5 +process.mcvsrecogoodverticesD0s51mm +
#    process.mcvsrecogoodverticesDA100um +process.mcvsrecogoodverticesDA100umV7 +process.mcvsrecogoodverticesDA100umV8 +
    + process.pvnofakevertices + process.pvgoodvertices 
#    + process.pvgoodverticesD0s5 + process.pvgoodverticesD0s51mm 
#    + process.pvgoodverticesDA100um + process.pvgoodverticesDA100umV7 + process.pvgoodverticesDA100umV8
                        ) 

#process.p520 = cms.Path(process.mcvertexweight+
#                        process.seqPVReco + process.seqPVSelection +
#                        process.weightedmcverticesanalyzer + process.weightedmcvsrecoverticesanalyzer +
#                        process.weightedmcvsreconofakeverticesanalyzer + process.weightedmcvsrecogoodverticesanalyzer +
#                        process.weightedmcvsrecogoodverticesD0s5 +process.weightedmcvsrecogoodverticesD0s51mm +
#                        process.weightedmcvsrecogoodverticesDA100um + process.weightedmcvsrecogoodverticesDA100umV7 + process.weightedmcvsrecogoodverticesDA100umV8
#                        )

#process.p450 = cms.Path(process.mcvertexweight45mm+
#                        process.seqPVReco + process.seqPVSelection +
#                        process.weighted45mmmcverticesanalyzer + process.weighted45mmmcvsrecoverticesanalyzer +
#                        process.weighted45mmmcvsreconofakeverticesanalyzer + process.weighted45mmmcvsrecogoodverticesanalyzer +
#                        process.weighted45mmmcvsrecogoodverticesD0s5 +process.weighted45mmmcvsrecogoodverticesD0s51mm +
#                        process.weighted45mmmcvsrecogoodverticesDA100um + process.weighted45mmmcvsrecogoodverticesDA100umV7 + process.weighted45mmmcvsrecogoodverticesDA100umV8
#                        )

#----GlobalTag ------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')


process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('mcvertices.root')
                                   )

