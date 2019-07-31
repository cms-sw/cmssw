import FWCore.ParameterSet.Config as cms

process = cms.Process("testDigi")

# Specify the maximum events to simulate
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery=cms.untracked.int32(5000)

################## STEP 1 - process.generator

process.source = cms.Source("PoolSource",
duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
     fileNames = 
 cms.untracked.vstring(
#   'file:/tmp/andriusj/6EC8FCC8-E2A8-E411-9506-002590596468.root'
#        '/store/relval/CMSSW_7_4_0_pre6/RelValPhotonJets_Pt_10_13/GEN-SIM-RECO/MCRUN2_74_V1-v1/00000/6EC8FCC8-E2A8-E411-9506-002590596468.root'
#     'file:/home/ferro/ferroCMS/CMSSW_7_5_0/src/MYtest441_.root',
#    'file:/home/ferro/ferroCMS/CMSSW_7_5_0/src/MYtest442_.root',
#     'file:/home/ferro/ferroCMS/CMSSW_7_5_0/src/MYtest444_.root',
#     'file:/home/ferro/ferroCMS/CMSSW_7_5_0/src/MYtest445_.root',
#     'file:/home/ferro/ferroCMS/CMSSW_7_5_0/src/MYtest447_.root',
#     'file:/home/ferro/ferroCMS/CMSSW_7_5_0/src/MYtest448_.root',
     'file:/home/ferro/ferroCMS/CMSSW_7_5_0_myCTPPS/CMSSW_7_5_0/src/MYtest44_.root'
  )
)


# Use random number generator service
process.load("Configuration.TotemCommon.RandomNumbers_cfi")
process.RandomNumberGeneratorService.RPixDetDigitizer = cms.PSet(initialSeed =cms.untracked.uint32(137137))


process.load("TotemAnalysis.TotemNtuplizer.TotemNtuplizer_cfi")
process.TotemNtuplizer.outputFileName = "test.ntuple.root"
process.TotemNtuplizer.RawEventLabel = 'source'
process.TotemNtuplizer.RPReconstructedProtonCollectionLabel = cms.InputTag('RP220Reconst')
process.TotemNtuplizer.RPReconstructedProtonPairCollectionLabel = cms.InputTag('RP220Reconst')
process.TotemNtuplizer.RPMulFittedTrackCollectionLabel = cms.InputTag("RPMulTrackNonParallelCandCollFit")
process.TotemNtuplizer.includeDigi = cms.bool(True)
process.TotemNtuplizer.includePatterns = cms.bool(True)


process.digiAnal = cms.EDAnalyzer("CTPPSPixelDigiAnalyzer",
      label=cms.untracked.string("RPixDetDigitizer"),
     Verbosity = cms.int32(0),
   RPixVerbosity = cms.int32(0),
   RPixActiveEdgeSmearing = cms.double(0.020),
    RPixActiveEdgePosition = cms.double(0.150)

)

process.p1 = cms.Path( process.digiAnal )


