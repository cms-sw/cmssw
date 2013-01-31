import FWCore.ParameterSet.Config as cms

process = cms.Process("runGenMuonRadCorrAnalyzer")

import os
import re

import TauAnalysis.Configuration.tools.castor as castor
import TauAnalysis.Configuration.tools.eos as eos

# import of standard configurations for RECOnstruction
# of electrons, muons and tau-jets with non-standard isolation cones
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.load('Configuration/Geometry/GeometryIdeal_cff')
process.load('Configuration/StandardSequences/MagneticField_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = cms.string('START53_V7A::All')

#--------------------------------------------------------------------------------
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        ##'file:/data1/veelken/CMSSW_5_3_x/skims/ZmumuTF_RECO_2012Oct03.root'
        '/store/user/veelken/CMSSW_5_3_x/skims/Embedding/goldenZmumuEvents_ZplusJets_madgraph_RECO_205_1_XhE.root',
        '/store/user/veelken/CMSSW_5_3_x/skims/Embedding/goldenZmumuEvents_ZplusJets_madgraph_RECO_206_1_OHz.root',
        '/store/user/veelken/CMSSW_5_3_x/skims/Embedding/goldenZmumuEvents_ZplusJets_madgraph_RECO_207_1_bgM.root',
        '/store/user/veelken/CMSSW_5_3_x/skims/Embedding/goldenZmumuEvents_ZplusJets_madgraph_RECO_208_1_szL.root',
        '/store/user/veelken/CMSSW_5_3_x/skims/Embedding/goldenZmumuEvents_ZplusJets_madgraph_RECO_209_1_Jqv.root'
    ),
    skipEvents = cms.untracked.uint32(0)            
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# set input files
inputFilePath = '/store/user/veelken/CMSSW_5_3_x/skims/GoldenZmumu/2012Oct09/'
inputFile_regex = r"[a-zA-Z0-9_/:.]*goldenZmumuEvents_ZplusJets_madgraph_2012Oct09_AOD_(?P<gridJob>\d*)(_(?P<gridTry>\d*))*_(?P<hash>[a-zA-Z0-9]*).root"

# check if name of inputFile matches regular expression
inputFileNames = []
files = None
if inputFilePath.startswith('/castor/'):
    files = [ "".join([ "rfio:", file_info['path'] ]) for file_info in castor.nslsl(inputFilePath) ]
elif inputFilePath.startswith('/store/'):
    files = [ file_info['path'] for file_info in eos.lsl(inputFilePath) ]
else:
    files = [ "".join([ "file:", inputFilePath, file ]) for file in os.listdir(inputFilePath) ]
for file in files:
    #print "file = %s" % file
    inputFile_matcher = re.compile(inputFile_regex)
    if inputFile_matcher.match(file):
        inputFileNames.append(file)
#print "inputFileNames = %s" % inputFileNames 

process.source.fileNames = cms.untracked.vstring(inputFileNames)
#--------------------------------------------------------------------------------

process.load("TauAnalysis/MCEmbeddingTools/ZmumuStandaloneSelection_cff")
process.goldenZmumuFilter.src = cms.InputTag('goldenZmumuCandidatesGe0IsoMuons')

process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.genMuonRadCorrAnalyzer = cms.PSet(
    initialSeed = cms.untracked.uint32(12345),
    engineName = cms.untracked.string('TRandom3')
)
process.RandomNumberGeneratorService.genMuonRadCorrAnalyzerPYTHIA = process.RandomNumberGeneratorService.genMuonRadCorrAnalyzer.clone()
process.RandomNumberGeneratorService.genMuonRadCorrAnalyzerPHOTOS = process.RandomNumberGeneratorService.genMuonRadCorrAnalyzer.clone()

process.load("TauAnalysis/MCEmbeddingTools/genMuonRadCorrAnalyzer_cfi")
from Configuration.Generator.PythiaUEZ2starSettings_cfi import *
process.genMuonRadCorrAnalyzerPYTHIA.PythiaParameters = cms.PSet(
    pythiaUESettingsBlock,                      
    particleGunParameters = cms.vstring(
        'MSTP(41) = 0 ! Disable parton Showers',
        'MSTP(61) = 0 ! Disable initial state radiation',
        'MSTP(71) = 1 ! Enable final state radiation'
    ),
    parameterSets = cms.vstring(
        'pythiaUESettings',                                   
        'particleGunParameters'
    )
)
process.genMuonRadCorrAnalyzerPHOTOS.PhotosOptions = cms.PSet()
process.genMuonRadCorrAnalyzerSequence = cms.Sequence(
    process.genMuonRadCorrAnalyzer
   + process.genMuonRadCorrAnalyzerPYTHIA
   + process.genMuonRadCorrAnalyzerPHOTOS
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('/data1/veelken/tmp/runGenMuonRadCorrAnalyzer_2013Jan28.root')
)

process.analysisSequence = cms.Sequence(
    process.goldenZmumuSelectionSequence    
   + process.goldenZmumuFilter
   + process.genMuonRadCorrAnalyzerSequence
)
#--------------------------------------------------------------------------------

process.p = cms.Path(process.analysisSequence)

processDumpFile = open('runGenMuonRadCorrAnalyzer.dump' , 'w')
print >> processDumpFile, process.dumpPython()
