import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

# maximum number of events which will be analyzed
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(4000) )

# the input files holding RECO collections for events which have alreday been reconstructed
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_6_2_0_pre7/RelValZEE/GEN-SIM-DIGI-RECO/PRE_ST62_V7_FastSim-v3/00000/FE69F854-5BCE-E211-B4D6-00248C0BE018.root'
    )
    # file above was found with (many more files available):
    # dbs search --query='find dataset,file where dataset=/RelValZEE/CMSSW_6_2_0_pre7-PRE_ST62_V7_FastSim-v3/GEN-SIM-DIGI-RECO '                            
)

# import the settings of your analyzer from its own configuration file
process.load("TimeEventStudies.TimeAnalyzer.CfiFile_cfi")
# service needed to write out .root file with histograms
# the name of the root file which will hold your histograms
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('TimeAnalyzer_output.root')
                                   )


# EventContentAnalyzer prints the  list  of all the collections present in the event
# needed only if you're looking for a specific collection you don't know the name of 
process.dumpEvContent = cms.EDAnalyzer("EventContentAnalyzer")


# the schedule of the process: list of modules the CMSSW framwwork will execute at every event 
process.p = cms.Path(
    process.TimeAnalysis
    # * process.dumpEvContent
    )
