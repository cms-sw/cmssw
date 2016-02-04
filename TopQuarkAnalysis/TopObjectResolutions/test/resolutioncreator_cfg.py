import FWCore.ParameterSet.Config as cms
  
process = cms.Process("Resolutions")

#Message Logger (more info see: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMessageLogger)
process.MessageLogger = cms.Service("MessageLogger",
   
    destinations = cms.untracked.vstring("InfoSummary_SC"), 

    debugModules = cms.untracked.vstring('*'), #for all modules

    categories = cms.untracked.vstring("inputChain","decayChain","MainResults"), # list of categories
                                                                                 # inputChain and decayChain come from the TopDecaySubsetModule and need to be suppressedin warning- and info-summary

    InfoSummary_SC = cms.untracked.PSet( threshold = cms.untracked.string("INFO"), categories = cms.untracked.vstring("MainResults"),
                                         inputChain = cms.untracked.PSet(limit = cms.untracked.int32(0)),
                                         decayChain = cms.untracked.PSet(limit = cms.untracked.int32(0)))
)

process.load("TopBrussels.SanityChecker.PATLayer1_Ttjets_MG_input_cfi")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100000) )

## std sequence to produce the ttGenEvt
process.load("TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff")

process.load("TopBrussels.SanityChecker.ResolutionChecker_cfi")
process.TFileService = cms.Service("TFileService",
        fileName = cms.string('Resolutions.root')
	)
	

process.p = cms.Path(process.makeGenEvt * (process.Resolutions_lJets + process.Resolutions_bJets + process.Resolutions_muons + process.Resolutions_electrons + process.Resolutions_met))
