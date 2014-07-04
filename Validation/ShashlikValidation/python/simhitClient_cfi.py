import FWCore.ParameterSet.Config as cms

shashlikSimHitClient = cms.EDAnalyzer("ShashlikSimHitClient", 
                                      OutputFile = cms.untracked.string(''),
                                      DQMDirName = cms.string("/") ,
)
