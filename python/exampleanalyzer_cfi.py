import FWCore.ParameterSet.Config as cms

demo = cms.EDAnalyzer('ExampleAnalyzer',
caloMETlabel_       = cms.InputTag("met"),    
tcMETlabel_         = cms.InputTag("tcMet"),      
muCorrMETlabel_     = cms.InputTag("corMetGlobalMuons"),  
pfMETlabel_         = cms.InputTag("pfMet"),      
muJESCorrMETlabel_  = cms.InputTag("metMuonJESCorSC5"),
muonLabel_          = cms.InputTag("muons"),
muValueMaplabel_    = cms.InputTag("muonMETValueMapProducer", "muCorrData"),  
tcMETValueMaplabel_ = cms.InputTag("muonTCMETValueMapProducer", "muCorrData")
)
