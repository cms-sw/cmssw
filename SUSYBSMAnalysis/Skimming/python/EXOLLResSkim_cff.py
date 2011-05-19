import FWCore.ParameterSet.Config as cms
 
### HLT filter
## import copy
## from HLTrigger.HLTfilters.hltHighLevel_cfi import *
## ZMuHLTFilter = copy.deepcopy(hltHighLevel)
## ZMuHLTFilter.throw = cms.bool(False)
## ZMuHLTFilter.HLTPaths = ["HLT_Mu9","HLT_Mu11","HLT_Mu15","HLT_Mu15_v*","HLT_IsoMu17_v*","HLT_Mu20_v*","HLT_Mu24_v*","HLT_DoubleMu*"]

goodExoLLMuons = cms.EDFilter("MuonRefSelector",
                              src = cms.InputTag("muons"),
                              cut = cms.string("pt > 30 && " +
                                               "isGlobalMuon && isTrackerMuon"
                                               ),
                              )


goodExoLLElectrons = cms.EDFilter("GsfElectronRefSelector",
                                  src = cms.InputTag("gsfElectrons"),
                                  cut = cms.string(    "pt > 30 " 
                                                       ),
                                  )

diExoLLMuons = cms.EDProducer("CandViewShallowCloneCombiner",
                              decay       = cms.string("goodExoLLMuons goodExoLLMuons"),
                              checkCharge = cms.bool(False),
                              cut         = cms.string("mass > 40"),
                              )

diExoLLElectrons = cms.EDProducer("CandViewShallowCloneCombiner",
                                  decay       = cms.string("goodExoLLElectrons goodExoLLElectrons"),
                                  checkCharge = cms.bool(False),
                                  cut         = cms.string("mass > 40"),
                                  )
crossExoLLLeptons  = cms.EDProducer("CandViewShallowCloneCombiner",
                                    decay       = cms.string("goodExoLLMuons goodExoLLElectrons"),
                                    checkCharge = cms.bool(False),
                                    cut         = cms.string("mass > 40"),
                                    )

diExoLLMuonsFilter = cms.EDFilter("CandViewCountFilter",
                                  src = cms.InputTag("diExoLLMuons"),
                                  minNumber = cms.uint32(1)
                                  )
diExoLLElectronsFilter = cms.EDFilter("CandViewCountFilter",
                                      src = cms.InputTag("diExoLLElectrons"),
                                      minNumber = cms.uint32(1)
                                      )
crossExoLLLeptonsFilter = cms.EDFilter("CandViewCountFilter",
                                       src = cms.InputTag("crossExoLLLeptons"),
                                       minNumber = cms.uint32(1)
                                       )

exoLLResdiMuonSequence = cms.Sequence( goodExoLLMuons * diExoLLMuons * diExoLLMuonsFilter )

exoLLResdiElectronSequence = cms.Sequence( goodExoLLElectrons * diExoLLElectrons * diExoLLElectronsFilter )

exoLLResEleMuSequence = cms.Sequence( goodExoLLMuons * goodExoLLElectrons * crossExoLLLeptons * crossExoLLLeptonsFilter )
