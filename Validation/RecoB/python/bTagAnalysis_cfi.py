import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from DQMOffline.RecoB.bTagCommon_cff import *

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
bTagValidation = DQMEDAnalyzer('BTagPerformanceAnalyzerMC',
                                bTagCommonBlock,
                                applyPtHatWeight = cms.bool(False),
                                jetCorrection = cms.string(''),
                                recJetMatching = cms.PSet(refJetCorrection = cms.string(''),
                                                          recJetCorrection = cms.string(''),
                                                          maxChi2 = cms.double(50),
                                                          # Corrected calo jets
                                                          sigmaDeltaR = cms.double(0.1),
                                                          sigmaDeltaE = cms.double(0.15)
                                                          ),
                                flavPlots = cms.string("allbcl"),                            
                                leptonPlots = cms.uint32(0),
                                genJetsMatched = cms.InputTag("patJetGenJetMatch"),                            
                                doPUid = cms.bool(True)
                                )


bTagHarvestMC = DQMEDHarvester("BTagPerformanceHarvester",
                               bTagCommonBlock,
                               produceEps = cms.bool(False),
                               producePs = cms.bool(False),
                               flavPlots = cms.string("allbcl"),
                               differentialPlots = cms.bool(False), #not needed in validation procedure, put True to produce them                            
                               )
