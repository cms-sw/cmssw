import FWCore.ParameterSet.Config as cms

# Cambridge-Aachen top jet tagger parameters
# $Id
CATopCaloJetTagInfos = cms.EDProducer("CATopJetTagger",
                                      src = cms.InputTag("caTopCaloJets"),
                                      TopMass = cms.double(171),
                                      WMass = cms.double(80.4),
                                      verbose = cms.bool(False)
                                      )


CATopPFJetTagInfos = cms.EDProducer("CATopJetTagger",
                                    src = cms.InputTag("caTopPFJets"),
                                    TopMass = cms.double(171),
                                    WMass = cms.double(80.4),
                                    verbose = cms.bool(False)
                                    )
