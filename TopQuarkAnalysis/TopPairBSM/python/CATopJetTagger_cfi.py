import FWCore.ParameterSet.Config as cms

# Cambridge-Aachen top jet tagger parameters
# $Id
CATopCaloJetTagInfos = cms.EDProducer("CATopJetTagger",
                                      src = cms.InputTag("caTopCaloJets"),
                                      TopMass = cms.double(171),
                                      TopMassMin = cms.double(0.),
                                      TopMassMax = cms.double(250.),
                                      WMass = cms.double(80.4),
                                      WMassMin = cms.double(0.0),
                                      WMassMax = cms.double(200.0),
                                      MinMassMin = cms.double(0.0),
                                      MinMassMax = cms.double(200.0),
                                      verbose = cms.bool(False)
                                      )


CATopPFJetTagInfos = cms.EDProducer("CATopJetTagger",
                                    src = cms.InputTag("caTopPFJets"),
                                    TopMass = cms.double(171),
                                    TopMassMin = cms.double(0.),
                                    TopMassMax = cms.double(250.),
                                    WMass = cms.double(80.4),
                                    WMassMin = cms.double(0.0),
                                    WMassMax = cms.double(200.0),
                                    MinMassMin = cms.double(0.0),
                                    MinMassMax = cms.double(200.0),
                                    verbose = cms.bool(False)
                                    )
