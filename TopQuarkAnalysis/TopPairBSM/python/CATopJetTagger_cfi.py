import FWCore.ParameterSet.Config as cms

# Cambridge-Aachen top jet tagger parameters
# $Id
CATopJetTagger = cms.EDProducer("CATopJetTagger",
                                src = cms.InputTag("caTopJetsProducer"),
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
