import FWCore.ParameterSet.Config as cms

#
# module to make mvaTraining for jet parton associations
#
buildTraintree = cms.EDAnalyzer("TtSemiLepSignalSelMVATrainer",
    #input tags used in the example
    muons  = cms.InputTag("selectedPatMuons"),
    elecs  = cms.InputTag("selectedPatElectrons"),                         
    jets   = cms.InputTag("selectedPatJets"),
    mets   = cms.InputTag("patMETs"),

    # ------------------------------------------------
    # select semileptonic signal channel
    # (all others are taken as background for training)
    # 1: electron, 2: muon, 3: tau
    # ------------------------------------------------
    lepChannel = cms.int32(2),

    #three possibilities:
    # whatData=-1: in your training-file both, signal and background events are available
    # whatData=1: your training-file contains only signal events
    # whatData=0: your training-file contains only background events
    whatData = cms.int32(1),

    #maximum number of training events to be used
    # maxEv = -1: all events are used
    # for example maxEv = 5000: writes only the first 5000 events to the training tree
    maxEv = cms.int32(-1),
)
