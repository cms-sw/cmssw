import FWCore.ParameterSet.Config as cms

dtrechitclients = cms.EDProducer("DTRecHitClients",
    doStep2 = cms.untracked.bool(False),
    # Switches for analysis at various steps
    doStep1 = cms.untracked.bool(False),
    # Lable to retrieve RecHits from the event
    doStep3 = cms.untracked.bool(True),
    doall   = cms.untracked.bool(False),
    local   = cms.untracked.bool(False)
)

dt2dsegmentclients = cms.EDProducer("DT2DSegmentClients",
    do2D    = cms.untracked.bool(False),
    doSLPhi = cms.untracked.bool(False)
)
dt4dsegmentclients = cms.EDProducer("DT4DSegmentClients",
    doall = cms.untracked.bool(False)
)
                               
##dtLocalRecoValidationClients = cms.Sequence(dt2dsegmentclients*dt4dsegmentclients)
dtLocalRecoValidationClients = cms.Sequence(dtrechitclients*dt2dsegmentclients*dt4dsegmentclients)


