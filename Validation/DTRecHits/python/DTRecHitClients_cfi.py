import FWCore.ParameterSet.Config as cms

dtrechitclients = cms.EDAnalyzer("DTRecHitClients")
dt2dsegmentclients = cms.EDAnalyzer("DT2DSegmentClients")
dt4dsegmentclients = cms.EDAnalyzer("DT4DSegmentClients")
                               
##dtLocalRecoValidationClients = cms.Sequence(dt2dsegmentclients*dt4dsegmentclients)
dtLocalRecoValidationClients = cms.Sequence(dtrechitclients*dt2dsegmentclients*dt4dsegmentclients)


