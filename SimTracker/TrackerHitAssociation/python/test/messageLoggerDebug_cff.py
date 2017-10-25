import FWCore.ParameterSet.Config as cms

from FWCore.MessageService.MessageLogger_cfi import *
# MessageLogger.destinations = cms.untracked.vstring('cout')
MessageLogger.debugModules.append('testassociator')
MessageLogger.categories.append('TrkHitAssocTrace')
MessageLogger.categories.append('TrkHitAssocDbg')
MessageLogger.cout = cms.untracked.PSet(
  threshold = cms.untracked.string('DEBUG'),
  default = cms.untracked.PSet( limit = cms.untracked.int32(0) )
    , TrkHitAssocTrace = cms.untracked.PSet(limit = cms.untracked.int32(-1))
    , TrkHitAssocDbg = cms.untracked.PSet(limit = cms.untracked.int32(-1))
)
