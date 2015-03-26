import FWCore.ParameterSet.Config as cms

from FWCore.MessageService.MessageLogger_cfi import *
options = cms.untracked.PSet(
  wantSummary      = cms.untracked.bool( False )
, allowUnscheduled = cms.untracked.bool( True )
)

from Configuration.Geometry.GeometryIdeal_cff import *
from Configuration.StandardSequences.MagneticField_cff import *
from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *
