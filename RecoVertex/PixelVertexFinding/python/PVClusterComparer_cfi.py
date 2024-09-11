import FWCore.ParameterSet.Config as cms

pvClusterComparer = cms.PSet(
  track_pt_min   = cms.double(     1.0),
  track_pt_max   = cms.double(    10.0), # SD: 20.
  track_chi2_max = cms.double(999999. ), # SD: 20
  track_prob_min = cms.double(    -1. ), # RM: 0.001
)
