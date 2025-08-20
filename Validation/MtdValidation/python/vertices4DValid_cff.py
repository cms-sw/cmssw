import FWCore.ParameterSet.Config as cms

from Validation.MtdValidation.vertices4DValid_cfi import vertices4DValid
from RecoVertex.Configuration.RecoVertex_cff import unsortedOfflinePrimaryVertices4D

# higher eta cut for the phase 2 tracker
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(vertices4DValid, TkFilterParameters = cms.PSet(unsortedOfflinePrimaryVertices4D.TkFilterParameters.clone()))
