import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.Configuration.ecalLocalRecoSequence_cff import *

localreco = cms.Sequence(ecalLocalRecoSequence)
