import FWCore.ParameterSet.Config as cms

from RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi import *

#scalers = cms.EDProducer('ScalersRawToDigi')
BeamSpotESProducer = cms.ESProducer("OnlineBeamSpotESProducer")

onlineBeamSpot = cms.Sequence( onlineBeamSpotProducer )

