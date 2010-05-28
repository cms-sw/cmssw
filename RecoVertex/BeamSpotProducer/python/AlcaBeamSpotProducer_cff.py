import FWCore.ParameterSet.Config as cms

from RecoVertex.BeamSpotProducer.AlcaBeamSpotProducer_cfi import *
alcaBeamSpot = cms.Sequence( alcaBeamSpotProducer )
