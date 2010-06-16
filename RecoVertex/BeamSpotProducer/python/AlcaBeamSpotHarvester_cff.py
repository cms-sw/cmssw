import FWCore.ParameterSet.Config as cms

from RecoVertex.BeamSpotProducer.AlcaBeamSpotHarvester_cfi import *
alcaBeamSpotHarvesting = cms.Sequence( alcaBeamSpotHarvester )
