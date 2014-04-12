import FWCore.ParameterSet.Config as cms

from SimG4Core.Application.hectorParameter_cfi import *
from SimTransport.HectorProducer.HectorTransport_cfi import *

LHCTransport.FP420Transport = cms.bool(False) ## main flag to set transport for FP420
LHCTransport.Hector.smearEnergy = cms.bool(False)
LHCTransport.Hector.smearAng    = cms.bool(False)
