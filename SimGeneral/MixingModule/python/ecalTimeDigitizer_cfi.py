import FWCore.ParameterSet.Config as cms

from SimCalorimetry.EcalSimProducers.ecalTimeDigiParameters_cff import *

ecalTimeDigitizer = cms.PSet(
    ecal_time_digi_parameters,
    accumulatorType = cms.string("EcalTimeDigiProducer"),
)
