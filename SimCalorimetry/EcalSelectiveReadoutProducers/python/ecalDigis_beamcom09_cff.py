import FWCore.ParameterSet.Config as cms

# Define EcalSelectiveReadoutProducer module as "simEcalDigis" with default settings 
from SimCalorimetry.EcalSelectiveReadoutProducers.ecalDigis_cfi import *

# Changes settings to 2009 and 2010 beam ones:
#
# DCC ZS FIR weights.
simEcalDigis.dccNormalizedWeights = cms.vdouble(-1.1865, 0.0195, 0.2900, 0.3477, 0.3008, 0.2266)

# Index of time sample (staring from 1) the first DCC weights is implied
simEcalDigis.ecalDccZs1stSample = cms.int32(3)

# ZS energy threshold in GeV to apply to low interest channels of barrel
simEcalDigis.ebDccAdcToGeV = cms.double(0.035)
simEcalDigis.srpBarrelLowInterestChannelZS = cms.double(2.25*0.035)

# ZS energy threshold in GeV to apply to low interest channels of endcap
simEcalDigis.eeDccAdcToGeV = cms.double(0.06)
simEcalDigis.srpEndcapLowInterestChannelZS = cms.double(3.75*0.06)

