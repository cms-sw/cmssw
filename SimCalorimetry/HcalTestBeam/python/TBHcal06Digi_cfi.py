import FWCore.ParameterSet.Config as cms

# Geometry of calorimeter
from SimCalorimetry.HcalTestBeam.TBHcal06Geometry_cfi import *
# use trivial ESProducer for tests
#
from CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi import *
from SimCalorimetry.HcalTestBeam.hardwired_conditions_cfi import *
# unsuppressed digi's
from SimCalorimetry.HcalTestBeam.ecaldigi_testbeam_cfi import *
from SimCalorimetry.HcalTestBeam.TBHcal06HcalDigi_cfi import *
# MixingModule is required for digitization,
# at least, in the zero-pileup mode (as below)
#
mix = cms.EDFilter("MixingModule",
    bunchspace = cms.int32(25)
)

doAllDigi = cms.Sequence(mix*ecalUnsuppressedDigis*hcalDigis)

