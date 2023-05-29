# A simple distribution for Run3 studies consisting of a flat distribution from 55
# to 75 for the average pileup.

import FWCore.ParameterSet.Config as cms
from SimGeneral.MixingModule.mix_probFunction_25ns_PoissonOOTPU_cfi import *
mix.input.nbPileupEvents.probFunctionVariable = cms.vint32(range(201))

mix.input.nbPileupEvents.probValue = cms.vdouble([0.00497512 for x in range(201)])
