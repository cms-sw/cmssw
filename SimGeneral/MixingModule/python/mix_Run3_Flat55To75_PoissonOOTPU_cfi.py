# A simple distribution for Run3 studies consisting of a flat distribution from 55
# to 75 for the average pileup.

import FWCore.ParameterSet.Config as cms
from SimGeneral.MixingModule.mix_probFunction_25ns_PoissonOOTPU_cfi import *
mix.input.nbPileupEvents.probFunctionVariable = cms.vint32(
    55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
    75
    )

mix.input.nbPileupEvents.probValue = cms.vdouble(
    0.047619047619, 0.047619047619, 0.047619047619, 0.047619047619, 0.047619047619,
    0.047619047619, 0.047619047619, 0.047619047619, 0.047619047619, 0.047619047619,
    0.047619047619, 0.047619047619, 0.047619047619, 0.047619047619, 0.047619047619,
    0.047619047619, 0.047619047619, 0.047619047619, 0.047619047619, 0.047619047619,
    0.047619047619
    )

