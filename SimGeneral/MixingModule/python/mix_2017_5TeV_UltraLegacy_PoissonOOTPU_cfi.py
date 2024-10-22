import FWCore.ParameterSet.Config as cms
from SimGeneral.MixingModule.mix_probFunction_25ns_PoissonOOTPU_cfi import *
mix.input.nbPileupEvents.probFunctionVariable = cms.vint32(
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28
    )

mix.input.nbPileupEvents.probValue = cms.vdouble(
    0.191452846727, 0.352881149019, 0.353135299622, 0.100289548466, 0.00224108424297,
    7.19230424545e-08, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0
    )

