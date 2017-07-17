import FWCore.ParameterSet.Config as cms

# configuration to model pileup for initial physics phase
# Now, import base configuration instead of specfiying all parameters
from SimGeneral.MixingModule.mix_probFunction_25ns_PoissonOOTPU_cfi import *
mix.input.nbPileupEvents.probFunctionVariable = cms.vint32(0,1,2,3,4,5,6,7,8,9,10,11,12),
mix.input.nbPileupEvents.probValue = cms.vdouble(
                   0.862811884308912,
                   0.122649088500652,
                   0.013156023430157,
                   0.001271967352346,
                   0.000100080253829,
                   8.21711557752311E-06,
                   1.47486689852979E-06,
                   2.10695271218541E-07,
                   2.10695271218541E-07,
                   2.10695271218541E-07,
                   2.10695271218541E-07,
                   2.10695271218541E-07,
                   2.10695271218541E-07)
