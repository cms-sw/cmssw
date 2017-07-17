import FWCore.ParameterSet.Config as cms

# this is a minimum configuration of the Mixing module,
# to run it in the zero-pileup mode
#
from SimGeneral.MixingModule.mixNoPU_cfi import *

mix.mixObjects.mixHepMC.makeCrossingFrame = True
