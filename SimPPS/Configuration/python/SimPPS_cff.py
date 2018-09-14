import FWCore.ParameterSet.Config as cms

# PPS Digitization
#from SimPPS.PPSPixelDigiProducer.RPixDetConf_cfi import *
from SimPPS.PPSPixelDigiProducer.RPixDetDigitizer_cfi import *
from SimPPS.RPDigiProducer.RPSiDetConf_cfi import *

<<<<<<< HEAD
from IOMC.RandomEngine.IOMC_cff import *
RandomNumberGeneratorService.RPixDetDigitizer = cms.PSet(initialSeed =cms.untracked.uint32(137137))
RandomNumberGeneratorService.RPSiDetDigitizer = cms.PSet(initialSeed =cms.untracked.uint32(137137))

=======
>>>>>>> PPS Full Simulation branch to PR
ppsDigi = cms.Sequence(RPixDetDigitizer+RPSiDetDigitizer)
