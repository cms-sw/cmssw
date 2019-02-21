import FWCore.ParameterSet.Config as cms

# PPS Digitization
#from SimPPS.PPSPixelDigiProducer.RPixDetConf_cfi import *
from SimPPS.PPSPixelDigiProducer.RPixDetDigitizer_cfi import *
from SimPPS.RPDigiProducer.RPSiDetConf_cfi import *

from IOMC.RandomEngine.IOMC_cff import *
RandomNumberGeneratorService.RPixDetDigitizer = cms.PSet(initialSeed =cms.untracked.uint32(137137))
RandomNumberGeneratorService.RPSiDetDigitizer = cms.PSet(initialSeed =cms.untracked.uint32(137137))

ppsDigi = cms.Sequence()

# add PPS 2016 digi modules
from Configuration.Eras.Modifier_pps_2016_cff import pps_2016
_pps_2016_Digi = ppsDigi.copy()
_pps_2016_Digi = cms.Sequence(RPSiDetDigitizer)
pps_2016.toReplaceWith(ppsDigi,_pps_2016_Digi)

# add PPS 2017 digi modules
from Configuration.Eras.Modifier_pps_2017_cff import pps_2017
_pps_2017_Digi = ppsDigi.copy()
_pps_2017_Digi = cms.Sequence(RPixDetDigitizer+RPSiDetDigitizer)
pps_2017.toReplaceWith(ppsDigi,_pps_2017_Digi)

# add PPS 2018 digi modules
from Configuration.Eras.Modifier_pps_2018_cff import pps_2018
_pps_2018_Digi = ppsDigi.copy()
_pps_2018_Digi = cms.Sequence(RPixDetDigitizer)
pps_2018.toReplaceWith(ppsDigi,_pps_2018_Digi)

#from Configuration.Eras.Modifier_fastSim_cff import fastSim
#fastSim.toReplaceWith(doAllDigi,doAllDigi.copyAndExclude([RPixDetDigitizer,RPSiDetDigitizer]))
