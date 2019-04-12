import FWCore.ParameterSet.Config as cms

# PPS Digitization
from SimPPS.PPSPixelDigiProducer.RPixDetDigitizer_cfi import *
from SimPPS.RPDigiProducer.RPSiDetDigitizer_cfi import *

ppsDigi = cms.Sequence()

# add PPS 2016 digi modules
from Configuration.Eras.Modifier_pps_2016_cff import pps_2016 
pps_2016.toReplaceWith(ppsDigi, cms.Sequence(RPSiDetDigitizer))

# add PPS 2017 digi modules
from Configuration.Eras.Modifier_pps_2017_cff import pps_2017
pps_2017.toReplaceWith(ppsDigi, cms.Sequence(RPixDetDigitizer+RPSiDetDigitizer))

# add PPS 2018 digi modules
from Configuration.Eras.Modifier_pps_2018_cff import pps_2018
pps_2018.toReplaceWith(ppsDigi, cms.Sequence(RPixDetDigitizer))

