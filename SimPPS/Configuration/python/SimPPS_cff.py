import FWCore.ParameterSet.Config as cms

# PPS Digitization
#from SimPPS.PPSPixelDigiProducer.RPixDetConf_cfi import *
from SimPPS.PPSPixelDigiProducer.RPixDetDigitizer_cfi import *
from SimPPS.RPDigiProducer.RPSiDetConf_cfi import *

ppsDigi = cms.Sequence(RPixDetDigitizer+RPSiDetDigitizer)
