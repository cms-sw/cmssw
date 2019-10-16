import FWCore.ParameterSet.Config as cms

# PPS Digitization
from SimPPS.PPSPixelDigiProducer.RPixDetDigitizer_cfi import *
from SimPPS.RPDigiProducer.RPSiDetDigitizer_cfi import *
RPixDetDigitizerTask=cms.Task(RPixDetDigitizer)
RPSiDetDigitizerTask=cms.Task(RPSiDetDigitizer)

ctppsDigiTask = cms.Task()
ctppsDigi = cms.Sequence(ctppsDigiTask)

# add PPS 2016 digi modules
from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
ctpps_2016.toReplaceWith( ctppsDigiTask, RPSiDetDigitizerTask)

# add PPS 2017 digi modules
from Configuration.Eras.Modifier_ctpps_2017_cff import ctpps_2017
ctpps_2017Task = cms.Task(RPixDetDigitizer,RPSiDetDigitizer)
ctpps_2017.toReplaceWith(ctppsDigiTask, ctpps_2017Task)

# add PPS 2018 digi modules
from Configuration.Eras.Modifier_ctpps_2018_cff import ctpps_2018
ctpps_2018.toReplaceWith(ctppsDigiTask, RPixDetDigitizerTask)

