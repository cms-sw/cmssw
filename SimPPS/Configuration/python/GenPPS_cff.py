import FWCore.ParameterSet.Config as cms
from SimTransport.PPSProtonTransport.PPSTransport_cff import LHCTransport


PPSTransportTask = cms.Task()

# so far, it is not yet defined the optic for 2017 and 2018

from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
ctpps_2016.toReplaceWith(PPSTransportTask, cms.Task(LHCTransport))

from Configuration.Eras.Modifier_ctpps_2017_cff import ctpps_2017
ctpps_2017.toReplaceWith(PPSTransportTask, cms.Task(LHCTransport))

from Configuration.Eras.Modifier_ctpps_2018_cff import ctpps_2018
ctpps_2018.toReplaceWith(PPSTransportTask, cms.Task(LHCTransport))

from Configuration.Eras.Modifier_ctpps_2021_cff import ctpps_2021
ctpps_2021.toReplaceWith(PPSTransportTask, cms.Task(LHCTransport))
