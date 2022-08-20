import FWCore.ParameterSet.Config as cms
from SimTransport.PPSProtonTransport.PPSTransport_cff import LHCTransport


PPSTransportTask = cms.Task()

# The 2016-2018 commented line below need to be activated to integrate the simulation into Run2

#from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
#ctpps_2016.toReplaceWith(PPSTransportTask, cms.Task(LHCTransport))

#from Configuration.Eras.Modifier_ctpps_2017_cff import ctpps_2017
#ctpps_2017.toReplaceWith(PPSTransportTask, cms.Task(LHCTransport))

#from Configuration.Eras.Modifier_ctpps_2018_cff import ctpps_2018
#ctpps_2018.toReplaceWith(PPSTransportTask, cms.Task(LHCTransport))

from Configuration.Eras.Modifier_ctpps_2022_cff import ctpps_2022
ctpps_2022.toReplaceWith(PPSTransportTask, cms.Task(LHCTransport))
