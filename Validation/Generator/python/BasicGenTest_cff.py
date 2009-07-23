import FWCore.ParameterSet.Config as cms

from DQMServices.Components.DQMEnvironment_cfi import *

DQMStore = cms.Service("DQMStore")

dqmSaver.convention = 'Offline'
dqmSaver.workflow = '/BasicGenTest/Workflow/GEN'
