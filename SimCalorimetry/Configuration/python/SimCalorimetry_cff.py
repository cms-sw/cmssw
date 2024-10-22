import FWCore.ParameterSet.Config as cms

from SimCalorimetry.Configuration.ecalDigiSequence_cff import *
from SimCalorimetry.Configuration.hcalDigiSequence_cff import *
from SimCalorimetry.Configuration.castorDigiSequence_cff import *

calDigiTask = cms.Task(ecalDigiTask, hcalDigiTask, castorDigiTask)
calDigi = cms.Sequence(calDigiTask)

# fastsim has no castor model
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(calDigiTask, calDigiTask.copyAndExclude([castorDigiTask]))
