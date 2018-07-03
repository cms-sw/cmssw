import FWCore.ParameterSet.Config as cms

# Trigger Primitive Producer
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_mc_cfi import *
# esmodule creating  records + corresponding empty essource
from SimCalorimetry.EcalTrigPrimProducers.ecalTrigPrimESProducer_mc_cff import *

#Common
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify( simEcalTriggerPrimitiveDigis, BarrelOnly = cms.bool(True) )
