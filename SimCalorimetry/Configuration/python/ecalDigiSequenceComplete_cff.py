import FWCore.ParameterSet.Config as cms

from Geometry.CaloEventSetup.CaloGeometry_cff import *
from Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi import *
from SimCalorimetry.EcalSimProducers.ecaldigi_cfi import *
from Geometry.EcalMapping.EcalMapping_cfi import *
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff import *
from SimCalorimetry.EcalSelectiveReadoutProducers.ecalDigis_cfi import *
from SimCalorimetry.EcalZeroSuppressionProducers.ecalPreshowerDigis_cfi import *
ecalDigiSequenceComplete = cms.Sequence(simEcalUnsuppressedDigis*simEcalTriggerPrimitiveDigis*simEcalDigis*simEcalPreshowerDigis)

