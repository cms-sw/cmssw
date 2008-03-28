import FWCore.ParameterSet.Config as cms

# Calo geometry service model
from Geometry.CaloEventSetup.CaloGeometry_cff import *
from Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi import *
# removed by tommaso
# use trivial ESProducer for tests
# include "CalibCalorimetry/EcalTrivialCondModules/data/EcalTrivialCondRetriever.cfi"
# unsuppressed digis simulation - fast preshower
from SimCalorimetry.EcalSimProducers.ecaldigi_cfi import *
# ecal mapping
from Geometry.EcalMapping.EcalMapping_cfi import *
# Trigger Primitives Generation producer
# not used in 0_8_0
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff import *
# Selective Readout Processor producer
from SimCalorimetry.EcalSelectiveReadoutProducers.ecalDigis_cfi import *
# Preshower Zero suppression producer
from SimCalorimetry.EcalZeroSuppressionProducers.ecalPreshowerDigis_cfi import *
ecalDigiSequence = cms.Sequence(ecalUnsuppressedDigis*ecalTriggerPrimitiveDigis*ecalDigis*ecalPreshowerDigis)
ecalUnsuppressedDigis.doFast = True
ecalPreshowerDigis.ESNoiseSigma = 2.98576

