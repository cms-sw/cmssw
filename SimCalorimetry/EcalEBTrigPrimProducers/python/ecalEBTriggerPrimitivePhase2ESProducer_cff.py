import os
import FWCore.ParameterSet.Config as cms

# esmodule creating  records + corresponding empty essource
EcalEBTrigPrimPhase2ESProducer = cms.ESProducer("EcalEBTrigPrimPhase2ESProducer",
    DatabaseFile = cms.untracked.string('TPG_beamv5_MC_startup.txt.gz'),
#    WeightTextFile = cms.untracked.string(os.environ['CMSSW_BASE'] + '/src/SimCalorimetry/EcalEBTrigPrimProducers/data/AmpTimeOnPeakXtalWeights.txt.gz'),
#    WeightTextFile = cms.untracked.string(os.environ['CMSSW_BASE'] + '/src/SimCalorimetry/EcalEBTrigPrimProducers/data/AmpTimeOnPeakXtalWeights_6samples_peakOnSix.txt.gz'),
#    WeightTextFile = cms.untracked.string(os.environ['CMSSW_BASE'] + '/src/SimCalorimetry/EcalEBTrigPrimProducers/data/AmpTimeOnPeakXtalWeightsCMSSWPulse_8samples_peakOnSix.txt.gz'),
#     WeightTextFile = cms.untracked.string(os.environ['CMSSW_BASE'] + '/src/SimCalorimetry/EcalEBTrigPrimProducers/data/AmpTimeOnPeakXtalWeightsCMSSWPulse_8samples_peakOnSix_WithFixLlinear.txt.gz')
#   WeightTextFile = cms.untracked.string(os.environ['CMSSW_BASE'] + '/src/SimCalorimetry/EcalEBTrigPrimProducers/data/AmpTimeOnPeakXtalWeightsCMSSWPulse_8samples_peakOnSix_WithFixLlinear.txt.gz'),
WeightTextFile = cms.untracked.string(os.environ['CMSSW_BASE'] + '/src/SimCalorimetry/EcalEBTrigPrimProducers/data/AmpTimeOnPeakXtalWeightsCMSSWPulse_8samples_peakOnSix_WithAndyFixes.txt.gz'),
   WriteInFile = cms.bool(False)
)

tpparams = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalEBPhase2TPGLinearizationConstRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

tpparams2 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalEBPhase2TPGPedestalsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


tpparams4 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalEBPhase2TPGAmplWeightIdMapRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

tpparams17 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalEBPhase2TPGTimeWeightIdMapRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


tpparams5 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGWeightGroupRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


tpparams12 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGPhysicsConstRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

tpparams13 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGCrystalStatusRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

tpparams14 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGTowerStatusRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

tpparams15 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGSpikeRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

tpparams16 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGStripStatusRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

