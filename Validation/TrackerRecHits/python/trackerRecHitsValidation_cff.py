import FWCore.ParameterSet.Config as cms

from Validation.TrackerRecHits.SiPixelRecHitsValid_cfi import *
from Validation.TrackerRecHits.SiStripRecHitsValid_cfi import *
import DQM.SiStripMonitorSummary.SiStripMonitorCondData_cfi 

condDataValidation = DQM.SiStripMonitorSummary.SiStripMonitorCondData_cfi.CondDataMonitoring.clone(
    FillConditions_PSet=DQM.SiStripMonitorSummary.SiStripMonitorCondData_cfi.CondDataMonitoring.FillConditions_PSet.clone(
    FolderName_For_QualityAndCabling_SummaryHistos=cms.string("SiStrip/SiStripMonitorSummary"),
    HistoMaps_On    = cms.bool(False),
    ActiveDetIds_On = cms.bool(False)),
    MonitorSiStripPedestal     = cms.bool(True),
    MonitorSiStripNoise        = cms.bool(True),
    MonitorSiStripQuality      = cms.bool(True),
    MonitorSiStripCabling      = cms.bool(True),
    MonitorSiStripLowThreshold = cms.bool(True),
    MonitorSiStripHighThreshold= cms.bool(True),
    MonitorSiStripApvGain      = cms.bool(True),                             
    MonitorSiStripLorentzAngle = cms.bool(True),        
    OutputMEsInRootFile        = cms.bool(False),
    SiStripPedestalsDQM_PSet=DQM.SiStripMonitorSummary.SiStripMonitorCondData_cfi.CondDataMonitoring.SiStripPedestalsDQM_PSet.clone(
    ActiveDetIds_On     = cms.bool(True),
    FillSummaryAtLayerLevel = cms.bool(False)),
    SiStripNoisesDQM_PSet=DQM.SiStripMonitorSummary.SiStripMonitorCondData_cfi.CondDataMonitoring.SiStripNoisesDQM_PSet.clone(
    ActiveDetIds_On        = cms.bool(True),
    FillSummaryAtLayerLevel = cms.bool(False)),
    SiStripQualityDQM_PSet=DQM.SiStripMonitorSummary.SiStripMonitorCondData_cfi.CondDataMonitoring.SiStripQualityDQM_PSet.clone(
    ActiveDetIds_On       = cms.bool(True),
    FillSummaryAtLayerLevel = cms.bool(False)),
    SiStripCablingDQM_PSet=DQM.SiStripMonitorSummary.SiStripMonitorCondData_cfi.CondDataMonitoring.SiStripCablingDQM_PSet.clone(
    ActiveDetIds_On       = cms.bool(True)),
    SiStripLowThresholdDQM_PSet=DQM.SiStripMonitorSummary.SiStripMonitorCondData_cfi.CondDataMonitoring.SiStripLowThresholdDQM_PSet.clone(
    ActiveDetIds_On  = cms.bool(True),
    FillSummaryAtLayerLevel = cms.bool(False)),
    SiStripHighThresholdDQM_PSet=DQM.SiStripMonitorSummary.SiStripMonitorCondData_cfi.CondDataMonitoring.SiStripHighThresholdDQM_PSet.clone(
    ActiveDetIds_On = cms.bool(True),
    FillSummaryAtLayerLevel = cms.bool(False)),
    SiStripApvGainsDQM_PSet=DQM.SiStripMonitorSummary.SiStripMonitorCondData_cfi.CondDataMonitoring.SiStripApvGainsDQM_PSet.clone(
    ActiveDetIds_On      = cms.bool(True),
    FillSummaryAtLayerLevel = cms.bool(False)),
    SiStripLorentzAngleDQM_PSet=DQM.SiStripMonitorSummary.SiStripMonitorCondData_cfi.CondDataMonitoring.SiStripLorentzAngleDQM_PSet.clone(ActiveDetIds_On  = cms.bool(False),
                                                                                           FillSummaryAtLayerLevel = cms.bool(False),
                                                                                           CondObj_fillId = cms.string('ProfileAndCumul') )
    )




trackerRecHitsValidation = cms.Sequence(pixRecHitsValid+stripRecHitsValid+condDataValidation)

