import FWCore.ParameterSet.Config as cms

# beam optics
from CondCore.CondDB.CondDB_cfi import *
from CalibPPS.ESProducers.ctppsBeamParametersFromLHCInfoESSource_cfi import *
from CalibPPS.ESProducers.ctppsInterpolatedOpticalFunctionsESSource_cff import *
ctppsInterpolatedOpticalFunctionsESSource.lhcInfoLabel = ""

"""
# For optical functions and LHCinfo not yet in a GT, use the definitino below
ppsTransportESSource = cms.ESSource("PoolDBESSource")

_ppsESSource2021 = cms.ESSource("PoolDBESSource",
     connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
     timetype = cms.untracked.string('runnumber'),
     toGet = cms.VPSet(
                 cms.PSet(
                     record = cms.string('LHCInfoRcd'),
                     tag = cms.string("LHCInfo_2021_mc_v1")
                 ),
                 cms.PSet(
                     record = cms.string('CTPPSOpticsRcd'),
                     tag = cms.string("PPSOpticalFunctions_2021_mc_v1")
                 )
     )
)

from Configuration.Eras.Modifier_ctpps_2021_cff import ctpps_2021
ctpps_2021.toReplaceWith(ppsTransportESSource,_ppsESSource2021)
del _ppsESSource2021

from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
ctpps_2016.toReplaceWith(_opticsConfig, opticalfunctionsTransportSetup_2016.opticalFunctionConfig)

from Configuration.Eras.Modifier_ctpps_2017_cff import ctpps_2017
ctpps_2017.toReplaceWith(_opticsConfig, opticalfunctionsTransportSetup_2017.opticalFunctionConfig)

from Configuration.Eras.Modifier_ctpps_2018_cff import ctpps_2018
ctpps_2018.toReplaceWith(_opticsConfig, opticalfunctionsTransportSetup_2018.opticalFunctionConfig)

from Configuration.Eras.Modifier_ctpps_2021_cff import ctpps_2021
ctpps_2021.toReplaceWith(_opticsConfig, opticalfunctionsTransportSetup_2021.opticalFunctionConfig)

ctppsBeamParametersESSource.halfXangleX45 = _opticsConfig.defaultCrossingAngle
ctppsBeamParametersESSource.halfXangleX56 = _opticsConfig.defaultCrossingAngle
ctppsOpticalFunctionsESSource.configuration.append(_opticsConfig.es_source)
# clean up to avoid spreading uneeded modules up in the configuration chain
del _opticsConfig
del opticalfunctionsTransportSetup_2016
del opticalfunctionsTransportSetup_2018
del opticalfunctionsTransportSetup_2017
del opticalfunctionsTransportSetup_2021
"""
