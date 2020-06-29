import FWCore.ParameterSet.Config as cms

# beam optics
from CalibPPS.ESProducers.ctppsBeamParametersFromLHCInfoESSource_cfi import *
from CalibPPS.ESProducers.ctppsInterpolatedOpticalFunctionsESSource_cfi import *
ctppsInterpolatedOpticalFunctionsESSource.lhcInfoLabel = ""


# For optical functions from root file, when they are not yet in the database, use the definitino below
"""
from SimTransport.PPSProtonTransport.OpticalFunctionsConfig_cfi import *

_opticsConfig = cms.PSet(
                    defaultCrossingAngle=cms.double(0.0),
                    es_source = cms.PSet()
                )

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
