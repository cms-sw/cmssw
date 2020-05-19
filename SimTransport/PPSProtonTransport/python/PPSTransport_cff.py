import FWCore.ParameterSet.Config as cms
from SimG4Core.Application.g4SimHits_cfi import *
g4SimHits.Generator.MinEtaCut = cms.double(-13.0)
g4SimHits.Generator.MaxEtaCut = cms.double( 13.0)
g4SimHits.Generator.HepMCProductLabel   = 'LHCTransport'
g4SimHits.SteppingAction.MaxTrackTime = cms.double(2000.0)
g4SimHits.StackingAction.MaxTrackTime = cms.double(2000.0)

from IOMC.RandomEngine.IOMC_cff import *
RandomNumberGeneratorService.LHCTransport.engineName   = cms.untracked.string('TRandom3')

#
# to avoid higher level moodules to import uneeded objects, import module as _module
#
from SimTransport.PPSProtonTransport.CommonParameters_cfi import commonParameters as _commonParameters
from SimTransport.PPSProtonTransport.HectorTransport_cfi import hector_2016 as _hector_2016
from SimTransport.PPSProtonTransport.TotemTransport_cfi import totemTransportSetup_2016  as _totemTransportSetup_2016 
from SimTransport.PPSProtonTransport.OpticalFunctionsConfig_cfi import opticalfunctionsTransportSetup_2016 as _opticalfunctionsTransportSetup_2016
from SimTransport.PPSProtonTransport.OpticalFunctionsConfig_cfi import opticalfunctionsTransportSetup_2017 as _opticalfunctionsTransportSetup_2017
from SimTransport.PPSProtonTransport.OpticalFunctionsConfig_cfi import opticalfunctionsTransportSetup_2018 as _opticalfunctionsTransportSetup_2018
from SimTransport.PPSProtonTransport.OpticalFunctionsConfig_cfi import opticalfunctionsTransportSetup_2021 as _opticalfunctionsTransportSetup_2021

_LHCTransportPSet = cms.PSet()

# so far, it is not yet defined the optic for 2017 and 2018, if needed, change the config for these year to the 2016 one

from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
#ctpps_2016.toReplaceWith(LHCTransportPSet, _totemTransportSetup_2016)
ctpps_2016.toReplaceWith(_LHCTransportPSet, _opticalfunctionsTransportSetup_2016.optics_parameters)

from Configuration.Eras.Modifier_ctpps_2017_cff import ctpps_2017
ctpps_2017.toReplaceWith(_LHCTransportPSet, _opticalfunctionsTransportSetup_2017.optics_parameters)

from Configuration.Eras.Modifier_ctpps_2018_cff import ctpps_2018
ctpps_2018.toReplaceWith(_LHCTransportPSet, _opticalfunctionsTransportSetup_2018.optics_parameters)

from Configuration.Eras.Modifier_ctpps_2021_cff import ctpps_2021
ctpps_2021.toReplaceWith(_LHCTransportPSet, _opticalfunctionsTransportSetup_2021.optics_parameters)

LHCTransport = cms.EDProducer("PPSSimTrackProducer",_commonParameters,_LHCTransportPSet)
