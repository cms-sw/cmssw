import FWCore.ParameterSet.Config as cms

from GeneratorInterface.ExternalDecays.TauolaSettings_cff import *
from Configuration.Generator.Pythia8CUEP8S1CTEQ6L1Settings_cfi import *

generator = cms.EDFilter("Pythia8HadronizerFilter",
                         ExternalDecays = cms.PSet(
        Tauola = cms.untracked.PSet(
            TauolaPolar,
            TauolaDefaultInputCards
            ),
        parameterSets = cms.vstring('Tauola')
        ),
                         UseExternalGenerators = cms.untracked.bool(True),                        
                         maxEventsToPrint = cms.untracked.int32(1),
                         pythiaPylistVerbosity = cms.untracked.int32(1),
                         filterEfficiency = cms.untracked.double(1.0),
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         comEnergy = cms.double(8000.),
                         PythiaParameters = cms.PSet(
        pythia8CUEP8S1cteqSettingsBlock,
        processParameters = cms.vstring(
            'Main:timesAllowErrors = 10000',
            'ParticleDecays:tauMax = 10',
            ),
        parameterSets = cms.vstring('pythia8CUEP8S1cteqSettings',
                                    'processParameters')
        )
 )
