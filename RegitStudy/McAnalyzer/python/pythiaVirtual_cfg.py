import FWCore.ParameterSet.Config as cms


from Configuration.Generator.PythiaUESettings_cfi import *
source = cms.Source("EmptySource")

generator = cms.EDFilter("Pythia6GeneratorFilter",
                    maxEventsToPrint = cms.untracked.int32(5),
                    pythiaPylistVerbosity = cms.untracked.int32(1),
                    filterEfficiency = cms.untracked.double(1.0),
                    pythiaHepMCVerbosity = cms.untracked.bool(False),
                    comEnergy = cms.double(5500.0),
                    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        processParameters = cms.vstring(
            'MSEL=11             !Z0/gamma*',
            'MSTP(43)=1          !Only gamma*', 
            'CKIN(7) = -2.4      !(D=-10) lower limit rapidity',
            'CKIN(8) = 2.4       !(D=10) upper limit rapidity'		
            ),
        parameterSets = cms.vstring('pythiaUESettings', 
                                    'processParameters')
        )
)

#ProductionFilterSequence = cms.Sequence(generator)
