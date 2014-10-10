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
            'MSEL=0              !User defined processes', 
            'MSUB(1)=1           !Z0/gamma* production, ISUB=1',
            'MSUB(15)=1          !Z0/gamma*+jet production, ISUB=15', 
            'MSUB(30)=1          !Z0/gamma*+jet production, ISUB=15,30', 
            'MDME(174,1)=0       !Z decay into d dbar',
            'MDME(175,1)=0       !Z decay into u ubar',
            'MDME(176,1)=0       !Z decay into s sbar',
            'MDME(177,1)=0       !Z decay into c cbar',
            'MDME(178,1)=0       !Z decay into b bbar',
            'MDME(179,1)=0       !Z decay into t tbar',
            'MDME(182,1)=0       !Z decay into e- e+',
            'MDME(183,1)=0       !Z decay into nu_e nu_ebar',
            'MDME(184,1)=1       !Z decay into mu- mu+',
            'MDME(185,1)=0       !Z decay into nu_mu nu_mubar',
            'MDME(186,1)=0       !Z decay into tau- tau+',
            'MDME(187,1)=0       !Z decay into nu_tau nu_taubar',
            'CKIN(3) = 10.       !(D=0 GeV) lower lim pT_hat',
            'CKIN(4) = 40.       !(D=-1 GeV) upper lim pT_hat, if < 0 innactive',
            'CKIN(7) = -2.4      !(D=-10) lower limit rapidity',
            'CKIN(8) = 2.4       !(D=10) upper limit rapidity'		
            ),
        parameterSets = cms.vstring('pythiaUESettings', 
                                    'processParameters')
        )
                         )

