import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUESettings_cfi import *

pyquenPythiaDefaultBlock = cms.PSet(
    pythiaUESettingsBlock,
    hydjetPythiaDefault = cms.vstring(
        'MSEL=0        ! user processes',
        'CKIN(3)=6.    ! ptMin',
        'MSTP(81)=0    ! multiple interaction OFF'
        ),
    pythiaJets = cms.vstring(
        'MSUB(11)=1', # q+q->q+q
        'MSUB(12)=1', # q+qbar->q+qbar
        'MSUB(13)=1', # q+qbar->g+g
        'MSUB(28)=1', # q+g->q+g
        'MSUB(53)=1', # g+g->q+qbar
        'MSUB(68)=1'  # g+g->g+g
        ),
    pythiaPromptPhotons = cms.vstring(
        'MSUB(14)=1', # q+qbar->g+gamma
        'MSUB(18)=1', # q+qbar->gamma+gamma
        'MSUB(29)=1', # q+g->q+gamma
        'MSUB(114)=1', # g+g->gamma+gamma
        'MSUB(115)=1' # g+g->g+gamma
        ),
    
    pythiaWeakBosons = cms.vstring(
        'MSUB(1)=1',
        'MSUB(2)=1'),
    
    pythiaCharmoniumNRQCD = cms.vstring(
        'MSUB(421) = 1',
        'MSUB(422) = 1',
        'MSUB(423) = 1',
        'MSUB(424) = 1',
        'MSUB(425) = 1',
        'MSUB(426) = 1',
        'MSUB(427) = 1',
        'MSUB(428) = 1',
        'MSUB(429) = 1',
        'MSUB(430) = 1',
        'MSUB(431) = 1',
        'MSUB(432) = 1',
        'MSUB(433) = 1',
        'MSUB(434) = 1',
        'MSUB(435) = 1',
        'MSUB(436) = 1',
        'MSUB(437) = 1',
        'MSUB(438) = 1',
        'MSUB(439) = 1'
        ),

    pythiaBottomoniumNRQCD = cms.vstring(
        'MSUB(461) = 1',
        'MSUB(462) = 1',
        'MSUB(463) = 1',
        'MSUB(464) = 1',
        'MSUB(465) = 1',
        'MSUB(466) = 1',
        'MSUB(467) = 1',
        'MSUB(468) = 1',
        'MSUB(469) = 1',
        'MSUB(470) = 1',
        'MSUB(471) = 1',
        'MSUB(472) = 1',
        'MSUB(473) = 1',
        'MSUB(474) = 1',
        'MSUB(475) = 1',
        'MSUB(476) = 1',
        'MSUB(477) = 1',
        'MSUB(478) = 1',
        'MSUB(479) = 1',
        ),
    
    pythiaQuarkoniaSettings = cms.vstring(
        'PARP(141)=1.16',   # Matrix Elements
        'PARP(142)=0.0119',
        'PARP(143)=0.01',
        'PARP(144)=0.01',
        'PARP(145)=0.05',
        'PARP(146)=9.28',
        'PARP(147)=0.15',
        'PARP(148)=0.02',
        'PARP(149)=0.02',
        'PARP(150)=0.085',                                                
        # Meson spin
        'PARJ(13)=0.60',
        'PARJ(14)=0.162',
        'PARJ(15)=0.018',
        'PARJ(16)=0.054',
        # Polarization
        'MSTP(145)=0',
        'MSTP(146)=0',
        'MSTP(147)=0',
        'MSTP(148)=1',
        'MSTP(149)=1',
        # Chi_c branching ratios
        'BRAT(861)=0.202',
        'BRAT(862)=0.798',
        'BRAT(1501)=0.013',
        'BRAT(1502)=0.987',
        'BRAT(1555)=0.356',
        'BRAT(1556)=0.644'
        )
    )    

generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    crossSection = cms.untracked.double(71260000000.),
    comEnergy = cms.double(7000.0),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        processParameters = cms.vstring('MSEL=0         ! User defined processes', 
                                        'CKIN(3)=6.     ! ptMin'
                                        #,'MSTP(81)=0     ! multiple interaction OFF',
                                        ),
        pythiaJets = cms.vstring(
            'MSUB(11)=1', # q+q->q+q
            'MSUB(12)=1', # q+qbar->q+qbar
            'MSUB(13)=1', # q+qbar->g+g
            'MSUB(28)=1', # q+g->q+g
            'MSUB(53)=1', # g+g->q+qbar
            'MSUB(68)=1'  # g+g->g+g
            ),
        pythiaPromptPhotons = cms.vstring(
            'MSUB(14)=1', # q+qbar->g+gamma
            'MSUB(18)=1', # q+qbar->gamma+gamma
            'MSUB(29)=1', # q+g->q+gamma
            'MSUB(114)=1', # g+g->gamma+gamma
            'MSUB(115)=1' # g+g->g+gamma
            ),

        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
                                    'processParameters',
                                    'pythiaJets',
                                    'pythiaPromptPhotons')
        )
                         )

