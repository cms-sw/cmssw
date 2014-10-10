import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUESettings_cfi import *

collisionParameters7GeV = cms.PSet(
    aBeamTarget = cms.double(208.0), # beam/target atomic number
    comEnergy = cms.double(7000.0)   # collision en
    )

collisionParameters2760GeV = cms.PSet(
    aBeamTarget = cms.double(208.0), # beam/target atomic number
    comEnergy = cms.double(2760.0)   # collision en
    )

collisionParameters = collisionParameters2760GeV.clone()

qgpParameters = cms.PSet(qgpInitialTemperature     = cms.double(1.0), ## initial temperature of QGP; allowed range [0.2,2.0]GeV;
                         qgpProperTimeFormation    = cms.double(0.1), ## proper time of QGP formation; allowed range [0.01,10.0]fm/c;
                         hadronFreezoutTemperature = cms.double(0.14),
                         doRadiativeEnLoss         = cms.bool(True),  ## if true, perform partonic radiative en loss
                         doCollisionalEnLoss       = cms.bool(True),  ## 2.76--- False
                         qgpNumQuarkFlavor         = cms.int32(0),    ## num. active quark flavors in qgp; allowed values: 0,1,2,3 
                         numQuarkFlavor            = cms.int32(0)     ## to be removed
                         )

pyquenParameters  = cms.PSet(doIsospin = cms.bool(True),
                             angularSpectrumSelector = cms.int32(0),             ## angular emitted gluon spectrum :
                             embeddingMode           = cms.bool(False),
                             backgroundLabel         = cms.InputTag("generator") ## ineffective in no mixing
                             )

hydjetParameters = cms.PSet(sigmaInelNN             = cms.double(58),
                            shadowingSwitch         = cms.int32(0),
                            nMultiplicity           = cms.int32(26000), # 2.76-- 21500
                            fracSoftMultiplicity    = cms.double(1.),
                            maxLongitudinalRapidity = cms.double(3.75), #2.76---4.5
                            maxTransverseRapidity   = cms.double(1.),
                            rotateEventPlane        = cms.bool(True),
                            allowEmptyEvents        = cms.bool(False),
                            embeddingMode           = cms.bool(False)
                            )

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
        ),
    pythiaOfficialSettings = cms.vstring('MSEL=0         ! User defined processes', 
                                         'MSUB(11)=1     ! Min bias process', 
                                         'MSUB(12)=1     ! Min bias process', 
                                         'MSUB(13)=1     ! Min bias process', 
                                         'MSUB(28)=1     ! Min bias process', 
                                         'MSUB(53)=1     ! Min bias process', 
                                         'MSUB(68)=1     ! Min bias process', 
                                         'MSUB(92)=1     ! Min bias process, single diffractive', 
                                         'MSUB(93)=1     ! Min bias process, single diffractive', 
                                         'MSUB(94)=1     ! Min bias process, double diffractive', 
                                         'MSUB(95)=1     ! Min bias process')
    )    

# This one is not to be used
impactParameters = cms.PSet(cFlag = cms.int32(1),
                            bFixed = cms.double(0),
                            bMin = cms.double(0),
                            bMax = cms.double(30)
                            )


generator = cms.EDFilter("HydjetGeneratorFilter",
                         collisionParameters,
                         qgpParameters,
                         hydjetParameters,
                         impactParameters,
                         hydjetMode = cms.string('kHydroQJets'),
                         PythiaParameters = cms.PSet(pyquenPythiaDefaultBlock,
                                                     # Quarkonia and Weak Bosons added back upon dilepton group's request.
                                                     parameterSets = cms.vstring('pythiaUESettings',
                                                                                 'hydjetPythiaDefault',
                                                                                 'pythiaOfficialSettings'
                                                                                 )

                                                     )
                         )
  # Valid entries:
                            # kHydroOnly  //jet production off (pure HYDRO event): nhsel=0
                            # kHydroJets  //jet production on, jet quenching off (HYDRO+njet*PYTHIA events):nhsel=1
                            # kHydroQJets //jet production & jet quenching on (HYDRO+njet*PYQUEN events);nhsel=2
                            # kJetsOnly   //jet production on, jet quenching off, HYDRO off (njet*PYTHIA events):nhsel=3
                            # kQJetsOnly  //jet production & jet quenching on, HYDRO off (njet*PYQUEN events):nhsel=4  
