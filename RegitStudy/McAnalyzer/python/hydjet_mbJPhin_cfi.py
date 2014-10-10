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
                         doCollisionalEnLoss       = cms.bool(False),
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
                            nMultiplicity           = cms.int32(21500),
                            fracSoftMultiplicity    = cms.double(1.),
                            maxLongitudinalRapidity = cms.double(4.5),
                            maxTransverseRapidity   = cms.double(1.),
                            rotateEventPlane        = cms.bool(True),
                            allowEmptyEvents        = cms.bool(False),
                            embeddingMode           = cms.bool(False)                            
                            )

pyquenPythiaDefaultBlock = cms.PSet(
    pythiaUESettingsBlock,
    hydjetPythiaDefault = cms.vstring('MSEL=0        ! user processes',
                                      'CKIN(3)=6.    ! ptMin',
                                      'MSTP(81)=0    ! multiple interaction OFF'
                                      ),
    pythiaJets = cms.vstring('MSUB(11)=1', # q+q->q+q
                             'MSUB(12)=1', # q+qbar->q+qbar
                             'MSUB(13)=1', # q+qbar->g+g
                             'MSUB(28)=1', # q+g->q+g
                             'MSUB(53)=1', # g+g->q+qbar
                             'MSUB(68)=1'  # g+g->g+g
                             ),
    pythiaPromptPhotons = cms.vstring('MSUB(14)=1', # q+qbar->g+gamma
                                      'MSUB(18)=1', # q+qbar->gamma+gamma
                                      'MSUB(29)=1', # q+g->q+gamma
                                      'MSUB(114)=1', # g+g->gamma+gamma
                                      'MSUB(115)=1' # g+g->g+gamma
                                      )
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
                                                                                 'pythiaJets',
                                                                                 'pythiaPromptPhotons'
                                                                                 )
                                                     )
                         )
