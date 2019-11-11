import FWCore.ParameterSet.Config as cms

# This object modifies hcalSimParameters for different scenarios

hcalSimParameters = cms.PSet(
    #  In HF, the SimHits energy is actually
    # the number of photoelectrons from the shower
    # library.  However, we need a lot more
    # smearing, because the first level of the PMT
    # only gives ~6 electrons per pe, which
    # comes out to a 40% smearing of the single pe peak!
    #
    hf1 = cms.PSet(
        readoutFrameSize = cms.int32(4),
        binOfMaximum = cms.int32(3),
        samplingFactor = cms.double(0.383),
        doPhotoStatistics = cms.bool(True),
        photoelectronsToAnalog = cms.double(2.79),
        simHitToPhotoelectrons = cms.double(6.0),
        syncPhase = cms.bool(True),
        timePhase = cms.double(14.0),
        doSiPMSmearing = cms.bool(False),
        sipmTau = cms.double(0.),
        threshold_currentTDC = cms.double(-999.),
    ),
    hf2 = cms.PSet(
        readoutFrameSize = cms.int32(4),
        binOfMaximum = cms.int32(3),
        samplingFactor = cms.double(0.368),
        doPhotoStatistics = cms.bool(True),
        photoelectronsToAnalog = cms.double(1.843),
        simHitToPhotoelectrons = cms.double(6.0),
        syncPhase = cms.bool(True),
        timePhase = cms.double(13.0),
        doSiPMSmearing = cms.bool(False),
        sipmTau = cms.double(0.),
        threshold_currentTDC = cms.double(-999.),
    ),
    ho = cms.PSet(
        readoutFrameSize = cms.int32(10),
        firstRing = cms.int32(1),
        binOfMaximum = cms.int32(5),
        doPhotoStatistics = cms.bool(True),
        simHitToPhotoelectrons = cms.double(4000.0), # is not actually used
        samplingFactors = cms.vdouble(231.0, 231.0, 231.0, 231.0,
            360.0, 360.0, 360.0, 360.0, 360.0, 360.0,
            360.0, 360.0, 360.0, 360.0, 360.0),
        syncPhase = cms.bool(True),
        timePhase = cms.double(5.0),
        timeSmearing = cms.bool(False),
        # 0 is HPD, 1 is SiPM, 2 fetches HPD/Zecotek/Hamamatsu from DB
        siPMCode = cms.int32(2),
        doSiPMSmearing = cms.bool(False),
        sipmTau = cms.double(5.),
        threshold_currentTDC = cms.double(-999.),
    ),
    hb = cms.PSet(
        readoutFrameSize = cms.int32(10),
        firstRing = cms.int32(1),
        binOfMaximum = cms.int32(5),
        doPhotoStatistics = cms.bool(True),
        simHitToPhotoelectrons = cms.double(2000.0),
        samplingFactors = cms.vdouble(
            125.44, 125.54, 125.32, 125.13, 124.46,
            125.01, 125.22, 125.48, 124.45, 125.90,
            125.83, 127.01, 126.82, 129.73, 131.83,
            143.52),            
        syncPhase = cms.bool(True),
        timePhase = cms.double(6.0),
        timeSmearing = cms.bool(True),
        doSiPMSmearing = cms.bool(False),
        sipmTau = cms.double(0.),
        threshold_currentTDC = cms.double(-999.),
    ),
    he = cms.PSet(
        readoutFrameSize = cms.int32(10),
        firstRing = cms.int32(16),
        binOfMaximum = cms.int32(5),
        doPhotoStatistics = cms.bool(True),
        simHitToPhotoelectrons = cms.double(2000.0),
        samplingFactors = cms.vdouble(
            210.55, 197.93, 186.12, 189.64, 189.63,
            190.28, 189.61, 189.60, 190.12, 191.22,
            190.90, 193.06, 188.42, 188.42),
        syncPhase = cms.bool(True),
        timePhase = cms.double(6.0),
        timeSmearing = cms.bool(True),
        doSiPMSmearing = cms.bool(False),
        sipmTau = cms.double(0.),
        threshold_currentTDC = cms.double(-999.),
    ),
    zdc = cms.PSet(
        readoutFrameSize = cms.int32(10),
        binOfMaximum = cms.int32(5),
        samplingFactor = cms.double(1.000),
        doPhotoStatistics = cms.bool(True),
        photoelectronsToAnalog = cms.double(1.843),
        simHitToPhotoelectrons = cms.double(6.0),
        syncPhase = cms.bool(True),
        timePhase = cms.double(-4.0),
        doSiPMSmearing = cms.bool(False),
        sipmTau = cms.double(0.),
        threshold_currentTDC = cms.double(-999.),
    ),
)

hcalSimParameters.hoZecotek = hcalSimParameters.ho.clone()

hcalSimParameters.hoHamamatsu = hcalSimParameters.ho.clone()

# Customises the HCal digitiser for post LS1 running
from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toModify( hcalSimParameters, 
    ho = dict(
        siPMCode = cms.int32(1)
    ),
    hf1 = dict( samplingFactor = cms.double(0.335) ),
    hf2 = dict( samplingFactor = cms.double(0.335) )
)

from Configuration.Eras.Modifier_run2_HE_2017_cff import run2_HE_2017
run2_HE_2017.toModify( hcalSimParameters,
    he = dict(
        doSiPMSmearing = cms.bool(True),
        sipmTau = cms.double(10.),
    )
)

from Configuration.Eras.Modifier_run2_HF_2017_cff import run2_HF_2017
run2_HF_2017.toModify( hcalSimParameters,
    hf1 = dict(
               readoutFrameSize = cms.int32(3), 
               binOfMaximum     = cms.int32(2)
              ),
    hf2 = dict(
               readoutFrameSize = cms.int32(3), 
               binOfMaximum     = cms.int32(2)
              )
)

from Configuration.Eras.Modifier_run2_HB_2018_cff import run2_HB_2018
run2_HB_2018.toModify( hcalSimParameters,
    hb = dict(
               readoutFrameSize = cms.int32(8), 
               binOfMaximum     = cms.int32(4)
              )
)
from Configuration.Eras.Modifier_run2_HE_2018_cff import run2_HE_2018
run2_HE_2018.toModify( hcalSimParameters,
    he = dict(
               readoutFrameSize = cms.int32(8), 
               binOfMaximum     = cms.int32(4),
               threshold_currentTDC = cms.double(18.7)
              )
)


from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
run3_HB.toModify( hcalSimParameters,
    hb = dict(
        doSiPMSmearing = cms.bool(True),
        threshold_currentTDC = cms.double(18.7),
        sipmTau = cms.double(10.),
    )
)

_newFactors = cms.vdouble(
    210.55, 197.93, 186.12, 189.64, 189.63,
    189.96, 190.03, 190.11, 190.18, 190.25,
    190.32, 190.40, 190.47, 190.54, 190.61,
    190.69, 190.83, 190.94, 190.94, 190.94,
    190.94, 190.94, 190.94, 190.94, 190.94,
    190.94, 190.94, 190.94, 190.94, 190.94,
    190.94, 190.94, 190.94, 190.94, 190.94,
    190.94, 190.94, 190.94, 190.94, 190.94,
    190.94, 190.94, 190.94, 190.94, 190.94,
    190.94, 190.94, 190.94, 190.94, 190.94,
    190.94, 190.94, 190.94, 190.94, 190.94,
    190.94, 190.94, 190.94, 190.94, 190.94,
    190.94, 190.94, 190.94, 190.94, 190.94,
    190.94, 190.94, 190.94, 190.94, 190.94,
    190.94, 190.94, 190.94, 190.94, 190.94,
    190.94, 190.94, 190.94, 190.94, 190.94,
    190.94, 190.94, 190.94, 190.94, 190.94 )

from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
phase2_hcal.toModify( hcalSimParameters,
    he = dict(
        samplingFactors = _newFactors,
    )
)
