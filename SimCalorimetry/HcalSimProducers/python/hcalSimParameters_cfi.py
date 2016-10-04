import FWCore.ParameterSet.Config as cms

# This object modifies hcalSimParameters for different scenarios
from Configuration.StandardSequences.Eras import eras

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
        timePhase = cms.double(14.0)        
    ),
    hf2 = cms.PSet(
        readoutFrameSize = cms.int32(4),
        binOfMaximum = cms.int32(3),
        samplingFactor = cms.double(0.368),
        doPhotoStatistics = cms.bool(True),
        photoelectronsToAnalog = cms.double(1.843),
        simHitToPhotoelectrons = cms.double(6.0),
        syncPhase = cms.bool(True),
        timePhase = cms.double(13.0)
    ),
    ho = cms.PSet(
        readoutFrameSize = cms.int32(10),
        firstRing = cms.int32(1),
        binOfMaximum = cms.int32(5),
        doPhotoStatistics = cms.bool(True),
        photoelectronsToAnalog = cms.vdouble(0.24, 0.24, 0.24, 0.24,
            0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 
            0.17, 0.17, 0.17, 0.17, 0.17), 
        simHitToPhotoelectrons = cms.double(4000.0), # is not actually used
        samplingFactors = cms.vdouble(231.0, 231.0, 231.0, 231.0,
            360.0, 360.0, 360.0, 360.0, 360.0, 360.0,
            360.0, 360.0, 360.0, 360.0, 360.0),
        syncPhase = cms.bool(True),
        timePhase = cms.double(5.0),
        timeSmearing = cms.bool(False),
        # 0 is HPD, 1 is SiPM, 2 fetches HPD/Zecotek/Hamamatsufrom DB
        siPMCode = cms.int32(2)
    ),
    hb = cms.PSet(
        readoutFrameSize = cms.int32(10),
        firstRing = cms.int32(1),
        binOfMaximum = cms.int32(5),
        doPhotoStatistics = cms.bool(True),
        photoelectronsToAnalog = cms.vdouble([0.3305]*16),
        simHitToPhotoelectrons = cms.double(2000.0),
        samplingFactors = cms.vdouble(
            125.44, 125.54, 125.32, 125.13, 124.46,
            125.01, 125.22, 125.48, 124.45, 125.90,
            125.83, 127.01, 126.82, 129.73, 131.83,
            143.52),            
        syncPhase = cms.bool(True),
        timePhase = cms.double(6.0),
        timeSmearing = cms.bool(True)
    ),
    he = cms.PSet(
        readoutFrameSize = cms.int32(10),
        firstRing = cms.int32(16),
        binOfMaximum = cms.int32(5),
        doPhotoStatistics = cms.bool(True),
        photoelectronsToAnalog = cms.vdouble([0.3305]*14),
        simHitToPhotoelectrons = cms.double(2000.0),
        samplingFactors = cms.vdouble(
            210.55, 197.93, 186.12, 189.64, 189.63,
            190.28, 189.61, 189.60, 190.12, 191.22,
            190.90, 193.06, 188.42, 188.42),
        syncPhase = cms.bool(True),
        timePhase = cms.double(6.0),
        timeSmearing = cms.bool(True)
    ),
    zdc = cms.PSet(
        readoutFrameSize = cms.int32(10),
        binOfMaximum = cms.int32(5),
        samplingFactor = cms.double(1.000),
        doPhotoStatistics = cms.bool(True),
        photoelectronsToAnalog = cms.double(1.843),
        simHitToPhotoelectrons = cms.double(6.0),
        syncPhase = cms.bool(True),
        timePhase = cms.double(-4.0)
    ),
)

hcalSimParameters.hoZecotek = hcalSimParameters.ho.clone()
hcalSimParameters.hoZecotek.pixels = cms.int32(36000)
hcalSimParameters.hoZecotek.photoelectronsToAnalog = [3.0]*16
hcalSimParameters.hoZecotek.sipmDarkCurrentuA = cms.double(0.055)
hcalSimParameters.hoZecotek.sipmCrossTalk = cms.double(0.32)

hcalSimParameters.hoHamamatsu = hcalSimParameters.ho.clone()
hcalSimParameters.hoHamamatsu.pixels = cms.int32(960)
hcalSimParameters.hoHamamatsu.photoelectronsToAnalog = [3.0]*16
hcalSimParameters.hoHamamatsu.sipmDarkCurrentuA = cms.double(0.055)
hcalSimParameters.hoHamamatsu.sipmCrossTalk = cms.double(0.32)

# Customises the HCal digitiser for post LS1 running
eras.run2_common.toModify( hcalSimParameters, 
    ho = dict(
        photoelectronsToAnalog = cms.vdouble([4.0]*16),
        siPMCode = cms.int32(1),
        pixels = cms.int32(2500),
        doSiPMSmearing = cms.bool(False)
    ),
    hf1 = dict( samplingFactor = cms.double(0.67) ),
    hf2 = dict( samplingFactor = cms.double(0.67) )
)

eras.run2_HE_2017.toModify( hcalSimParameters,
    he = dict(
        photoelectronsToAnalog = cms.vdouble([57.5]*14),
        pixels = cms.int32(27370), 
        sipmDarkCurrentuA = cms.double(0.055),
        sipmCrossTalk = cms.double(0.32),
        doSiPMSmearing = cms.bool(True),
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

eras.phase2_hcal.toModify( hcalSimParameters,
    hb = dict(
        photoelectronsToAnalog = cms.vdouble([57.5]*16),
        pixels = cms.int32(27370),
        sipmDarkCurrentuA = cms.double(0.055),
        sipmCrossTalk = cms.double(0.32),
        doSiPMSmearing = cms.bool(True),
    ),
    he = dict(
        samplingFactors = _newFactors,
        photoelectronsToAnalog = cms.vdouble([57.5]*len(_newFactors)),
        pixels = cms.int32(27370),
        sipmDarkCurrentuA = cms.double(0.055),
        sipmCrossTalk = cms.double(0.32),
        doSiPMSmearing = cms.bool(True),
    )
)
