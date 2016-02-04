import FWCore.ParameterSet.Config as cms

hcalSimParameters = cms.PSet(
    #  In HF, the SimHits energy is actually
    # the number of photoelectrons from the shower
    # library.  However, we need a lot more
    # smearing, because the first level of the PMT
    # only gives ~6 electrons per pe, which
    # comes out to a 40% smearing of the single pe peak!
    #
    hf1 = cms.PSet(
        readoutFrameSize = cms.int32(10),
        binOfMaximum = cms.int32(5),
        samplingFactor = cms.double(0.383),
        doPhotoStatistics = cms.bool(True),
        photoelectronsToAnalog = cms.double(2.79),
        simHitToPhotoelectrons = cms.double(6.0),
        syncPhase = cms.bool(True),
        timePhase = cms.double(9.0)        
    ),
    hf2 = cms.PSet(
        readoutFrameSize = cms.int32(10),
        binOfMaximum = cms.int32(5),
        samplingFactor = cms.double(0.368),
        doPhotoStatistics = cms.bool(True),
        photoelectronsToAnalog = cms.double(1.843),
        simHitToPhotoelectrons = cms.double(6.0),
        syncPhase = cms.bool(True),
        timePhase = cms.double(9.0)
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
        # 0 is HPD, 1 is SiPM, 2, is hardcoded combination 
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
            118.98, 118.60, 118.97, 118.76, 119.13,
            118.74, 117.80, 118.14, 116.87, 117.87,
            117.46, 116.79, 117.15, 117.29, 118.41,
            134.86),
        syncPhase = cms.bool(True),
        timePhase = cms.double(5.0),
        timeSmearing = cms.bool(True),
        siPMCells = cms.vint32()
    ),
    he = cms.PSet(
        readoutFrameSize = cms.int32(10),
        firstRing = cms.int32(16),
        binOfMaximum = cms.int32(5),
        doPhotoStatistics = cms.bool(True),
        photoelectronsToAnalog = cms.vdouble([0.3305]*14),
        simHitToPhotoelectrons = cms.double(2000.0),
        samplingFactors = cms.vdouble(
            197.84, 184.67, 170.60, 172.06, 173.08,
            171.92, 173.00, 173.22, 173.72, 174.21,
            173.91, 175.88, 171.65, 171.65),
        syncPhase = cms.bool(True),
        timePhase = cms.double(5.0),
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

hcalSimParameters.hoHamamatsu = hcalSimParameters.ho.clone()
hcalSimParameters.hoHamamatsu.pixels = cms.int32(960)
hcalSimParameters.hoHamamatsu.photoelectronsToAnalog = [3.0]*16
