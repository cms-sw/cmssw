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
        readoutFrameSize = cms.int32(5),
        binOfMaximum = cms.int32(3),
        samplingFactor = cms.double(0.383),
        doPhotoStatistics = cms.bool(True),
        photoelectronsToAnalog = cms.double(2.79),
        simHitToPhotoelectrons = cms.double(6.0),
        syncPhase = cms.bool(True),
        timePhase = cms.double(14.0)        
    ),
    hf2 = cms.PSet(
        readoutFrameSize = cms.int32(5),
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

hcalSimParameters.hoHamamatsu = hcalSimParameters.ho.clone()
hcalSimParameters.hoHamamatsu.pixels = cms.int32(960)
hcalSimParameters.hoHamamatsu.photoelectronsToAnalog = [3.0]*16
