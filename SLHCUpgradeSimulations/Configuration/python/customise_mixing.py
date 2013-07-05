import FWCore.ParameterSet.Config as cms

#this is obsolete I think
def customise_pixelMixing(process):
    process.mix.digitizers.pixel.MissCalibrate = False
    process.mix.digitizers.pixel.LorentzAngle_DB = False
    process.mix.digitizers.pixel.killModules = False
    process.mix.digitizers.pixel.useDB = False
    process.mix.digitizers.pixel.DeadModules_DB = False
    process.mix.digitizers.pixel.NumPixelBarrel = cms.int32(4)
    process.mix.digitizers.pixel.NumPixelEndcap = cms.int32(3)
    process.mix.digitizers.pixel.ThresholdInElectrons_FPix = cms.double(2000.0)
    process.mix.digitizers.pixel.ThresholdInElectrons_BPix = cms.double(2000.0)
    process.mix.digitizers.pixel.ThresholdInElectrons_BPix_L1 = cms.double(2000.0)
    process.mix.digitizers.pixel.thePixelColEfficiency_BPix4 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelEfficiency_BPix4 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelChipEfficiency_BPix4 = cms.double(0.999)
    return (process)

# Remove the Crossing Frames to save memory
def customise_NoCrossing(process):
    process.mix.mixObjects.mixSH.crossingFrames = cms.untracked.vstring(
        'BSCHits',
        'FP420SI',
        'MuonCSCHits',
        'MuonDTHits',
        'MuonRPCHits',
        'TotemHitsRP',
        'TotemHitsT1',
        'TotemHitsT2Gem')
    process.mix.mixObjects.mixCH.crossingFrames = cms.untracked.vstring('')
    process.mix.mixObjects.mixTracks.makeCrossingFrame = cms.untracked.bool(False)
    process.mix.mixObjects.mixVertices.makeCrossingFrame = cms.untracked.bool(False)
    process.mix.mixObjects.mixHepMC.makeCrossingFrame = cms.untracked.bool(False)
    process.digitisation_step.remove(process.simSiStripDigiSimLink)
    process.digitisation_step.remove(process.mergedtruth)
    return (process)

def customise_pixelMixing_PU(process):
    if hasattr(process,'mix'): 
        n=0
        if hasattr(process.mix,'input'):
            n=process.mix.input.nbPileupEvents.averageNumber.value()
        if n>0:
            process.mix.digitizers.pixel.thePixelColEfficiency_BPix1 = cms.double(1.0-(0.0238*n/50.0))
            process.mix.digitizers.pixel.thePixelColEfficiency_BPix2 = cms.double(1.0-(0.0046*n/50.0))
            process.mix.digitizers.pixel.thePixelColEfficiency_BPix3 = cms.double(1.0-(0.0018*n/50.0))
            process.mix.digitizers.pixel.thePixelColEfficiency_BPix4 = cms.double(1.0-(0.0008*n/50.0))
            process.mix.digitizers.pixel.thePixelColEfficiency_FPix1 = cms.double(1.0-(0.0018*n/50.0))
            process.mix.digitizers.pixel.thePixelColEfficiency_FPix2 = cms.double(1.0-(0.0018*n/50.0))
            process.mix.digitizers.pixel.thePixelColEfficiency_FPix3 = cms.double(1.0-(0.0018*n/50.0))
        
    return (process)

def customise_NoCrossing_PU(process):
    process=customise_pixelMixing_PU(process)
    process=customise_NoCrossing(process)
    return (process)

