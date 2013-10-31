import FWCore.ParameterSet.Config as cms

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
    return (process)

def customise_pixelMixing_PU(process):
    if hasattr(process,'noPixelMixing'):
        return process #avoid race condition
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

def customise_noPixelMixing(process):
    process.noPixelMixing=True
    if hasattr(process,'mix'):
        process.mix.digitizers.pixel.thePixelColEfficiency_BPix1 = cms.double(1.0)
        process.mix.digitizers.pixel.thePixelColEfficiency_BPix2 = cms.double(1.0)
        process.mix.digitizers.pixel.thePixelColEfficiency_BPix3 = cms.double(1.0)
        process.mix.digitizers.pixel.thePixelColEfficiency_BPix4 = cms.double(1.0)
        process.mix.digitizers.pixel.thePixelColEfficiency_FPix1 = cms.double(1.0)
        process.mix.digitizers.pixel.thePixelColEfficiency_FPix2 = cms.double(1.0)
        process.mix.digitizers.pixel.thePixelColEfficiency_FPix3 = cms.double(1.0)
    return process    
        

def customise_NoCrossing_PU(process):
    process=customise_pixelMixing_PU(process)
    process=customise_NoCrossing(process)
    return (process)

