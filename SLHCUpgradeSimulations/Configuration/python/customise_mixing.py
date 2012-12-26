import FWCore.ParameterSet.Config as cms

# Remove the Crossing Frames to save memory
def customise_NoCrossing(process):
    process.load('SLHCUpgradeSimulations.Geometry.mixLowLumPU_Phase1_R30F12_cff')
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
    process.mix.digitizers.pixel.thePixelColEfficiency_BPix4 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelEfficiency_BPix4 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelChipEfficiency_BPix4 = cms.double(0.999)

    return (process)

