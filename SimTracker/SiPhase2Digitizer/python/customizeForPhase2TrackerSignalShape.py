import FWCore.ParameterSet.Config as cms

#
# activate signal shape in IT only
#

def customizeSiPhase2ITSignalShape(process):
    ## for standard mixing
    if hasattr(process,'mix') and hasattr(process.mix,'digitizers') and hasattr(process.mix.digitizers,'pixel'): 
        if hasattr(process.mix.digitizers.pixel,'PixelDigitizerAlgorithm'):
            print("# Activating signal shape emulation in IT pixel (planar)")
            process.mix.digitizers.pixel.PixelDigitizerAlgorithm.ApplyTimewalk = cms.bool(True)
        if hasattr(process.mix.digitizers.pixel,'Pixel3DDigitizerAlgorithm'):
            print("# Activating signal shape emulation in IT pixel (3D)")
            process.mix.digitizers.pixel.Pixel3DDigitizerAlgorithm.ApplyTimewalk = cms.bool(True)


    ## for pre-mixing
    if hasattr(process, "mixData") and hasattr(process.mixData, "workers") and hasattr(process.mixData.workers, "pixel"):
        if hasattr(process.mixData.workers.pixel,'PixelDigitizerAlgorithm'):
            print("# Activating signal shape emulation in IT pixel (planar)")
            process.mixData.workers.pixel.PixelDigitizerAlgorithm.ApplyTimewalk = cms.bool(True)
        if hasattr(process.mixData.workers.pixel,'Pixel3DDigitizerAlgorithm'):
            print("# Activating signal shape emulation in IT pixel (3D)")
            process.mixData.workers.pixel.Pixel3DDigitizerAlgorithm.ApplyTimewalk = cms.bool(True)

    return process

