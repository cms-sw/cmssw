def customise_HFSLrun1(process):
    #enable Run 1 HF shower library
    process.g4SimHits.HCalSD.UseShowerLibrary   = True
    process.g4SimHits.HCalSD.UseParametrize     = False
    process.g4SimHits.HCalSD.UsePMTHits         = False
    process.g4SimHits.HCalSD.UseFibreBundleHits = False
    process.g4SimHits.HFShower.UseShowerLibrary = True
    process.g4SimHits.HFShower.UseHFGflash      = False

    return process
