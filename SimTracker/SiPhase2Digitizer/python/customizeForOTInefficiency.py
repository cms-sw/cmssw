import FWCore.ParameterSet.Config as cms

def _commonCustomizeForInefficiency(process):
    ## for standard mixing
    if hasattr(process,'mix') and hasattr(process.mix,'digitizers') and hasattr(process.mix.digitizers,'pixel'): 
        if hasattr(process.mix.digitizers.pixel,'PSPDigitizerAlgorithm'):
            print("# Activating Bias Rail Inefficiency in macro-pixels")
            process.mix.digitizers.pixel.PSPDigitizerAlgorithm.BiasRailInefficiencyFlag = cms.int32(1)

        if hasattr(process.mix.digitizers.pixel,'PSSDigitizerAlgorithm'):
            print("# Activating bad strip simulation for s-sensors in PS modules from DB")
            process.mix.digitizers.pixel.PSSDigitizerAlgorithm.KillModules = cms.bool(True)
            process.mix.digitizers.pixel.PSSDigitizerAlgorithm.DeadModules_DB = cms.bool(True)

        if hasattr(process.mix.digitizers.pixel,'SSDigitizerAlgorithm'):
            print("# Activating bad strip simulation for SS modules from DB")
            process.mix.digitizers.pixel.SSDigitizerAlgorithm.KillModules = cms.bool(True)
            process.mix.digitizers.pixel.SSDigitizerAlgorithm.DeadModules_DB = cms.bool(True)

    ## for pre-mixing
    if hasattr(process, "mixData") and hasattr(process.mixData, "workers") and hasattr(process.mixData.workers, "pixel"):
        if hasattr(process.mixData.workers.pixel,'PSPDigitizerAlgorithm'):
            print("# Activating Bias Rail Inefficiency in macro-pixels")
            process.mixData.workers.pixel.PSPDigitizerAlgorithm.BiasRailInefficiencyFlag = cms.int32(1)

        if hasattr(process.mixData.workers.pixel,'PSSDigitizerAlgorithm'):
            print("# Activating bad strip simulation for s-sensors in PS modules from DB")
            process.mixData.workers.pixel.PSSDigitizerAlgorithm.KillModules = cms.bool(True)
            process.mixData.workers.pixel.PSSDigitizerAlgorithm.DeadModules_DB = cms.bool(True)

        if hasattr(process.mixData.workers.pixel,'SSDigitizerAlgorithm'):
            print("# Activating bad strip simulation for SS modules from DB")
            process.mixData.workers.pixel.SSDigitizerAlgorithm.KillModules = cms.bool(True)
            process.mixData.workers.pixel.SSDigitizerAlgorithm.DeadModules_DB = cms.bool(True)

    return process

#
# activate bias rail inefficiency and 1% random bad strips
#
def customizeSiPhase2OTInefficiencyOnePercent(process):

    _commonCustomizeForInefficiency(process)

    if hasattr(process,'SiPhase2OTFakeBadStripsESSource') :
        print("# Adding 1% of randomly generated bad strips")
        process.SiPhase2OTFakeBadStripsESSource.badComponentsFraction = 0.01 # 1% bad components

    return process

#
# activate bias rail inefficiency and 5% random bad strips
#
def customizeSiPhase2OTInefficiencyFivePercent(process):

    _commonCustomizeForInefficiency(process)

    if hasattr(process,'SiPhase2OTFakeBadStripsESSource') :
        print("# Adding 5% of randomly generated bad strips")
        process.SiPhase2OTFakeBadStripsESSource.badComponentsFraction = 0.05 # 5% bad components

    return process

#
# activate bias rail inefficiency and 10% random bad strips
#
def customizeSiPhase2OTInefficiencyTenPercent(process):

    _commonCustomizeForInefficiency(process)
 
    if hasattr(process,'SiPhase2OTFakeBadStripsESSource') :
        print("# Adding 10% of randomly generated bad strips")
        process.SiPhase2OTFakeBadStripsESSource.badComponentsFraction = 0.1 # 10% bad components

    return process
