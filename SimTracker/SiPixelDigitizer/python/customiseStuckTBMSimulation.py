import FWCore.ParameterSet.Config as cms

#
# activate the stuck-TBM simulation and loads from DB the corresponding conditions
#
def activateStuckTBMSimulation2018NoPU(process):

    if hasattr(process,'mix') and hasattr(process.mix,'digitizers') and hasattr(process.mix.digitizers,'pixel'):
        print("activating Pixel Stuck TBM Simulation")
        process.mix.digitizers.pixel.KillBadFEDChannels = cms.bool(True)

    if not hasattr(process.GlobalTag,'toGet'):
        process.GlobalTag.toGet=cms.VPSet()

    process.GlobalTag.toGet.extend(cms.VPSet(cms.PSet(record = cms.string('SiPixelStatusScenarioProbabilityRcd'),
                                                      tag = cms.string('SiPixelQualityProbabilities_2018_noPU_v0_mc'),
                                                      connect = cms.string('frontier://FrontierPrep/CMS_CONDITIONS')),
                                             cms.PSet(record = cms.string('SiPixelStatusScenariosRcd'),
                                                      tag = cms.string('SiPixelFEDChannelContainer_StuckTBM_2018_v0_mc'),
                                                      connect = cms.string('frontier://FrontierPrep/CMS_CONDITIONS')
                                                      )
                                             )
                                   )            
    return process
                
