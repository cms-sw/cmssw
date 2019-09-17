import FWCore.ParameterSet.Config as cms

#
# activate the SiStrip AVP dynamic gain simulation and loads from DB the corresponding conditions
#
def activateSiStripAPVDynamicGain(process):

    if hasattr(process,'mix') and hasattr(process.mix,'digitizers') and hasattr(process.mix.digitizers,'strip'):
        print("activating SiStrip APV Dynamic Gain simulation")
        process.mix.digitizers.strip.includeAPVSimulation = cms.bool(True)

    if not hasattr(process.GlobalTag,'toGet'):
        process.GlobalTag.toGet=cms.VPSet()

    process.GlobalTag.toGet.extend(cms.VPSet(cms.PSet(record = cms.string('SiStripApvSimulationParametersRcd'),
                                                      tag = cms.string('SiStripApvSimulationParameters_2016preVFP_v1'),
                                                      #connect = cms.string('frontier://FrontierPrep/CMS_CONDITIONS'))
                                                      connect = cms.string('sqlite_file:/tmp/musich/CMSSW_11_0_X_2019-09-16-2300/src/CondTools/SiStrip/test/dbfile.db'))
                                         )
                                   )            
    return process
