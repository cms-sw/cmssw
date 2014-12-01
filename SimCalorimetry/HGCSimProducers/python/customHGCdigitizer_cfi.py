import FWCore.ParameterSet.Config as cms

"""
implement here different variations for testing purposes
"""
def customHGCdigitizer(process, version='simple0', debug=False):
    
    #configure digitizer model
    if version=='femodel':
        print 'Adapting for FE model (14th Nov)'
        process.mix.digitizers.hgceeDigitizer.digiCfg.feCfg.fwVersion = cms.uint32(1)
        process.mix.digitizers.hgceeDigitizer.digiCfg.feCfg.shaperN = cms.double(8.02)
        process.mix.digitizers.hgceeDigitizer.digiCfg.feCfg.shaperTau = cms.double(3.26)
        process.mix.digitizers.hgchefrontDigitizer.digiCfg.feCfg.fwVersion = cms.uint32(1)
        process.mix.digitizers.hgchefrontDigitizer.digiCfg.feCfg.shaperN = cms.double(8.02)
        process.mix.digitizers.hgchefrontDigitizer.digiCfg.feCfg.shaperTau = cms.double(3.26)
        process.mix.digitizers.hgchebackDigitizer.digiCfg.feCfg.fwVersion = cms.uint32(0)
        process.mix.digitizers.hgchebackDigitizer.digiCfg.feCfg.shaperN = cms.double(1)
        process.mix.digitizers.hgchebackDigitizer.digiCfg.feCfg.shaperTau = cms.double(10)
    elif version.find('simple')>=0 :
        tau=float(version.replace('simple',''))
        print 'Adapting simple pulse shape with tau=%f'%tau
        process.mix.digitizers.hgceeDigitizer.digiCfg.feCfg.fwVersion = cms.uint32(0)
        process.mix.digitizers.hgceeDigitizer.digiCfg.feCfg.shaperTau = cms.double(tau)
        process.mix.digitizers.hgchefrontDigitizer.digiCfg.feCfg.fwVersion = cms.uint32(0)
        process.mix.digitizers.hgchefrontDigitizer.digiCfg.feCfg.shaperTau = cms.double(tau)
        process.mix.digitizers.hgchebackDigitizer.digiCfg.feCfg.fwVersion = cms.uint32(0)
        process.mix.digitizers.hgchebackDigitizer.digiCfg.feCfg.shaperN = cms.double(1)
        process.mix.digitizers.hgchebackDigitizer.digiCfg.feCfg.shaperTau = cms.double(tau)

    #save all the time samples to the output
    if debug:
        process.mix.digitizers.hgceeDigitizer.digiCfg.doTimeSamples      = cms.bool(True)
        process.mix.digitizers.hgchefrontDigitizer.digiCfg.doTimeSamples = cms.bool(True)
        process.mix.digitizers.hgchebackDigitizer.digiCfg.doTimeSamples  = cms.bool(True)
