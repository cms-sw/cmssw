import FWCore.ParameterSet.Config as cms

process = cms.Process("GlobalVal")
process.load("SimGeneral.MixingModule.mixLowLumPU_cfi")

process.load("DQM.SiStripCommon.DaqMonitorROOTBackEnd_cfi")

process.MessageLogger = cms.Service("MessageLogger")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        mix = cms.untracked.uint32(56789)
    )
)

process.source = cms.Source("PoolSource",
#old    fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/RelVal/2007/7/11/RelVal-RelVal160pre4SingleEPt35-1184176348/0000/5EF3794C-7530-DC11-833F-000423D6C8EE.root')

   fileNames = cms.untracked.vstring(
   	'/store/relval/CMSSW_3_0_0_pre6/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/28116A15-E9DD-DD11-9BA6-001617E30F4C.root',
	'/store/relval/CMSSW_3_0_0_pre6/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/28116A15-E9DD-DD11-9BA6-001617E30F4C.root',
        '/store/relval/CMSSW_3_0_0_pre6/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/341262AD-41DE-DD11-B261-000423D94990.root',
        '/store/relval/CMSSW_3_0_0_pre6/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/6CD0AB7B-EDDD-DD11-A19A-000423D98B6C.root',
        '/store/relval/CMSSW_3_0_0_pre6/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/D83A1B28-ECDD-DD11-A6D2-000423D9853C.root')
)

#readFiles = cms.untracked.vstring()
#secFiles = cms.untracked.vstring() 
#source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
#readFiles.extend( [
#       '/store/relval/CMSSW_3_0_0_pre6/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/28116A15-E9DD-DD11-9BA6-001617E30F4C.root',
#       '/store/relval/CMSSW_3_0_0_pre6/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/341262AD-41DE-DD11-B261-000423D94990.root',
#       '/store/relval/CMSSW_3_0_0_pre6/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/6CD0AB7B-EDDD-DD11-A19A-000423D98B6C.root',
#       '/store/relval/CMSSW_3_0_0_pre6/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/D83A1B28-ECDD-DD11-A6D2-000423D9853C.root' ] );


#secFiles.extend( [
#               ] )


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.test = cms.EDAnalyzer("GlobalTest",
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-5),
    fileName = cms.string('GlobalHistos.root')
)

process.p = cms.Path(process.mix+process.test)
#process.outpath = cms.EndPath(process.out)
process.mix.input.type = 'fixed'
process.mix.input.nbPileupEvents = cms.PSet(
    averageNumber = cms.double(10.0)
)


