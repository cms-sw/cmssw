import FWCore.ParameterSet.Config as cms

from SimGeneral.MixingModule.mixObjects_cfi import *

process = cms.Process("PRODVAL1")
process.load("DQM.SiStripCommon.DaqMonitorROOTBackEnd_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        mix = cms.untracked.uint32(12345+5)
    )
)

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/RelVal/2007/7/11/RelVal-RelVal160pre4SingleEPt35-1184176348/0000/5EF3794C-7530-DC11-833F-000423D6C8EE.root')
#    fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/relval/CMSSW_3_0_0_pre6/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/28116A15-E9DD-DD11-9BA6-001617E30F4C.root')
   fileNames = cms.untracked.vstring(
   	'/store/relval/CMSSW_3_0_0_pre6/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/28116A15-E9DD-DD11-9BA6-001617E30F4C.root',
	'/store/relval/CMSSW_3_0_0_pre6/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/28116A15-E9DD-DD11-9BA6-001617E30F4C.root',
        '/store/relval/CMSSW_3_0_0_pre6/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/341262AD-41DE-DD11-B261-000423D94990.root',
        '/store/relval/CMSSW_3_0_0_pre6/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/6CD0AB7B-EDDD-DD11-A19A-000423D98B6C.root',
        '/store/relval/CMSSW_3_0_0_pre6/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/D83A1B28-ECDD-DD11-A6D2-000423D9853C.root')

)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.mix = cms.EDProducer("MixingModule",
    LabelPlayback = cms.string(''),
    
    maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
    ),
    
    input = cms.SecSource("PoolRASource",
    
#    fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/RelVal/2007/7/11/RelVal-RelVal160pre4SingleEPt35-1184176348/0000/5EF3794C-7530-DC11-833F-000423D6C8EE.root'),
#    fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/relval/CMSSW_3_0_0_pre6/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/28116A15-E9DD-DD11-9BA6-001617E30F4C.root'),
    fileNames = cms.untracked.vstring(
   	'/store/relval/CMSSW_3_0_0_pre6/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/28116A15-E9DD-DD11-9BA6-001617E30F4C.root',
	'/store/relval/CMSSW_3_0_0_pre6/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/28116A15-E9DD-DD11-9BA6-001617E30F4C.root',
        '/store/relval/CMSSW_3_0_0_pre6/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/341262AD-41DE-DD11-B261-000423D94990.root',
        '/store/relval/CMSSW_3_0_0_pre6/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/6CD0AB7B-EDDD-DD11-A19A-000423D98B6C.root',
        '/store/relval/CMSSW_3_0_0_pre6/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/D83A1B28-ECDD-DD11-A6D2-000423D9853C.root'),

        seed = cms.int32(1234567),
        type = cms.string('fixed'),
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(1.0)
        ),

        maxEventsToSkip = cms.untracked.uint32(0),
        
    ),
    
    maxBunch = cms.int32(12345),
    minBunch = cms.int32(12345),
    bunchspace = cms.int32(25),
    Label = cms.string(''),
    
    mixObjects = cms.PSet(
        mixCH = cms.PSet(
            mixCaloHits
        ),
        mixTracks = cms.PSet(
            mixSimTracks
        ),
        mixVertices = cms.PSet(
            mixSimVertices
        ),
        mixSH = cms.PSet(
            mixSimHits
        ),
        mixHepMC = cms.PSet(
            mixHepMCProducts
        )
    )
)

process.test = cms.EDAnalyzer("TestSuite",
    maxBunch = cms.int32(34567),
    BunchNr = cms.int32(12345),
    minBunch = cms.int32(23456),
    fileName = cms.string('histos.root')
)

process.p = cms.Path(process.mix+process.test)
#process.outpath = cms.EndPath(process.out)


