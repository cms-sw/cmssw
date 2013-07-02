import FWCore.ParameterSet.Config as cms

process = cms.Process("FastGEM")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi')
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi')

process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMXML_cfi')
#process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMr08v01XML_cfi')
#process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMr10v01XML_cfi')
process.load('Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi')
process.load('Geometry.CommonDetUnit.globalTrackingGeometry_cfi')
process.load('Geometry.MuonNumbering.muonNumberingInitialization_cfi')
process.load('Geometry.GEMGeometry.gemGeometry_cfi')



### GEM-CSC trigger pad digi producer

#process.load('SimMuon.GEMDigitizer.muonGEMCSCPadDigis_cfi')

# the analyzer configuration
process.load('GEMCode.SimMuL1.FastGEMCSCProducer_cfi')
#process.FastGEMCSCProducer.verbose = 2
#process.FastGEMCSCProducer.createNtuple = cms.untracked.bool(False)
#process.FastGEMCSCProducer.lctInput = cms.untracked.InputTag("simCscTriggerPrimitiveDigis", "MPCSORTED")
#process.FastGEMCSCProducer.productInstanceName = cms.untracked.string("FastGEM")
#process.FastGEMCSCProducer.minPt = 1.5

#process.FastGEMCSCProducer.cscType = cms.untracked.int32(5)
#process.FastGEMCSCProducer.zOddGEM = cms.untracked.double(798.3)
#process.FastGEMCSCProducer.zEvenGEM = cms.untracked.double(796.2)

#process.FastGEMCSCProducer.cscType = cms.untracked.int32(2)
#process.FastGEMCSCProducer.zOddGEM = cms.untracked.double(569.7)
#process.FastGEMCSCProducer.zEvenGEM = cms.untracked.double(567.6)
#process.FastGEMCSCProducer.simTrackMatching.useCSCChamberTypes = cms.untracked.vint32( 2, )

#process.FastGEMCSCProducer.phiSmearCSC = [-1.]*11
#process.FastGEMCSCProducer.phiSmearGEM = [-1.]*11

#process.FastGEMCSCProducer.simTrackMatching.verboseSimHit = 1
#process.FastGEMCSCProducer.simTrackMatching.verboseCSCDigi = 1
#process.FastGEMCSCProducer.simTrackMatching.verboseCSCStub = 1
#process.FastGEMCSCProducer.simTrackMatching.simMuOnlyCSC = False
#process.FastGEMCSCProducer.simTrackMatching.discardEleHitsCSC = False


# extend the random generator service with our new producer:
process.RandomNumberGeneratorService.FastGEMCSCProducer = cms.PSet(
    initialSeed = cms.untracked.uint32(1234567),
    engineName = cms.untracked.string('HepJamesRandom')
)



### the analyzer configuration

process.load('GEMCode.GEMValidation.GEMCSCAnalyzer_cfi')

#process.GEMCSCAnalyzer.verbose = 2
process.GEMCSCAnalyzer.ntupleTrackChamberDelta = False
process.GEMCSCAnalyzer.ntupleTrackEff = True
process.GEMCSCAnalyzer.minPt = 1.5
#process.GEMCSCAnalyzer.simTrackMatching.verboseSimHit = 1
#process.GEMCSCAnalyzer.simTrackMatching.verboseGEMDigi = 1
#process.GEMCSCAnalyzer.simTrackMatching.verboseCSCDigi = 1
#process.GEMCSCAnalyzer.simTrackMatching.verboseCSCStub = 1
#process.GEMCSCAnalyzer.simTrackMatching.simMuOnlyGEM = False
#process.GEMCSCAnalyzer.simTrackMatching.simMuOnlyCSC = False
#process.GEMCSCAnalyzer.simTrackMatching.discardEleHitsCSC = False
#process.GEMCSCAnalyzer.simTrackMatching.discardEleHitsGEM = False


process.FastGEMCSCAnalyzer = process.GEMCSCAnalyzer.clone()
process.FastGEMCSCAnalyzer.stationsToUse = [1,2]
process.FastGEMCSCAnalyzer.maxEta = 2.48
process.FastGEMCSCAnalyzer.simTrackMatching.useCSCChamberTypes = [2, 5]
process.FastGEMCSCAnalyzer.simTrackMatching.cscLCTInput = cms.untracked.InputTag("FastGEMCSCProducer","FastGEM")




### GlobalTag ###

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'POSTLS161_V12::All'

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(50000) )

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )


### Input ###

dirPt5 = '/afs/cern.ch/cms/MUON/gem/muonGun_50k_pT20_digi_v3/'
dirPt20 = '/afs/cern.ch/cms/MUON/gem/muonGun_50k_pT5_digi_v2/'

dirPt5 = '/pnfs/cms/WAX/11/store/user/lpcgem/dildick/dildick/pT5_1M_v1/Digi+L1CSC-MuonGunPt5_1M/82325e40d6202e6fec2dd983c477f3ca/'
dirPt20 = '/pnfs/cms/WAX/11/store/user/lpcgem/dildick/dildick/pT20_1M_v1/Digi+L1CSC-MuonGunPt20_1M/82325e40d6202e6fec2dd983c477f3ca/'

dirPt5Pt40 = '/pnfs/cms/WAX/11/store/user/lpcgem/yasser1/yasser/MuomGUN_SIM_Pt5-40_50k/MuomGun_digi_Pt5-40_L1CSC_50k/82325e40d6202e6fec2dd983c477f3ca/'
dirPt2Pt50 = '/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/khotilov/MuomGUN_SIM_Pt2-50_100k/MuonGun_DIGI_L1_Pt2-50_100k/29891ddb18281fff4c42a6e5f5d4bc55/'

dir_pt5  = '/pnfs/cms/WAX/11/store/user/lpcgem/yasser1/yasser/muonGun_50k_pT5_lpcgem/MuomGunPtL1CSC50k5_digi/82325e40d6202e6fec2dd983c477f3ca/'
dir_pt10 = '/pnfs/cms/WAX/11/store/user/lpcgem/yasser1/yasser/muonGun_50k_pT10_lpcgem/MuomGunPt10L1CSC50k_digi/82325e40d6202e6fec2dd983c477f3ca/'
dir_pt15 = '/pnfs/cms/WAX/11/store/user/lpcgem/yasser1/yasser/muonGun_50k_pT15_lpcgem/MuomGun_Pt15_L1CSC_50k_digi/82325e40d6202e6fec2dd983c477f3ca/'
dir_pt20 = '/pnfs/cms/WAX/11/store/user/lpcgem/yasser1/yasser/muonGun_50k_pT20_lpcgem/MuomGunPt20L1CSC50k_digi/82325e40d6202e6fec2dd983c477f3ca/'
dir_pt30 = '/pnfs/cms/WAX/11/store/user/lpcgem/yasser1/yasser/MuonGun_Sim_50k_pT30_v2/MuomGun_Pt30_L1CSC_50k_digi/82325e40d6202e6fec2dd983c477f3ca/'
dir_pt40 = '/pnfs/cms/WAX/11/store/user/lpcgem/yasser1/yasser/muonGun_50k_pT40_lpcgem/MuomGunPt40L1CSC50k_digi/82325e40d6202e6fec2dd983c477f3ca/'

dir_pt5  = '/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/khotilov/MuomGUN_SIM_Pt5_50k_v3/MuonGUN_DIGI_L1_Pt5_50k_v3/c7e314a0a74ffa7b3385ebb7535b6693/'
dir_pt20 = '/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/khotilov/MuomGUN_SIM_Pt20_50k_v3/MuonGUN_DIGI_L1_Pt20_50k_v3/c7e314a0a74ffa7b3385ebb7535b6693/'


import os


inputDir = dir_pt5  ; ntupleFile = 'fast_ge21_pt5_sharp.root'
inputDir = dir_pt20 ; ntupleFile = 'fast_ge21_pt20_sharp.root'

inputDir = dir_pt5  ; ntupleFile = 'fast_ge21_pt5_smear.root'
inputDir = dir_pt20 ; ntupleFile = 'fast_ge21_pt20_smear.root'


#inputDir = dir_pt15  ; ntupleFile = 'fast_ge21_pt15.root'
#inputDir = dir_pt20  ; ntupleFile = 'fast_ge21_pt20.root'
#inputDir = dir_pt30  ; ntupleFile = 'fast_ge21_pt30.root'
#inputDir = dir_pt40  ; ntupleFile = 'fast_ge21_pt40.root'

#inputDir = dirPt2Pt50  ; ntupleFile = 'fast_ge21_pt2pt50.root'


ls = os.listdir(inputDir)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        #'file:out_sim.root'
#       'file:/tmp/khotilov/out.root'
#    'file:/afs/cern.ch/cms/MUON/gem/SingleMuPt40Fwd/SingleMuPt40Fwd_20121205_FixedGeometry_DIGI.root'
     #['file:'+inputDir+x for x in ls if x.endswith('root')]
     [inputDir[16:] + x for x in ls if x.endswith('root')]
    )
)

process.output = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string("test_out1k.root"),
)


### GEM-CSC trigger pad digi producer - just to be sure
process.load('SimMuon.GEMDigitizer.muonGEMCSCPadDigis_cfi')


process.TFileService = cms.Service("TFileService",
    fileName = cms.string(ntupleFile)
)

#process.contentAna = cms.EDAnalyzer("EventContentAnalyzer")


#process.p = cms.Path( process.FastGEMCSCProducer + process.contentAna)
#process.p = cms.Path( process.FastGEMCSCProducer)
process.p = cms.Path( process.simMuonGEMCSCPadDigis + process.GEMCSCAnalyzer + process.FastGEMCSCProducer + process.FastGEMCSCAnalyzer)
#process.p = cms.Path( process.simMuonGEMCSCPadDigis +  process.FastGEMCSCProducer + process.FastGEMCSCAnalyzer)

#process.out_step  = cms.EndPath(process.output)

