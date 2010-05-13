import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

isData = False
process.load("Configuration/StandardSequences/Geometry_cff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
process.load("Configuration/StandardSequences/RawToDigi_cff")
process.load('Configuration.EventContent.EventContent_cff')
process.load("RecoMET/Configuration/RecoMET_BeamHaloId_cff")
process.load('RecoMET.METAnalyzers.CSCHaloFilter_cfi')
process.CSCBasedHaloFilter.FilterDigiLevel = True    #### For < 36X, this requires the RAW-DIGI
process.CSCBasedHaloFilter.FilterTriggerLevel = True
process.CSCBasedHaloFilter.FilterRecoLevel = True 

#FOR MC = 6, FOR DATA =3
if isData:
    process.CSCBasedHaloFilter.ExpectedBX = cms.int32(3)
    process.CSCHaloData.ExpectedBX = cms.int32(3)
    process.GlobalTag.globaltag = 'GR_R_35X_V8::All'
else:
    process.CSCBasedHaloFilter.ExpectedBX = cms.int32(6)
    process.CSCHaloData.ExpectedBX = cms.int32(6)
    process.GlobalTag.globaltag = 'MC_3XY_V27::All'


process.pathCSCBasedHaloFilter = cms.Path(process.CSCBasedHaloFilter)


process.load("Configuration/StandardSequences/ReconstructionCosmics_cff")

process.load("RecoMuon/Configuration/RecoMuon_cff")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",

                            fileNames = cms.untracked.vstring(
    #'/store/relval/CMSSW_3_5_8/RelValMinBias/GEN-SIM-RECO/START3X_V26-v1/0017/E26CCE65-8F52-DF11-9FA4-00304867901A.root',
    #'/store/relval/CMSSW_3_5_8/RelValMinBias/GEN-SIM-RECO/START3X_V26-v1/0016/C86F45C5-4D52-DF11-9725-001A92810AD4.root'
    #'/store/relval/CMSSW_3_5_8/RelValTTbar/GEN-SIM-RECO/MC_3XY_V26-v1/0017/E608ED1A-8F52-DF11-B455-0030486790BA.root',
    #'/store/relval/CMSSW_3_5_8/RelValTTbar/GEN-SIM-RECO/MC_3XY_V26-v1/0016/E6A643C1-4E52-DF11-87FE-00261894392D.root',
    #'/store/relval/CMSSW_3_5_8/RelValTTbar/GEN-SIM-RECO/MC_3XY_V26-v1/0016/E62330A9-4852-DF11-A46E-001A92971B8C.root',
    #'/store/relval/CMSSW_3_5_8/RelValTTbar/GEN-SIM-RECO/MC_3XY_V26-v1/0016/70881F95-4252-DF11-A409-002618FDA248.root',
    #'/store/relval/CMSSW_3_5_8/RelValTTbar/GEN-SIM-RECO/MC_3XY_V26-v1/0016/5E421825-4952-DF11-91E5-002618943880.root',
    #'/store/relval/CMSSW_3_5_8/RelValTTbar/GEN-SIM-RECO/MC_3XY_V26-v1/0016/40796A28-4952-DF11-B6C6-00261894387E.root',
    #'/store/relval/CMSSW_3_5_8/RelValTTbar/GEN-SIM-RECO/MC_3XY_V26-v1/0016/36C1F1DB-4352-DF11-918A-002618FDA279.root',
    #'/store/relval/CMSSW_3_5_8/RelValTTbar/GEN-SIM-RECO/MC_3XY_V26-v1/0016/008C33BB-4752-DF11-BAE4-00304867D836.root',
    #'/store/relval/CMSSW_3_5_8/RelValTTbar/GEN-SIM-RECO/MC_3XY_V26-v1/0016/005017A4-4852-DF11-BF8A-0026189438AD.root'
    
#    '/store/relval/CMSSW_3_5_8/RelValZMM/GEN-SIM-RECO/START3X_V26-v1/0017/46476117-8F52-DF11-8F82-00304866C398.root',
#    '/store/relval/CMSSW_3_5_8/RelValZMM/GEN-SIM-RECO/START3X_V26-v1/0016/FA3ED2C4-5B52-DF11-8DC9-001A92971B7E.root',
#    '/store/relval/CMSSW_3_5_8/RelValZMM/GEN-SIM-RECO/START3X_V26-v1/0016/F263C13D-5E52-DF11-BF2E-0026189438BC.root',
#    '/store/relval/CMSSW_3_5_8/RelValZMM/GEN-SIM-RECO/START3X_V26-v1/0016/BE8075FA-5452-DF11-AF79-0030486790BE.root',
#    '/store/relval/CMSSW_3_5_8/RelValZMM/GEN-SIM-RECO/START3X_V26-v1/0016/82C47C1C-5652-DF11-804B-001A92971ACC.root',
#    '/store/relval/CMSSW_3_5_8/RelValZMM/GEN-SIM-RECO/START3X_V26-v1/0016/60DCA072-5452-DF11-A7A5-00248C55CC3C.root'
 
    
    #'rfio:/castor/cern.ch/user/r/rcr/SkimOutput_SingleMuon_1-10.root'
    'file:/afs/cern.ch/user/r/rcr/scratch0/Skim_4.root'
    #'/store/relval/CMSSW_3_5_8/RelValBeamHalo/GEN-SIM-RECO/START3X_V26-v1/0017/D6AA4854-9052-DF11-9317-0030486791BA.root',
    #'/store/relval/CMSSW_3_5_8/RelValBeamHalo/GEN-SIM-RECO/START3X_V26-v1/0016/9E2F2DEB-6352-DF11-BC8D-002618943951.root'
    
#    'rfio:/castor/cern.ch/user/r/rcr/ExpressPhysics/Skims/Run130445/ExpressPhysics__Commissioning10-Express-v3__FEVT____run130445_1.root',
#    'rfio:/castor/cern.ch/user/r/rcr/ExpressPhysics/Skims/Run130445/ExpressPhysics__Commissioning10-Express-v3__FEVT____run130445_2.root',
#    'rfio:/castor/cern.ch/user/r/rcr/ExpressPhysics/Skims/Run130445/ExpressPhysics__Commissioning10-Express-v3__FEVT____run130445_3.root',
#    'rfio:/castor/cern.ch/user/r/rcr/ExpressPhysics/Skims/Run130445/ExpressPhysics__Commissioning10-Express-v3__FEVT____run130445_4.root',
#    'rfio:/castor/cern.ch/user/r/rcr/ExpressPhysics/Skims/Run130445/ExpressPhysics__Commissioning10-Express-v3__FEVT____run130445_5.root',
#    'rfio:/castor/cern.ch/user/r/rcr/ExpressPhysics/Skims/Run130445/ExpressPhysics__Commissioning10-Express-v3__FEVT____run130445_6.root',
#    'rfio:/castor/cern.ch/user/r/rcr/ExpressPhysics/Skims/Run130445/ExpressPhysics__Commissioning10-Express-v3__FEVT____run130445_7.root',
#    'rfio:/castor/cern.ch/user/r/rcr/ExpressPhysics/Skims/Run130445/ExpressPhysics__Commissioning10-Express-v3__FEVT____run130445_8.root',
#    'rfio:/castor/cern.ch/user/r/rcr/ExpressPhysics/Skims/Run130445/ExpressPhysics__Commissioning10-Express-v3__FEVT____run130445_9.root',
#    'rfio:/castor/cern.ch/user/r/rcr/ExpressPhysics/Skims/Run130445/ExpressPhysics__Commissioning10-Express-v3__FEVT____run130445_10.root',
#    'rfio:/castor/cern.ch/user/r/rcr/ExpressPhysics/Skims/Run130445/ExpressPhysics__Commissioning10-Express-v3__FEVT____run130445_11.root'
    
    ),
)

process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')

#process.L1T1=process.hltLevel1GTSeed.clone()
#process.L1T1.L1TechTriggerSeeding = cms.bool(True)
#process.L1T1.L1SeedsLogicalExpression = cms.string('(0 OR 1 OR 2 OR 3 OR 4 OR 5 OR 6 OR 7) AND (36 OR 37 OR 38 OR 39)')
#process.L1T1.L1SeedsLogicalExpression = cms.string('36 OR 37 OR 38 OR 39')
#process.L1T1.L1SeedsLogicalExpression = cms.string('0 OR 1 OR 2 OR 3 OR 4 OR 5 OR 6 OR 7')

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True))


process.p = cms.Path(process.CSCBasedHaloFilter)

process.schedule = cms.Schedule(
    process.p
    )



                                         
                                     
