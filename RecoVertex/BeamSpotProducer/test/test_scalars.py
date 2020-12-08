import FWCore.ParameterSet.Config as cms

process = cms.Process("DumpDTRaw")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
    #'/store/data/BeamCommissioning09/RandomTriggers/RAW/v1/000/123/576/4AD8F322-B4E1-DE11-BFF2-0030487A322E.root'
    
#    '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/606/E69C1903-B740-DF11-BC4E-0030487CD178.root',
#    '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/606/D6C81B03-B740-DF11-B467-0030487CD76A.root',
#    '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/646/E6614FE2-1D41-DF11-BEE0-000423D98EA8.root',
    #'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/646/DC9311B9-2041-DF11-ABB4-0030487CD840.root',
    #'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/646/C8583EB8-2041-DF11-9029-000423D98EA8.root',
    #'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/646/9EC8C779-2141-DF11-AE46-001D09F24303.root',
    #'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/646/78153729-2241-DF11-AF93-001D09F248F8.root',
    #'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/646/6AFBB204-2041-DF11-A936-001D09F28EA3.root',
    #'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/646/543F6704-2041-DF11-9CE8-001D09F24934.root',
    #'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/646/506F022F-2941-DF11-ABDD-000423D99AAE.root',
    #'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/646/4E397E08-2041-DF11-B0CD-001D09F2447F.root',
    #'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/646/42E3717A-2141-DF11-973E-001D09F24DA8.root',
    #'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/646/32375894-1E41-DF11-9413-000423D99896.root',
    #'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/646/20EDEDE7-1D41-DF11-B358-000423D9890C.root',
    #'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/646/0A5C557E-2141-DF11-9861-00151796CD80.root',
    #'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/646/0046AE7C-2141-DF11-9784-0030487C8CB6.root'

#	'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/659/FEC5118A-7D41-DF11-8A3F-001D09F2305C.root',
#	'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/659/EE97865A-8541-DF11-BE6D-001D09F28F0C.root',
#	'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/659/C06A61E8-7741-DF11-8298-0019B9F581C9.root',
#	'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/659/A40E9BA3-7841-DF11-B367-000423D990CC.root',
#	'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/659/96AD316B-7A41-DF11-BC54-001D09F276CF.root',
#	'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/659/8C49426A-7941-DF11-A11A-0030487C90EE.root',
#	'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/659/86961539-7741-DF11-A191-001D09F29619.root',
#	'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/659/3C747220-7C41-DF11-A1C1-001617E30E2C.root',
#	'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/659/06005386-7B41-DF11-BEA0-001D09F23A84.root',
	
#	'/store/data/Commissioning10/MinimumBias/RAW/v4/000/133/321/62B90803-1349-DF11-A72F-001D09F24259.root',
#	'/store/data/Commissioning10/MinimumBias/RAW/v4/000/133/321/58384B68-1449-DF11-A86A-001D09F2462D.root',
#	'/store/data/Commissioning10/MinimumBias/RAW/v4/000/133/321/542018B8-1349-DF11-B470-001D09F23D1D.root',
#	'/store/data/Commissioning10/MinimumBias/RAW/v4/000/133/321/3E4C594F-2049-DF11-B00A-001617E30E28.root',
#	'/store/data/Commissioning10/MinimumBias/RAW/v4/000/133/321/3CBA15DC-1549-DF11-89A8-001D09F24F65.root',
#	'/store/data/Commissioning10/MinimumBias/RAW/v4/000/133/321/3CA30CAA-1249-DF11-BD4D-0030487C6A66.root',

    '/store/data/Commissioning10/MinimumBias/RAW/v4/000/133/239/A6F9EAC4-5448-DF11-8CBB-0030487C90D4.root',
    '/store/data/Commissioning10/MinimumBias/RAW/v4/000/133/239/92103C17-5448-DF11-AB9F-000423D99614.root'
    
	
    #'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/660/FE8930BC-8141-DF11-87E1-00304879FA4A.root',
    #'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/661/E049CAC1-8141-DF11-B252-000423D944F0.root',
    #'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/661/C68552CC-8141-DF11-88A5-001D09F295A1.root',
    #'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/661/BA987D67-8241-DF11-95BA-000423D98834.root',
    #'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/661/BA7A7268-8241-DF11-AC43-001D09F244DE.root',
    #'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/661/B8B9F029-8341-DF11-9B1D-000423D6B358.root',
    #'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/661/B49AD6E8-8341-DF11-A31E-001D09F29849.root',
    #'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/661/B4635158-8041-DF11-879E-003048D3750A.root',
    #'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/661/88F11FE8-8341-DF11-9650-0019B9F70468.root',
    #'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/661/523338C4-8841-DF11-9F23-000423D6CA6E.root',
    #'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/661/2812F357-8041-DF11-9C37-000423D6CA72.root',
    #'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/661/1A6FC029-8341-DF11-8D57-000423D6C8EE.root',
    #'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/661/164A3C69-8241-DF11-863C-001D09F244BB.root',
    #'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/661/00DBB206-8041-DF11-8C5B-0019DB29C5FC.root'
    
    ),
                            skipEvents = cms.untracked.uint32(0) )

# process.source = cms.Source("NewEventStreamFileReader",
#                             fileNames = cms.untracked.vstring(
#     'file:/directory/pippo.dat'
#     ))

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(-1)
)        

process.load("CondCore.DBCommon.CondDBSetup_cfi")
 
process.BeamSpotDBSource = cms.ESSource("PoolDBESSource",
                                        process.CondDBSetup,
                                        toGet = cms.VPSet(cms.PSet(
    record = cms.string('BeamSpotObjectsRcd'),
    tag = cms.string('BeamSpotObjects_2009_v11_offline'))),
                                        #connect = cms.string('sqlite_file:EarlyCollision.db')
connect = cms.string('frontier://FrontierProd/CMS_COND_31X_BEAMSPOT')
)
process.es_prefer_beamspot = cms.ESPrefer("PoolDBESSource","BeamSpotDBSource")


process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    )
)

process.scalersRawToDigi = cms.EDProducer('ScalersRawToDigi')

process.load("RecoVertex.BeamSpotProducer.BeamSpotOnline_cff")

process.out = cms.OutputModule( "PoolOutputModule",
                                fileName = cms.untracked.string( '/uscmst1b_scratch/lpc1/cmsroc/yumiceva/tmp/onlineBeamSpotwithDB3.root' ),
                                outputCommands = cms.untracked.vstring(
    "keep *"
    )
                                )

process.dtRawDump = cms.Path(process.scalersRawToDigi + process.onlineBeamSpot )
#process.dtRawDump = cms.Path( process.onlineBeamSpot )

process.e = cms.EndPath( process.out )
