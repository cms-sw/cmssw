import FWCore.ParameterSet.Config as cms

process = cms.Process("d0phi")
# initialize MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("RecoVertex.BeamSpotProducer.d0_phi_analyzer_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/906/F253BBE8-4FE5-DE11-866A-0030487A1990.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/906/F0FACD7D-4EE5-DE11-AE94-000423D99394.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/906/F05A2E81-4EE5-DE11-8BF1-000423D94A04.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/906/F054FB7D-4EE5-DE11-A81D-000423D99A8E.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/906/D8CDB1E7-4FE5-DE11-A911-001617C3B6CE.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/906/9088EACA-4DE5-DE11-86AA-000423D9989E.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/906/8A93D9F2-4FE5-DE11-A65C-0030487A18A4.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/906/84FD58F1-4FE5-DE11-A844-003048D2C108.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/906/822EF6CC-4DE5-DE11-9FAD-001D09F23944.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/906/5EDBBE33-4FE5-DE11-8B3A-0030487A1990.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/906/5ECD7759-51E5-DE11-AC54-001D09F2532F.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/906/5A3F66C9-4DE5-DE11-81EC-001617DC1F70.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/906/58DD96E7-4EE5-DE11-9B5C-000423D9863C.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/906/22C04FE6-4EE5-DE11-A174-000423D992A4.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/906/124788C9-4DE5-DE11-9F87-000423D98BE8.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/906/044A4BCB-4DE5-DE11-88CD-000423D951D4.root'

#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/909/F0ADCD5C-67E5-DE11-BE4E-001D09F2441B.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/909/E2AB99C9-66E5-DE11-883B-001D09F244BB.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/909/DA9F87C7-66E5-DE11-8B1C-001D09F244DE.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/909/CA6260BC-68E5-DE11-B43E-001D09F25438.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/909/C8F657C6-66E5-DE11-AA76-001D09F24D67.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/909/B852CC68-68E5-DE11-B2D2-001D09F2527B.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/909/AA6CAA2E-63E5-DE11-BDD2-000423D6B444.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/909/A0618E2D-63E5-DE11-BF43-000423D99AAE.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/909/7637827D-64E5-DE11-BBFE-001D09F2B30B.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/909/74AD86C6-66E5-DE11-B692-001D09F2424A.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/909/720D5D3A-64E5-DE11-8AA1-0019B9F707D8.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/909/5CDDD336-64E5-DE11-A595-001D09F28F25.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/909/5AE8CD5D-67E5-DE11-9260-001D09F23D1D.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/909/528BFF34-64E5-DE11-A463-001D09F2A690.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/909/4C08F3C7-66E5-DE11-9FBC-0019B9F72CE5.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/909/3C3C9846-65E5-DE11-9623-001D09F251CC.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/909/348F5FC9-66E5-DE11-8B6E-001D09F24DDF.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/909/308B3667-68E5-DE11-AE18-001D09F2441B.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/909/2C287C61-67E5-DE11-85B8-001D09F28E80.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/909/0A7B2C46-65E5-DE11-B99E-001D09F2A49C.root'

# dbs --search --query "find file where dataset = */FEVT and run=123818 and lumi>2 and lumi< 48 " --url=http://cmsdbsprod.cern.ch/cms_dbs_caf_analysis_01/servlet/DBSServlet


'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/BAEF02C0-0BED-DE11-9EBA-00261894392F.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/A461CC43-03ED-DE11-8E44-00304867BFF2.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/8A29D1B9-07ED-DE11-BDFD-002618943843.root'
#'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/26BC3350-03ED-DE11-9683-002618943885.root',
#'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/24E9A529-14ED-DE11-99C2-00304867C034.root',
#'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/24597050-03ED-DE11-A701-00261894389D.root',
#'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0001/F8B5CB1B-01ED-DE11-8529-003048678FAE.root'

	#'file:FirstEvent.root'
#'/store/mc/Summer09/MinBias/ALCARECO/StreamTkAlMinBias-334_DESIGN_3X_V8A_v1/0082/22F2A8A8-8BD8-DE11-A2FE-00248C0BE01E.root',
#'/store/mc/Summer09/MinBias/ALCARECO/StreamTkAlMinBias-334_DESIGN_3X_V8A_v1/0082/3CF7DD76-8CD8-DE11-9C9A-0026189438D5.root',
#'/store/mc/Summer09/MinBias/ALCARECO/StreamTkAlMinBias-334_DESIGN_3X_V8A_v1/0082/66273AA8-8BD8-DE11-A9FA-0026189438BC.root',
#'/store/mc/Summer09/MinBias/ALCARECO/StreamTkAlMinBias-334_DESIGN_3X_V8A_v1/0082/988EDEDB-8DD8-DE11-8E82-00261894386D.root'

#'/store/mc/Summer09/MinBias/ALCARECO/StreamTkAlMinBias-334_STARTUP3X_V8D_v1/0082/16260D10-89D8-DE11-9578-0026189437E8.root',
#'/store/mc/Summer09/MinBias/ALCARECO/StreamTkAlMinBias-334_STARTUP3X_V8D_v1/0082/18DD5AEF-89D8-DE11-88AF-002618943956.root',
#'/store/mc/Summer09/MinBias/ALCARECO/StreamTkAlMinBias-334_STARTUP3X_V8D_v1/0082/488187EE-89D8-DE11-957D-00248C0BE01E.root',
#'/store/mc/Summer09/MinBias/ALCARECO/StreamTkAlMinBias-334_STARTUP3X_V8D_v1/0082/5E1915A7-8BD8-DE11-8C3D-002618943956.root'
#	'file:BSCskim_123151_Express.root'
#    'file:Run122314_BSCSkim_MinBiasPD_ReTracking.root'
#    'rfio:/castor/cern.ch/user/c/chiochia/09_beam_commissioning/BSCskim_123592_Express_bit40-41.root'
#    'rfio:/castor/cern.ch/user/g/gpetrucc/900GeV/DATA/bit40-run123596-lumi_68_129.root',
#    'rfio:/castor/cern.ch/user/g/gpetrucc/900GeV/DATA/bit40-run123596-lumi130_143.root'
#    'rfio:/castor/cern.ch/user/c/chiochia/09_beam_commissioning/BSCskim_123615_Express_bit40-41_LS72-88.root'
    )
)

process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('124120:1-124120:59')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1) #1500
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

# this is for filtering on L1 technical trigger bit
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('0 AND ( 40 OR 41 )')

#### remove beam scraping events
process.noScraping= cms.EDFilter("FilterOutScraping",
    applyfilter = cms.untracked.bool(True),
    debugOn = cms.untracked.bool(False), ## Or 'True' to get some per-event info
    numtrack = cms.untracked.uint32(10),
    thresh = cms.untracked.double(0.20)
)

process.p = cms.Path(process.hltLevel1GTSeed + process.d0_phi_analyzer)
process.MessageLogger.debugModules = ['BeamSpotAnalyzer']

#######################



# run over STA muons
#process.d0_phi_analyzer.BeamFitter.TrackCollection = cms.untracked.InputTag('ALCARECOTkAlMinBias') #,'UpdatedAtVtx')
#process.d0_phi_analyzer.BeamFitter.IsMuonCollection = True

process.d0_phi_analyzer.BeamFitter.MinimumTotalLayers = 6
process.d0_phi_analyzer.BeamFitter.MinimumPixelLayers = -1
process.d0_phi_analyzer.BeamFitter.MaximumNormChi2 = 10
process.d0_phi_analyzer.BeamFitter.MinimumInputTracks = 2
process.d0_phi_analyzer.BeamFitter.MinimumPt = 1.0
process.d0_phi_analyzer.BeamFitter.MaximumImpactParameter = 1.0 #5.0
process.d0_phi_analyzer.BeamFitter.TrackAlgorithm =  cms.untracked.vstring()
#process.d0_phi_analyzer.BeamFitter.TrackQuality = cms.untracked.vstring("highPurity")
#process.d0_phi_analyzer.BeamFitter.InputBeamWidth = 0.0400
process.d0_phi_analyzer.BeamFitter.InputBeamWidth = -1
process.d0_phi_analyzer.BeamFitter.Debug = True
#########################
process.d0_phi_analyzer.BeamFitter.AsciiFileName = 'BeamFit_124120_vpv4.txt'
process.d0_phi_analyzer.BeamFitter.OutputFileName = 'run124120_all_vpv3.root' #AtVtx10000.root'
process.d0_phi_analyzer.BeamFitter.SaveNtuple = True

#process.d0_phi_analyzer.PVFitter.Apply3DFit = True

# fit as function of lumi sections
process.d0_phi_analyzer.BSAnalyzerParameters.fitEveryNLumi = 2
process.d0_phi_analyzer.BSAnalyzerParameters.resetEveryNLumi = 2
