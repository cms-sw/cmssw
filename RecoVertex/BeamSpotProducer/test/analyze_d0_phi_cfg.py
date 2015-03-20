import FWCore.ParameterSet.Config as cms

process = cms.Process("d0phi")
# initialize MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("RecoVertex.BeamSpotProducer.d0_phi_analyzer_cff")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(

#"/store/data/Run2012D/MinimumBias/RECO/PromptReco-v1/000/203/991/E0CA3ABC-4D0E-E211-A047-E0CB4E4408C4.root",							  	  
#"/store/data/Run2012D/MinimumBias/RECO/PromptReco-v1/000/203/992/B2758E3E-5A0E-E211-92DE-003048F117F6.root"							  	  

"/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V1-v3/00000/067739D0-AFAB-E411-AC03-0025905A48D0.root",					 
"/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V1-v3/00000/12718F94-80AB-E411-96D7-0025905B8590.root",					 
"/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V1-v3/00000/56D1C481-82AB-E411-BEFB-0025905A611E.root",					 
"/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V1-v3/00000/5A48E246-7AAB-E411-B2C6-003048FFCC18.root",					 
"/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V1-v3/00000/682B4D67-85AB-E411-888C-0025905A48BB.root",					 
"/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V1-v3/00000/8446C23E-7AAB-E411-8274-0025905B8576.root",					 
"/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V1-v3/00000/8E4FA95F-7DAB-E411-83A0-0025905A60EE.root",					 
"/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V1-v3/00000/D2B87CCF-AFAB-E411-B2AF-0025905A48BC.root",					 
"/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V1-v3/00000/EA7B6485-87AB-E411-B2F8-0025905A612E.root",					 
"/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V1-v3/00000/F62C4E70-8CAB-E411-9DBE-0025905B8562.root",					 


# dbs --search --query "find file where dataset = */FEVT and run=123818 and lumi>2 and lumi< 48 " --url=http://cmsdbsprod.cern.ch/cms_dbs_caf_analysis_01/servlet/DBSServlet
																				  	  

#'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/BAEF02C0-0BED-DE11-9EBA-00261894392F.root',					  	  
#'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/A461CC43-03ED-DE11-8E44-00304867BFF2.root',					  	  
#'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/8A29D1B9-07ED-DE11-BDFD-002618943843.root' 					  	  
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

#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('124120:1-124120:59')

process.maxEvents = cms.untracked.PSet(
                                       input = cms.untracked.int32(-1) #1500
                                      )

process.options   = cms.untracked.PSet(
                                       wantSummary = cms.untracked.bool(False)
                                      )

# this is for filtering on L1 technical trigger bit
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')

process.hltLevel1GTSeed.L1TechTriggerSeeding     = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('0 AND ( 40 OR 41 )')

#### remove beam scraping events
process.noScraping = cms.EDFilter(
                                  "FilterOutScraping",
    				  applyfilter = cms.untracked.bool(True) ,
    				  debugOn     = cms.untracked.bool(False), ## Or 'True' to get some per-event info
    				  numtrack    = cms.untracked.uint32(10) ,
    				  thresh      = cms.untracked.double(0.20)
                                 )

#process.p = cms.Path(process.hltLevel1GTSeed + process.d0_phi_analyzer)
process.p = cms.Path(process.d0_phi_analyzer)
process.MessageLogger.debugModules = ['BeamSpotAnalyzer']

#######################

process.MessageLogger.cerr.FwkReport.reportEvery = 1000

# run over STA muons
#process.d0_phi_analyzer.BeamFitter.TrackCollection = cms.untracked.InputTag('ALCARECOTkAlMinBias') #,'UpdatedAtVtx')
#process.d0_phi_analyzer.BeamFitter.IsMuonCollection = True

process.d0_phi_analyzer.BeamFitter.MinimumTotalLayers 	     = 6
process.d0_phi_analyzer.BeamFitter.MinimumPixelLayers 	     = -1
process.d0_phi_analyzer.BeamFitter.MaximumNormChi2    	     = 10
process.d0_phi_analyzer.BeamFitter.MinimumInputTracks 	     = 2
process.d0_phi_analyzer.BeamFitter.MinimumPt                 = 1.0
process.d0_phi_analyzer.BeamFitter.MaximumImpactParameter    = 1.0 #5.0
process.d0_phi_analyzer.BeamFitter.TrackAlgorithm            =  cms.untracked.vstring()
#process.d0_phi_analyzer.BeamFitter.TrackQuality             = cms.untracked.vstring("highPurity")
#process.d0_phi_analyzer.BeamFitter.InputBeamWidth           = 0.0400
process.d0_phi_analyzer.BeamFitter.InputBeamWidth            = -1
process.d0_phi_analyzer.BeamFitter.Debug                     = True
process.d0_phi_analyzer.BeamFitter.AsciiFileName  	     = 'BeamFit_124120_vpv4.txt'
process.d0_phi_analyzer.BeamFitter.OutputFileName 	     = 'run124120_all_vpv3.root' #AtVtx10000.root'
process.d0_phi_analyzer.BeamFitter.SaveNtuple     	     = True
process.d0_phi_analyzer.BeamFitter.SavePVVertices 	     = True
process.d0_phi_analyzer.BeamFitter.SaveFitResults 	     = True

process.d0_phi_analyzer.PVFitter.Apply3DFit       	     = True

# fit as function of lumi sections
process.d0_phi_analyzer.BSAnalyzerParameters.fitEveryNLumi   = 2
process.d0_phi_analyzer.BSAnalyzerParameters.resetEveryNLumi = 2
