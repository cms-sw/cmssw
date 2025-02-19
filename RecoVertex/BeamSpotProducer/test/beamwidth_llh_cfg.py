import FWCore.ParameterSet.Config as cms

process = cms.Process("d0phi")
# initialize MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("RecoVertex.BeamSpotProducer.d0_phi_analyzer_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(


# MC: New 336_patch3 samples where new tag goes(change the accordingly)
'/store/relval/Summer09/MinBias/GEN-SIM-RECO/STARTUP3X_V8K_900GeV_preproduction_336patch3_withFixes-v1/0007/764C6A63-D905-DF11-9636-0030487CD6DA.root',
'/store/relval/Summer09/MinBias/GEN-SIM-RECO/STARTUP3X_V8K_900GeV_preproduction_336patch3_withFixes-v1/0007/66A9FF26-CF05-DF11-A618-0030487CD6DA.root'


#Data 2009
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


# dbs --search --query "find file where dataset = */FEVT and run=123818 and lumi>2 and lumi< 48 " --url=http://cmsdbsprod.cern.ch/cms_dbs_caf_analysis_01/servlet/DBSServlet

    )
)


#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('123909:16-123909:29')

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

#for 2009 data
#process.p = cms.Path(process.hltLevel1GTSeed + process.d0_phi_analyzer)

#for MC
process.p = cms.Path(process.d0_phi_analyzer)

process.MessageLogger.debugModules = ['BeamSpotAnalyzer']

#######################

# run over STA muons
#process.d0_phi_analyzer.BeamFitter.TrackCollection = cms.untracked.InputTag('ALCARECOTkAlMinBias') #,'UpdatedAtVtx')
#process.d0_phi_analyzer.BeamFitter.IsMuonCollection = True

process.d0_phi_analyzer.BSAnalyzerParameters.RunBeamWidthFit = True

process.d0_phi_analyzer.BeamFitter.MinimumTotalLayers = 11
process.d0_phi_analyzer.BeamFitter.MinimumPixelLayers = 3
process.d0_phi_analyzer.BeamFitter.MaximumNormChi2 = 2
process.d0_phi_analyzer.BeamFitter.MinimumInputTracks = 100
process.d0_phi_analyzer.BeamFitter.MinimumPt = 2.0  #  use 2GeV for 900, 2.36, ~5GeV for 7 TeV, ~10GeV for 10 TeV
process.d0_phi_analyzer.BeamFitter.MaximumImpactParameter = 5.0
#process.d0_phi_analyzer.BeamFitter.TrackAlgorithm =  cms.untracked.vstring()
#process.d0_phi_analyzer.BeamFitter.TrackQuality = cms.untracked.vstring("highPurity")
#process.d0_phi_analyzer.BeamFitter.InputBeamWidth = 0.0400
process.d0_phi_analyzer.BeamFitter.Debug = False 
#########################
process.d0_phi_analyzer.BeamFitter.AsciiFileName = 'BeamFit_test.txt'
process.d0_phi_analyzer.BeamFitter.OutputFileName = 'BeamFit_test.root' #AtVtx10000.root'
process.d0_phi_analyzer.BeamFitter.SaveNtuple = False

# fit as function of lumi sections
#process.d0_phi_analyzer.BSAnalyzerParameters.fitEveryNLumi = 2
#process.d0_phi_analyzer.BSAnalyzerParameters.resetEveryNLumi = 2
