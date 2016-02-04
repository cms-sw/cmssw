# test file for PFCandidate validation
# performs a matching with the genParticles collection. 
# creates a root file with histograms filled with PFCandidate data,
# present in the Candidate, and in the PFCandidate classes, for matched
# PFCandidates. Matching histograms (delta pt etc) are also available. 

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

fa = 'RelValQCD'
fb = 'FlatPt_15_3000_Fast'
fc = 'ParticleFlow'

#process.load("RecoParticleFlow.Configuration.DBS_Samples.%s_%s_cfi" % (fa, fb) )
process.source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring(
#'/../user/l/lacroix/MET_Validation/ttbar_fastsim_310_pre6_muonAndJEC/aod.root'
#'/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/133/532/EC93873A-D74B-DF11-A1B9-00E08179185D.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/133/531/D6E1CE68-ED4B-DF11-A676-003048D45F84.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/133/529/223C34BD-EC4B-DF11-9CA6-003048D476D4.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/133/526/B28AEED6-E94B-DF11-9124-00E08178C103.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/133/521/C0EFDC20-024C-DF11-A82A-00E08178C155.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/133/518/C620697E-B94B-DF11-A125-003048D46090.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/133/516/CA61DFDB-E14B-DF11-9193-003048D476B0.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0128/FEBBC289-385D-DF11-BBB2-001A928116BE.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0128/FCDC4C82-375D-DF11-8DEE-0026189438DE.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0128/F8DEAB3E-375D-DF11-B26C-0018F3D0967A.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0128/F654FD78-375D-DF11-AFCA-00261894388F.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0128/F471BC3A-375D-DF11-BD59-00261894382A.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0128/F4003186-385D-DF11-9189-002618FDA262.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0128/F243CB56-375D-DF11-8D2A-0018F3D09696.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0128/F202B148-375D-DF11-9AE7-001A928116D2.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0128/EE2122C1-375D-DF11-B479-00261894388F.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0128/ECF5B06F-375D-DF11-99F4-00261894382A.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0128/EC243E26-385D-DF11-BE91-0026189438F5.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0128/E4254CCF-375D-DF11-BC15-001A928116E2.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0128/E0921F83-385D-DF11-B9B3-001A928116FA.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0128/E0464F3B-385D-DF11-968D-0026189438FD.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0128/DEDCAF6E-385D-DF11-8D0B-001A928116BE.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0128/DE1BC96C-375D-DF11-A5B7-0018F3D0962E.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0128/D6176E33-375D-DF11-BF1A-001BFCDBD154.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0128/D4AA7035-385D-DF11-951E-001A92811738.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0128/D4A47284-385D-DF11-970E-001BFCDBD1B6.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0127/FC971980-345D-DF11-A4AF-0018F3D095EC.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0121/0C6B9744-9C5C-DF11-AF67-001A928116D8.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0007/8084C57C-805C-DF11-ACCE-0018F3D09676.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0007/20B9C08A-7F5C-DF11-821F-0018F3D096CE.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0007/1EF7BF2D-DD5C-DF11-AEEC-001A92971BB2.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0006/4A197A41-A85C-DF11-BB68-001A92971B84.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0006/08DF1E48-AB5C-DF11-A2BF-0018F3D09650.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0005/FE8B9C92-9A5C-DF11-84C6-002618943910.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0005/E25A4526-C95C-DF11-8239-0018F3D096A0.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0005/C6BB8D2C-DB5C-DF11-BC01-0018F3D096DE.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0005/30598A24-A35C-DF11-BC15-001A92810AA8.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0005/026A0A90-7F5C-DF11-B1A3-001A92811706.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0004/F255215F-D65C-DF11-B31C-001A928116EA.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0004/F04FE91D-D05C-DF11-9CF5-0018F3D09710.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0004/CA954359-9D5C-DF11-9EFE-002618943981.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0004/AE1C0BF9-CA5C-DF11-A4BB-003048678BB2.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0004/4E20E1DD-945C-DF11-B962-001A92971ADC.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/May6thPDSkim_GOODCOLL-v1/0004/04AA2FF2-A15C-DF11-8972-0026189437F0.root',
)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load("Validation.RecoParticleFlow.metBenchmark_cff")
process.pfMetBenchmark.mode = 1
process.caloMetBenchmark.mode = 1
process.UncorrCaloMetBenchmark.mode = 1

process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskAlgoTrigConfig_cff')
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')

process.L1T1coll=process.hltLevel1GTSeed.clone()
process.L1T1coll.L1TechTriggerSeeding = cms.bool(True)
process.L1T1coll.L1SeedsLogicalExpression = cms.string('0 AND (40 OR 41) AND NOT (36 OR 37 OR 38 OR 39) AND NOT ((42 AND NOT 43) OR (43 AND NOT 42))')

process.primaryVertexFilter = cms.EDFilter("VertexSelector",
   src = cms.InputTag("offlinePrimaryVertices"),
#   cut = cms.string("!isFake && ndof > 4 && abs(z) <= 15 && position.Rho <= 2"), # tracksSize() > 3 for the older cut
   cut = cms.string("!isFake && ndof > 4 && abs(z) <= 15"), # tracksSize() > 3 for the older cut
   filter = cms.bool(True),   # otherwise it won't filter the events, just produce an empty vertex collection.
)


process.noscraping = cms.EDFilter("FilterOutScraping",
applyfilter = cms.untracked.bool(True),
debugOn = cms.untracked.bool(False),
numtrack = cms.untracked.uint32(10),
thresh = cms.untracked.double(0.25)
)


process.dqmSaver.convention = 'Offline'
#process.dqmSaver.workflow = '/%s/%s/%s' % (fa, fb, fc)
process.dqmSaver.workflow = '/A/B/C'
process.dqmEnv.subSystemFolder = 'ParticleFlow'

process.p =cms.Path(
    process.dqmEnv +
    process.L1T1coll+process.primaryVertexFilter+process.noscraping+
    process.metBenchmarkSequenceData +
    process.dqmSaver
    )


process.schedule = cms.Schedule(process.p)


process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100000
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
