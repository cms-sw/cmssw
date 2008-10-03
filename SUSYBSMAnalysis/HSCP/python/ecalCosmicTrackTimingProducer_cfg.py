import FWCore.ParameterSet.Config as cms

process = cms.Process("EcalCosmicTrackTimingProducer")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("Geometry.EcalCommonData.EcalOnly_cfi")
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")
process.load("RecoEcal.EgammaClusterProducers.geometryForClustering_cff")
process.load("RecoEcal.EgammaClusterProducers.cosmicClusteringSequence_cff")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.CommonDetUnit.globalTrackingGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")
process.load("TrackingTools.TrackAssociator.default_cfi")


import RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi
process.ecalUncalibHit = RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi.ecalFixedAlphaBetaFitUncalibRecHit.clone()
process.load("RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi")

process.load("CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi")
process.load("HLTrigger.special.TriggerTypeFilter_cfi")
process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff")

process.load("Producers.EcalCosmicTrackTimingProducer.ecalCosmicTrackTimingProducer_cfi")
process.dumpEv = cms.EDAnalyzer("EventContentAnalyzer")

process.MessageLogger = cms.Service("MessageLogger",
    #suppressInfo = cms.untracked.vstring('ecalEBunpacker'),
    cout = cms.untracked.PSet(
      threshold = cms.untracked.string('WARNING')
      ),
    categories = cms.untracked.vstring('EcalCosmicTrackTimingProducer'),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    fileNames = cms.untracked.vstring(#'/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/289/1E1407F1-106D-DD11-97A7-000423D985E4.root'
        #'/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/058/359/005A40D9-1470-DD11-A2B6-001617C3B6DE.root')
        #'/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/771/00D18762-386E-DD11-A081-0016177CA7A0.root')
        #'/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/553/FC7FC218-896D-DD11-BC54-001617E30CD4.root')
         '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V2P_CRUZET4_InterimReco_v3/0003/04CA6441-E36E-DD11-9CFF-000423D9997E.root')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.outFile = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('ecalCosmicTrackTimingProducerTest.root'),
    outputCommands = cms.untracked.vstring('drop *',
           'keep *_ecalCosmicTrackTimingProducer_*_*',
           'keep *_cosmicMuons_*_*')
)

import CalibTracker.Configuration.Common.PoolDBESSource_cfi
process.siStripPedestalFrontier = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
process.siStripPedestalFrontier.connect = 'frontier://PromptProd/CMS_COND_21X_STRIP'
process.siStripPedestalFrontier.toGet = cms.VPSet(cms.PSet(
            record = cms.string('SiStripPedestalsRcd'),
                        tag = cms.string('SiStripPedestals_TKCC_21X_v3_hlt')
                    ))
process.siStripPedestalFrontier.BlobStreamerName = 'TBufferBlobStreamingService'
process.es_prefer_SiStripFake = cms.ESPrefer("PoolDBESSource","siStripPedestalFrontier")

#process.p = cms.Path(process.triggerTypeFilter*process.ecalEBunpacker*process.ecalUncalibHit*process.ecalRecHit*process.cosmicClusteringSequence*process.ecalCosmicTrackTimingProducer*process.dumpEv)
process.p = cms.Path(process.ecalCosmicTrackTimingProducer)
process.end = cms.EndPath(process.outFile)

process.GlobalTag.globaltag = 'CRUZET4_V5P::All'
process.ecalUncalibHit.EBdigiCollection = 'ecalEBunpacker:ebDigis'
process.ecalUncalibHit.EEdigiCollection = 'ecalEBunpacker:eeDigis'
process.ecalRecHit.ChannelStatusToBeExcluded = [1]
process.ecalRecHit.EBuncalibRecHitCollection = 'ecalUncalibHit:EcalUncalibRecHitsEB'
process.ecalRecHit.EEuncalibRecHitCollection = 'ecalUncalibHit:EcalUncalibRecHitsEE'
process.EcalTrivialConditionRetriever.producedEcalWeights = False
process.EcalTrivialConditionRetriever.producedEcalPedestals = False
process.EcalTrivialConditionRetriever.producedEcalIntercalibConstants = False
process.EcalTrivialConditionRetriever.producedEcalIntercalibErrors = False
process.EcalTrivialConditionRetriever.producedEcalGainRatios = False
process.EcalTrivialConditionRetriever.producedEcalADCToGeVConstant = False
process.EcalTrivialConditionRetriever.producedEcalLaserCorrection = False
process.EcalTrivialConditionRetriever.producedChannelStatus = False
process.EcalTrivialConditionRetriever.producedChannelStatus = True
process.EcalTrivialConditionRetriever.channelStatusFile = 'CaloOnlineTools/EcalTools/data/listCRUZET4.v5.hashed.txt'
process.es_prefer_EcalTrivialConditionRetriever = cms.ESPrefer("EcalTrivialConditionRetriever")
process.triggerTypeFilter.SelectedTriggerType = 1
process.cosmicBasicClusters.barrelUnHitProducer = "ecalUncalibHit"
process.cosmicBasicClusters.endcapUnHitProducer = "ecalUncalibHit"
