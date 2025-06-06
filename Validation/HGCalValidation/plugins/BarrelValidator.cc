#include <memory>

#include "Validation/HGCalValidation/interface/BarrelValidator.h"

#include "SimCalorimetry/HGCalAssociatorProducers/interface/AssociatorTools.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

using namespace std;
using namespace edm;
using namespace ticl;

bool assignTracksterMaps(const edm::Handle<std::vector<ticl::Trackster>>& tracksterHandle,
                         const edm::Handle<std::vector<ticl::Trackster>>& simTracksterHandle,
                         const edm::Handle<std::vector<ticl::Trackster>>& simTracksterFromCPHandle,
                         const std::vector<edm::Handle<TracksterToTracksterMap>>& tracksterToTracksterMapsHandles,
                         edm::Handle<TracksterToTracksterMap>& trackstersToSimTrackstersMap,
                         edm::Handle<TracksterToTracksterMap>& simTrackstersToTrackstersMap,
                         edm::Handle<TracksterToTracksterMap>& trackstersToSimTrackstersFromCPsMap,
                         edm::Handle<TracksterToTracksterMap>& simTrackstersFromCPsToTrackstersMap) {
  const auto recoTrackstersProductId = tracksterHandle.id();
  const auto simTrackstersProductId = simTracksterHandle.id();
  const auto simTrackstersFromCPsProductId = simTracksterFromCPHandle.id();
  if (recoTrackstersProductId == simTrackstersProductId or recoTrackstersProductId == simTrackstersFromCPsProductId) {
    edm::LogInfo("MissingProduct") << "no SimTrackster to Simtrackster map available.";
    return false;
  }
  for (const auto& handle : tracksterToTracksterMapsHandles) {
    const auto& firstID = handle->getCollectionIDs().first.id();
    const auto& secondID = handle->getCollectionIDs().second.id();

    if (firstID == recoTrackstersProductId && secondID == simTrackstersProductId) {
      trackstersToSimTrackstersMap = handle;
    } else if (firstID == simTrackstersProductId && secondID == recoTrackstersProductId) {
      simTrackstersToTrackstersMap = handle;
    } else if (firstID == recoTrackstersProductId && secondID == simTrackstersFromCPsProductId) {
      trackstersToSimTrackstersFromCPsMap = handle;
    } else if (firstID == simTrackstersFromCPsProductId && secondID == recoTrackstersProductId) {
      simTrackstersFromCPsToTrackstersMap = handle;
    }
  }
  if (not trackstersToSimTrackstersMap.isValid()) {
    edm::LogError("MissingProduct") << "trackstersToSimTrackstersMap is not valid";
    return false;
  }
  if (not simTrackstersToTrackstersMap.isValid()) {
    edm::LogError("MissingProduct") << "simTrackstersToTrackstersMap is not valid";
    return false;
  }
  if (not trackstersToSimTrackstersFromCPsMap.isValid()) {
    edm::LogError("MissingProduct") << "trackstersToSimTrackstersFromCPsMap is not valid";
    return false;
  }
  if (not simTrackstersFromCPsToTrackstersMap.isValid()) {
    edm::LogError("MissingProduct") << "simTrackstersFromCPsToTrackstersMap is not valid";
    return false;
  }
  return true;
}

BarrelValidator::BarrelValidator(const edm::ParameterSet& pset)
    : caloGeomToken_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
      label_lcl(pset.getParameter<edm::InputTag>("label_lcl")),
      associator_(pset.getUntrackedParameter<std::vector<edm::InputTag>>("associator")),
      associatorSim_(pset.getUntrackedParameter<std::vector<edm::InputTag>>("associatorSim")),
      SaveGeneralInfo_(pset.getUntrackedParameter<bool>("SaveGeneralInfo")),
      doCaloParticlePlots_(pset.getUntrackedParameter<bool>("doCaloParticlePlots")),
      doCaloParticleSelection_(pset.getUntrackedParameter<bool>("doCaloParticleSelection")),
      doSimClustersPlots_(pset.getUntrackedParameter<bool>("doSimClustersPlots")),
      label_SimClustersPlots_(pset.getParameter<std::string>("label_SimClusters")),
      label_SimClustersLevel_(pset.getParameter<std::string>("label_SimClustersLevel")),
      doLayerClustersPlots_(pset.getUntrackedParameter<bool>("doLayerClustersPlots")),
      label_layerClustersPlots_(pset.getParameter<std::string>("label_layerClustersPlots")),
      label_LCToCPLinking_(pset.getParameter<std::string>("label_LCToCPLinking")),
      hitsToken_(consumes<reco::MultiPFRecHitCollection>(pset.getParameter<edm::InputTag>("hits"))),
      scToCpMapToken_(
          consumes<SimClusterToCaloParticleMap>(pset.getParameter<edm::InputTag>("simClustersToCaloParticlesMap"))) {
  //In this way we can easily generalize to associations between other objects also.
  const edm::InputTag& label_cp_effic_tag = pset.getParameter<edm::InputTag>("label_cp_effic");
  const edm::InputTag& label_cp_fake_tag = pset.getParameter<edm::InputTag>("label_cp_fake");

  label_cp_effic = consumes<std::vector<CaloParticle>>(label_cp_effic_tag);
  label_cp_fake = consumes<std::vector<CaloParticle>>(label_cp_fake_tag);

  simVertices_ = consumes<std::vector<SimVertex>>(pset.getParameter<edm::InputTag>("simVertices"));

  for (auto& itag : label_clustersmask) {
    clustersMaskTokens_.push_back(consumes<std::vector<float>>(itag));
  }

  for (auto& itag : associatorSim_) {
    associatorMapRtSim.push_back(
        consumes<ticl::RecoToSimCollectionWithSimClustersT<reco::CaloClusterCollection>>(itag));
  }
  for (auto& itag : associatorSim_) {
    associatorMapSimtR.push_back(
        consumes<ticl::SimToRecoCollectionWithSimClustersT<reco::CaloClusterCollection>>(itag));
  }

  barrelHitMap_ =
      consumes<std::unordered_map<DetId, const unsigned int>>(edm::InputTag("recHitMapProducer", "barrelRecHitMap"));

  simClusters_ = consumes<std::vector<SimCluster>>(pset.getParameter<edm::InputTag>("label_scl"));

  layerclusters_ = consumes<reco::CaloClusterCollection>(label_lcl);

  for (auto& itag : associator_) {
    associatorMapRtS.push_back(consumes<ticl::RecoToSimCollectionT<reco::CaloClusterCollection>>(itag));
  }
  for (auto& itag : associator_) {
    associatorMapStR.push_back(consumes<ticl::SimToRecoCollectionT<reco::CaloClusterCollection>>(itag));
  }

  cpSelector = CaloParticleSelector(pset.getParameter<double>("ptMinCP"),
                                    pset.getParameter<double>("ptMaxCP"),
                                    pset.getParameter<double>("minRapidityCP"),
                                    pset.getParameter<double>("maxRapidityCP"),
                                    pset.getParameter<double>("lipCP"),
                                    pset.getParameter<double>("tipCP"),
                                    pset.getParameter<int>("minHitCP"),
                                    pset.getParameter<int>("maxSimClustersCP"),
                                    pset.getParameter<bool>("signalOnlyCP"),
                                    pset.getParameter<bool>("intimeOnlyCP"),
                                    pset.getParameter<bool>("chargedOnlyCP"),
                                    pset.getParameter<bool>("stableOnlyCP"),
                                    pset.getParameter<bool>("notConvertedOnlyCP"),
                                    pset.getParameter<std::vector<int>>("pdgIdCP"));

  tools_ = std::make_shared<hgcal::RecHitTools>();

  particles_to_monitor_ = pset.getParameter<std::vector<int>>("pdgIdCP");
  totallayers_to_monitor_ = pset.getParameter<int>("totallayers_to_monitor");

  ParameterSet psetForHistoProducerAlgo = pset.getParameter<ParameterSet>("histoProducerAlgoBlock");
  histoProducerAlgo_ = std::make_unique<BarrelVHistoProducerAlgo>(psetForHistoProducerAlgo);

  dirName_ = pset.getParameter<std::string>("dirName");
}

BarrelValidator::~BarrelValidator() {}

void BarrelValidator::bookHistograms(DQMStore::IBooker& ibook,
                                     edm::Run const&,
                                     edm::EventSetup const& setup,
                                     Histograms& histograms) const {
  if (SaveGeneralInfo_) {
    ibook.cd();
    ibook.setCurrentFolder(dirName_ + "GeneralInfo");
    histoProducerAlgo_->bookInfo(ibook, histograms.histoProducerAlgo);
  }

  if (doCaloParticlePlots_) {
    ibook.cd();

    for (auto const particle : particles_to_monitor_) {
      ibook.setCurrentFolder(dirName_ + "SelectedCaloParticles/" + std::to_string(particle));
      histoProducerAlgo_->bookCaloParticleHistos(
          ibook, histograms.histoProducerAlgo, particle, totallayers_to_monitor_);
    }
    ibook.cd();
    ibook.setCurrentFolder(dirName_);
  }

  //Booking histograms concerning with simClusters
  if (doSimClustersPlots_) {
    ibook.cd();
    ibook.setCurrentFolder(dirName_ + label_SimClustersPlots_ + "/" + label_SimClustersLevel_);
    histoProducerAlgo_->bookSimClusterHistos(ibook, histograms.histoProducerAlgo, totallayers_to_monitor_);

    for (unsigned int ws = 0; ws < label_clustersmask.size(); ws++) {
      ibook.cd();
      InputTag algo = label_clustersmask[ws];
      string dirName = dirName_ + label_SimClustersPlots_ + "/";
      if (!algo.process().empty())
        dirName += algo.process() + "_";
      LogDebug("BarrelValidator") << dirName << "\n";
      if (!algo.label().empty())
        dirName += algo.label() + "_";
      LogDebug("BarrelValidator") << dirName << "\n";
      if (!algo.instance().empty())
        dirName += algo.instance() + "_";
      LogDebug("BarrelValidator") << dirName << "\n";

      if (!dirName.empty()) {
        dirName.resize(dirName.size() - 1);
      }

      LogDebug("BarrelValidator") << dirName << "\n";

      ibook.setCurrentFolder(dirName);

      histoProducerAlgo_->bookSimClusterAssociationHistos(ibook, histograms.histoProducerAlgo, totallayers_to_monitor_);
    }  //end of loop over masks
  }  //if for simCluster plots

  //Booking histograms concerning with hgcal layer clusters
  if (doLayerClustersPlots_) {
    ibook.cd();
    ibook.setCurrentFolder(dirName_ + label_layerClustersPlots_ + "/ClusterLevel");
    histoProducerAlgo_->bookClusterHistos_ClusterLevel(ibook, histograms.histoProducerAlgo, totallayers_to_monitor_);
    ibook.cd();
    ibook.setCurrentFolder(dirName_ + label_layerClustersPlots_ + "/" + label_LCToCPLinking_);
    histoProducerAlgo_->bookClusterHistos_LCtoCP_association(
        ibook, histograms.histoProducerAlgo, totallayers_to_monitor_);

    ibook.cd();
    ibook.setCurrentFolder(dirName_ + label_layerClustersPlots_ + "/CellLevel");
    histoProducerAlgo_->bookClusterHistos_CellLevel(ibook, histograms.histoProducerAlgo, totallayers_to_monitor_);
  }
}

void BarrelValidator::cpParametersAndSelection(const Histograms& histograms,
                                               std::vector<CaloParticle> const& cPeff,
                                               std::vector<SimVertex> const& simVertices,
                                               std::vector<size_t>& selected_cPeff,
                                               unsigned int layers,
                                               std::unordered_map<DetId, const unsigned int> const& barrelHitMap,
                                               edm::MultiSpan<reco::PFRecHit> const& barrelHits) const {
  selected_cPeff.reserve(cPeff.size());

  size_t j = 0;
  for (auto const& caloParticle : cPeff) {
    int id = caloParticle.pdgId();

    if (!doCaloParticleSelection_ || (doCaloParticleSelection_ && cpSelector(caloParticle, simVertices))) {
      selected_cPeff.push_back(j);
      if (doCaloParticlePlots_) {
        histoProducerAlgo_->fill_caloparticle_histos(
            histograms.histoProducerAlgo, id, caloParticle, simVertices, layers, barrelHitMap, barrelHits);
      }
    }
    ++j;
  }  //end of loop over caloparticles
}

void BarrelValidator::dqmAnalyze(const edm::Event& event,
                                 const edm::EventSetup& setup,
                                 const Histograms& histograms) const {
  using namespace reco;

  LogDebug("BarrelValidator") << "\n===================================================="
                              << "\n"
                              << "Analyzing new event"
                              << "\n"
                              << "====================================================\n"
                              << "\n";

  edm::Handle<std::vector<SimVertex>> simVerticesHandle;
  event.getByToken(simVertices_, simVerticesHandle);
  std::vector<SimVertex> const& simVertices = *simVerticesHandle;

  edm::Handle<std::vector<CaloParticle>> caloParticleHandle;
  event.getByToken(label_cp_effic, caloParticleHandle);
  std::vector<CaloParticle> const& caloParticles = *caloParticleHandle;

  edm::ESHandle<CaloGeometry> geom = setup.getHandle(caloGeomToken_);
  tools_->setGeometry(*geom);
  histoProducerAlgo_->setRecHitTools(tools_);

  std::vector<ticl::RecoToSimCollectionT<reco::CaloClusterCollection>> recSimColl;
  std::vector<ticl::SimToRecoCollectionT<reco::CaloClusterCollection>> simRecColl;
  for (unsigned int i = 0; i < associatorMapRtS.size(); ++i) {
    edm::Handle<ticl::SimToRecoCollectionT<reco::CaloClusterCollection>> simToRecoCollectionH;
    event.getByToken(associatorMapStR[i], simToRecoCollectionH);
    simRecColl.push_back(*simToRecoCollectionH);
    edm::Handle<ticl::RecoToSimCollectionT<reco::CaloClusterCollection>> recoToSimCollectionH;
    event.getByToken(associatorMapRtS[i], recoToSimCollectionH);
    recSimColl.push_back(*recoToSimCollectionH);
  }

  edm::Handle<std::unordered_map<DetId, const unsigned int>> barrelHitMapHandle;
  event.getByToken(barrelHitMap_, barrelHitMapHandle);
  const std::unordered_map<DetId, const unsigned int>& barrelHitMap = *barrelHitMapHandle;

  if (!event.getHandle(hitsToken_).isValid()) {
    edm::LogWarning("BarrelValidator") << "MultiPFRecHitCollection token is not valid.";
    return;
  }

  // Protection against missing MultiPFRecHitCollection
  const auto& hits = event.get(hitsToken_);
  for (const auto& pfRecHitCollection : hits) {
    if (pfRecHitCollection->empty()) {
      edm::LogWarning("BarrelValidator") << "One of the PFRecHitCollections is not valid.";
    }
  }

  edm::MultiSpan<reco::PFRecHit> barrelRechitSpan(hits);
  if (barrelRechitSpan.size() == 0) {
    edm::LogWarning("BarrelValidator") << "The PFRecHitCollection MultiSpan is empty.";
  }

  //Some general info on layers etc.
  if (SaveGeneralInfo_) {
    histoProducerAlgo_->fill_info_histos(histograms.histoProducerAlgo, totallayers_to_monitor_);
  }

  std::vector<size_t> cPIndices;
  //Consider CaloParticles coming from the hard scatterer
  //excluding the PU contribution and save the indices.
  removeCPFromPU(caloParticles, cPIndices);

  // ##############################################
  // Fill caloparticles histograms
  // ##############################################
  // HGCRecHit are given to select the SimHits which are also reconstructed
  LogTrace("BarrelValidator") << "\n# of CaloParticles: " << caloParticles.size() << "\n" << std::endl;
  std::vector<size_t> selected_cPeff;
  cpParametersAndSelection(
      histograms, caloParticles, simVertices, selected_cPeff, totallayers_to_monitor_, barrelHitMap, barrelRechitSpan);

  //get collections from the event
  //simClusters
  edm::Handle<std::vector<SimCluster>> simClustersHandle;
  event.getByToken(simClusters_, simClustersHandle);
  std::vector<SimCluster> const& simClusters = *simClustersHandle;

  //Layer clusters
  edm::Handle<reco::CaloClusterCollection> clusterHandle;
  event.getByToken(layerclusters_, clusterHandle);
  const reco::CaloClusterCollection& clusters = *clusterHandle;

  auto nSimClusters = simClusters.size();
  std::vector<size_t> sCIndices;
  //There shouldn't be any SimTracks from different crossings, but maybe they will be added later.
  //At the moment there should be one SimTrack in each SimCluster.
  for (unsigned int scId = 0; scId < nSimClusters; ++scId) {
    if (simClusters[scId].g4Tracks()[0].eventId().event() != 0 or
        simClusters[scId].g4Tracks()[0].eventId().bunchCrossing() != 0) {
      LogDebug("BarrelValidator") << "Excluding SimClusters from event: "
                                  << simClusters[scId].g4Tracks()[0].eventId().event()
                                  << " with BX: " << simClusters[scId].g4Tracks()[0].eventId().bunchCrossing()
                                  << std::endl;
      continue;
    }
    sCIndices.emplace_back(scId);
  }

  // ##############################################
  // Fill simCluster histograms
  // ##############################################
  if (doSimClustersPlots_) {
    histoProducerAlgo_->fill_simCluster_histos(histograms.histoProducerAlgo, simClusters, totallayers_to_monitor_);

    for (unsigned int ws = 0; ws < label_clustersmask.size(); ws++) {
      const auto& inputClusterMask = event.get(clustersMaskTokens_[ws]);

      std::vector<ticl::RecoToSimCollectionWithSimClustersT<reco::CaloClusterCollection>> recSimColl;
      std::vector<ticl::SimToRecoCollectionWithSimClustersT<reco::CaloClusterCollection>> simRecColl;
      for (unsigned int i = 0; i < associatorMapRtSim.size(); ++i) {
        edm::Handle<ticl::SimToRecoCollectionWithSimClustersT<reco::CaloClusterCollection>> simtorecoCollectionH;
        event.getByToken(associatorMapSimtR[i], simtorecoCollectionH);
        simRecColl.push_back(*simtorecoCollectionH);
        edm::Handle<ticl::RecoToSimCollectionWithSimClustersT<reco::CaloClusterCollection>> recotosimCollectionH;
        event.getByToken(associatorMapRtSim[i], recotosimCollectionH);
        recSimColl.push_back(*recotosimCollectionH);
      }

      histoProducerAlgo_->fill_simClusterAssociation_histos(histograms.histoProducerAlgo,
                                                            ws,
                                                            clusterHandle,
                                                            clusters,
                                                            simClustersHandle,
                                                            simClusters,
                                                            sCIndices,
                                                            inputClusterMask,
                                                            barrelHitMap,
                                                            totallayers_to_monitor_,
                                                            recSimColl[0],
                                                            simRecColl[0],
                                                            barrelRechitSpan);

      //General Info on simClusters
      LogTrace("BarrelValidator") << "\n# of SimClusters: " << nSimClusters
                                  << ", layerClusters mask label: " << label_clustersmask[ws].label() << "\n";
    }  //end of loop overs masks
  }

  // ##############################################
  // Fill layercluster histograms
  // ##############################################
  int w = 0;  //counter counting the number of sets of histograms
  if (doLayerClustersPlots_) {
    histoProducerAlgo_->fill_generic_cluster_histos(histograms.histoProducerAlgo,
                                                    w,
                                                    clusterHandle,
                                                    clusters,
                                                    caloParticleHandle,
                                                    caloParticles,
                                                    cPIndices,
                                                    selected_cPeff,
                                                    barrelHitMap,
                                                    totallayers_to_monitor_,
                                                    recSimColl[0],
                                                    simRecColl[0],
                                                    barrelRechitSpan);

    for (unsigned int layerclusterIndex = 0; layerclusterIndex < clusters.size(); layerclusterIndex++) {
      histoProducerAlgo_->fill_cluster_histos(histograms.histoProducerAlgo, w, clusters[layerclusterIndex]);
    }

    //General Info on hgcalLayerClusters
    LogTrace("BarrelValidator") << "\n# of layer clusters with " << label_lcl.process() << ":" << label_lcl.label()
                                << ":" << label_lcl.instance() << ": " << clusters.size() << "\n";
  }
}

void BarrelValidator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // hgcalValidator
  edm::ParameterSetDescription desc;
  desc.add<double>("ptMinCP", 0.5);
  desc.add<double>("ptMaxCP", 300.0);
  desc.add<double>("minRapidityCP", -3.1);
  desc.add<double>("maxRapidityCP", 3.1);
  desc.add<double>("lipCP", 30.0);
  desc.add<double>("tipCP", 60);
  desc.add<bool>("chargedOnlyCP", false);
  desc.add<bool>("stableOnlyCP", false);
  desc.add<bool>("notConvertedOnlyCP", true);
  desc.add<std::vector<int>>("pdgIdCP",
                             {
                                 11,
                                 -11,
                                 13,
                                 -13,
                                 22,
                                 111,
                                 211,
                                 -211,
                                 321,
                                 -321,
                                 311,
                                 130,
                                 310,
                             });
  desc.add<bool>("signalOnlyCP", true);
  desc.add<bool>("intimeOnlyCP", true);
  desc.add<int>("minHitCP", 0);
  desc.add<int>("maxSimClustersCP", -1);
  {
    edm::ParameterSetDescription psd1;
    psd1.add<double>("minEta", -4.5);
    psd1.add<double>("maxEta", 4.5);
    psd1.add<int>("nintEta", 100);
    psd1.add<bool>("useFabsEta", false);
    psd1.add<double>("minEne", 0.0);
    psd1.add<double>("maxEne", 500.0);
    psd1.add<int>("nintEne", 250);
    psd1.add<double>("minPt", 0.0);
    psd1.add<double>("maxPt", 100.0);
    psd1.add<int>("nintPt", 100);
    psd1.add<double>("minPhi", -3.2);
    psd1.add<double>("maxPhi", 3.2);
    psd1.add<int>("nintPhi", 80);
    psd1.add<double>("minMixedHitsSimCluster", 0.0);
    psd1.add<double>("maxMixedHitsSimCluster", 800.0);
    psd1.add<int>("nintMixedHitsSimCluster", 100);
    psd1.add<double>("minMixedHitsCluster", 0.0);
    psd1.add<double>("maxMixedHitsCluster", 800.0);
    psd1.add<int>("nintMixedHitsCluster", 100);
    psd1.add<double>("minEneCl", 0.0);
    psd1.add<double>("maxEneCl", 110.0);
    psd1.add<int>("nintEneCl", 110);
    psd1.add<double>("minLongDepBary", 0.0);
    psd1.add<double>("maxLongDepBary", 110.0);
    psd1.add<int>("nintLongDepBary", 110);
    psd1.add<double>("minZpos", -550.0);
    psd1.add<double>("maxZpos", 550.0);
    psd1.add<int>("nintZpos", 1100);
    psd1.add<double>("minTotNsimClsperlay", 0.0);
    psd1.add<double>("maxTotNsimClsperlay", 50.0);
    psd1.add<int>("nintTotNsimClsperlay", 50);
    psd1.add<double>("minTotNClsperlay", 0.0);
    psd1.add<double>("maxTotNClsperlay", 1000.0);
    psd1.add<int>("nintTotNClsperlay", 50);
    psd1.add<double>("minEneClperlay", 0.0);
    psd1.add<double>("maxEneClperlay", 1000.0);
    psd1.add<int>("nintEneClperlay", 110);
    psd1.add<double>("minScore", 0.0);
    psd1.add<double>("maxScore", 1.02);
    psd1.add<int>("nintScore", 51);
    psd1.add<double>("minSharedEneFrac", 0.0);
    psd1.add<double>("maxSharedEneFrac", 1.02);
    psd1.add<int>("nintSharedEneFrac", 51);
    psd1.add<double>("minTSTSharedEneFracEfficiency", 0.5);
    psd1.add<double>("minTSTSharedEneFrac", 0.0);
    psd1.add<double>("maxTSTSharedEneFrac", 1.01);
    psd1.add<int>("nintTSTSharedEneFrac", 101);
    psd1.add<double>("minTotNTSTs", 0.0);
    psd1.add<double>("maxTotNTSTs", 50.0);
    psd1.add<int>("nintTotNTSTs", 50);
    psd1.add<double>("minTotNClsinTSTs", 0.0);
    psd1.add<double>("maxTotNClsinTSTs", 400.0);
    psd1.add<int>("nintTotNClsinTSTs", 100);

    psd1.add<double>("minTotNClsinTSTsperlayer", 0.0);
    psd1.add<double>("maxTotNClsinTSTsperlayer", 50.0);
    psd1.add<int>("nintTotNClsinTSTsperlayer", 50);
    psd1.add<double>("minMplofLCs", 0.0);
    psd1.add<double>("maxMplofLCs", 20.0);
    psd1.add<int>("nintMplofLCs", 20);
    psd1.add<double>("minSizeCLsinTSTs", 0.0);
    psd1.add<double>("maxSizeCLsinTSTs", 50.0);
    psd1.add<int>("nintSizeCLsinTSTs", 50);
    psd1.add<double>("minClEnepermultiplicity", 0.0);
    psd1.add<double>("maxClEnepermultiplicity", 10.0);
    psd1.add<int>("nintClEnepermultiplicity", 10);
    psd1.add<double>("minX", -300.0);
    psd1.add<double>("maxX", 300.0);
    psd1.add<int>("nintX", 100);
    psd1.add<double>("minY", -300.0);
    psd1.add<double>("maxY", 300.0);
    psd1.add<int>("nintY", 100);
    psd1.add<double>("minZ", -550.0);
    psd1.add<double>("maxZ", 550.0);
    psd1.add<int>("nintZ", 1100);
    desc.add<edm::ParameterSetDescription>("histoProducerAlgoBlock", psd1);
  }
  desc.add<edm::InputTag>("hits", edm::InputTag("recHitMapProducer", "MultiPFRecHitCollectionProduct"));
  desc.add<edm::InputTag>("label_lcl", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<std::vector<edm::InputTag>>("label_tst",
                                       {
                                           edm::InputTag("ticlTrackstersCLUE3DHigh"),
                                           edm::InputTag("ticlTrackstersMerge"),
                                           edm::InputTag("ticlSimTracksters", "fromCPs"),
                                           edm::InputTag("ticlSimTracksters"),
                                       });
  desc.add<edm::InputTag>("label_simTS", edm::InputTag("ticlSimTracksters"));
  desc.add<edm::InputTag>("label_simTSFromCP", edm::InputTag("ticlSimTracksters", "fromCPs"));
  desc.addUntracked<std::vector<edm::InputTag>>("associator",
                                                {edm::InputTag("barrelLayerClusterCaloParticleAssociation")});
  desc.addUntracked<std::vector<edm::InputTag>>("associatorSim",
                                                {edm::InputTag("barrelLayerClusterSimClusterAssociation")});
  desc.addUntracked<bool>("SaveGeneralInfo", true);
  desc.addUntracked<bool>("doCaloParticlePlots", true);
  desc.addUntracked<bool>("doCaloParticleSelection", true);
  desc.addUntracked<bool>("doSimClustersPlots", true);
  desc.add<std::string>("label_SimClusters", "SimClusters");
  desc.add<std::string>("label_SimClustersLevel", "ClusterLevel");
  desc.addUntracked<bool>("doLayerClustersPlots", true);
  desc.add<std::string>("label_layerClustersPlots", "LayerClusters");
  desc.add<std::string>("label_LCToCPLinking", "LCToCP_association");
  desc.add<edm::InputTag>("simClustersToCaloParticlesMap",
                          edm::InputTag("SimClusterToCaloParticleAssociation", "simClusterToCaloParticleMap"));
  desc.add<edm::InputTag>("label_cp_effic", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("label_cp_fake", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("label_scl", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("simVertices", edm::InputTag("g4SimHits"));
  desc.add<int>("totallayers_to_monitor", 5);
  desc.add<std::string>("dirName", "BarrelCalorimeters/BarrelValidator/");
  descriptions.add("barrelValidator", desc);
}
