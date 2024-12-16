#include "Validation/HGCalValidation/interface/HGCalValidator.h"

#include "SimCalorimetry/HGCalAssociatorProducers/interface/AssociatorTools.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

using namespace std;
using namespace edm;
using namespace ticl;

namespace {
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

}  // namespace

HGCalValidator::HGCalValidator(const edm::ParameterSet& pset)
    : caloGeomToken_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
      label_lcl(pset.getParameter<edm::InputTag>("label_lcl")),
      label_tst(pset.getParameter<std::vector<edm::InputTag>>("label_tst")),
      allTracksterTracksterAssociatorsLabels_(
          pset.getParameter<std::vector<edm::InputTag>>("allTracksterTracksterAssociatorsLabels")),
      allTracksterTracksterByHitsAssociatorsLabels_(
          pset.getParameter<std::vector<edm::InputTag>>("allTracksterTracksterByHitsAssociatorsLabels")),
      label_simTS(pset.getParameter<edm::InputTag>("label_simTS")),
      label_simTSFromCP(pset.getParameter<edm::InputTag>("label_simTSFromCP")),
      associator_(pset.getUntrackedParameter<edm::InputTag>("associator")),
      associatorSim_(pset.getUntrackedParameter<edm::InputTag>("associatorSim")),
      SaveGeneralInfo_(pset.getUntrackedParameter<bool>("SaveGeneralInfo")),
      doCaloParticlePlots_(pset.getUntrackedParameter<bool>("doCaloParticlePlots")),
      doCaloParticleSelection_(pset.getUntrackedParameter<bool>("doCaloParticleSelection")),
      doSimClustersPlots_(pset.getUntrackedParameter<bool>("doSimClustersPlots")),
      label_SimClustersPlots_(pset.getParameter<edm::InputTag>("label_SimClusters")),
      label_SimClustersLevel_(pset.getParameter<edm::InputTag>("label_SimClustersLevel")),
      doLayerClustersPlots_(pset.getUntrackedParameter<bool>("doLayerClustersPlots")),
      label_layerClustersPlots_(pset.getParameter<edm::InputTag>("label_layerClusterPlots")),
      label_LCToCPLinking_(pset.getParameter<edm::InputTag>("label_LCToCPLinking")),
      doTrackstersPlots_(pset.getUntrackedParameter<bool>("doTrackstersPlots")),
      label_TS_(pset.getParameter<std::string>("label_TS")),
      label_TSbyHitsCP_(pset.getParameter<std::string>("label_TSbyHitsCP")),
      label_TSbyHits_(pset.getParameter<std::string>("label_TSbyHits")),
      label_TSbyLCsCP_(pset.getParameter<std::string>("label_TSbyLCsCP")),
      label_TSbyLCs_(pset.getParameter<std::string>("label_TSbyLCs")),
      label_clustersmask(pset.getParameter<std::vector<edm::InputTag>>("LayerClustersInputMask")),
      doCandidatesPlots_(pset.getUntrackedParameter<bool>("doCandidatesPlots")),
      label_candidates_(pset.getParameter<std::string>("ticlCandidates")),
      cummatbudinxo_(pset.getParameter<edm::FileInPath>("cummatbudinxo")),
      isTICLv5_(pset.getUntrackedParameter<bool>("isticlv5")),
      hits_label_(pset.getParameter<std::vector<edm::InputTag>>("hits")),
      scToCpMapToken_(
          consumes<SimClusterToCaloParticleMap>(pset.getParameter<edm::InputTag>("simClustersToCaloParticlesMap"))) {
  //In this way we can easily generalize to associations between other objects also.
  const edm::InputTag& label_cp_effic_tag = pset.getParameter<edm::InputTag>("label_cp_effic");
  const edm::InputTag& label_cp_fake_tag = pset.getParameter<edm::InputTag>("label_cp_fake");

  for (auto& label : hits_label_) {
    hits_tokens_.push_back(consumes<HGCRecHitCollection>(label));
  }
  label_cp_effic = consumes<std::vector<CaloParticle>>(label_cp_effic_tag);
  label_cp_fake = consumes<std::vector<CaloParticle>>(label_cp_fake_tag);

  simVertices_ = consumes<std::vector<SimVertex>>(pset.getParameter<edm::InputTag>("simVertices"));

  for (auto& itag : label_clustersmask) {
    clustersMaskTokens_.push_back(consumes<std::vector<float>>(itag));
  }

  associatorMapSimtR = consumes<ticl::SimToRecoCollectionWithSimClusters>(associatorSim_);
  associatorMapRtSim = consumes<ticl::RecoToSimCollectionWithSimClusters>(associatorSim_);

  simTrackstersMap_ = consumes<std::map<uint, std::vector<uint>>>(edm::InputTag("ticlSimTracksters"));

  hitMap_ =
      consumes<std::unordered_map<DetId, const unsigned int>>(edm::InputTag("recHitMapProducer", "hgcalRecHitMap"));

  simClusters_ = consumes<std::vector<SimCluster>>(pset.getParameter<edm::InputTag>("label_scl"));

  layerclusters_ = consumes<reco::CaloClusterCollection>(label_lcl);
  for (const auto& tag : allTracksterTracksterAssociatorsLabels_) {
    tracksterToTracksterAssociatorsTokens_.emplace_back(consumes<TracksterToTracksterMap>(tag));
  }

  for (const auto& tag : allTracksterTracksterByHitsAssociatorsLabels_) {
    tracksterToTracksterByHitsAssociatorsTokens_.emplace_back(consumes<TracksterToTracksterMap>(tag));
  }

  if (doCandidatesPlots_) {
    edm::EDGetTokenT<std::vector<TICLCandidate>> TICLCandidatesToken =
        consumes<std::vector<TICLCandidate>>(pset.getParameter<edm::InputTag>("ticlTrackstersMerge"));
    edm::EDGetTokenT<std::vector<TICLCandidate>> simTICLCandidatesToken =
        consumes<std::vector<TICLCandidate>>(pset.getParameter<edm::InputTag>("simTiclCandidates"));
    edm::EDGetTokenT<std::vector<reco::Track>> recoTracksToken =
        consumes<std::vector<reco::Track>>(pset.getParameter<edm::InputTag>("recoTracks"));
    edm::EDGetTokenT<std::vector<ticl::Trackster>> trackstersToken =
        consumes<std::vector<ticl::Trackster>>(pset.getParameter<edm::InputTag>("ticlTrackstersMerge"));
    edm::EDGetTokenT<ticl::TracksterToTracksterMap> associatorMapRtSToken =
        consumes<ticl::TracksterToTracksterMap>(pset.getParameter<edm::InputTag>("mergeRecoToSimAssociator"));
    edm::EDGetTokenT<ticl::TracksterToTracksterMap> associatorMapStRToken =
        consumes<ticl::TracksterToTracksterMap>(pset.getParameter<edm::InputTag>("mergeSimToRecoAssociator"));

    candidateVal_ = std::make_unique<TICLCandidateValidator>(TICLCandidatesToken,
                                                             simTICLCandidatesToken,
                                                             recoTracksToken,
                                                             trackstersToken,
                                                             associatorMapRtSToken,
                                                             associatorMapStRToken,
                                                             isTICLv5_);
  }

  for (auto& itag : label_tst) {
    label_tstTokens.push_back(consumes<ticl::TracksterCollection>(itag));
  }

  simTracksters_ = consumes<ticl::TracksterCollection>(label_simTS);
  simTracksters_fromCPs_ = consumes<ticl::TracksterCollection>(label_simTSFromCP);

  associatorMapRtS = consumes<ticl::RecoToSimCollection>(associator_);
  associatorMapStR = consumes<ticl::SimToRecoCollection>(associator_);

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

  tools_.reset(new hgcal::RecHitTools());

  particles_to_monitor_ = pset.getParameter<std::vector<int>>("pdgIdCP");
  totallayers_to_monitor_ = pset.getParameter<int>("totallayers_to_monitor");
  thicknesses_to_monitor_ = pset.getParameter<std::vector<int>>("thicknesses_to_monitor");

  //For the material budget file here
  std::ifstream fmb(cummatbudinxo_.fullPath().c_str());
  double thelay = 0.;
  double mbg = 0.;
  for (unsigned ilayer = 1; ilayer <= totallayers_to_monitor_; ++ilayer) {
    fmb >> thelay >> mbg;
    cumulative_material_budget.insert(std::pair<double, double>(thelay, mbg));
  }

  fmb.close();

  ParameterSet psetForHistoProducerAlgo = pset.getParameter<ParameterSet>("histoProducerAlgoBlock");
  histoProducerAlgo_ = std::make_unique<HGVHistoProducerAlgo>(psetForHistoProducerAlgo);

  dirName_ = pset.getParameter<std::string>("dirName");
}

HGCalValidator::~HGCalValidator() {}

void HGCalValidator::bookHistograms(DQMStore::IBooker& ibook,
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
    ibook.setCurrentFolder(dirName_ + label_SimClustersPlots_.label() + "/" + label_SimClustersLevel_.label());
    histoProducerAlgo_->bookSimClusterHistos(
        ibook, histograms.histoProducerAlgo, totallayers_to_monitor_, thicknesses_to_monitor_);

    for (unsigned int ws = 0; ws < label_clustersmask.size(); ws++) {
      ibook.cd();
      InputTag algo = label_clustersmask[ws];
      string dirName = dirName_ + label_SimClustersPlots_.label() + "/";
      if (!algo.process().empty())
        dirName += algo.process() + "_";
      LogDebug("HGCalValidator") << dirName << "\n";
      if (!algo.label().empty())
        dirName += algo.label() + "_";
      LogDebug("HGCalValidator") << dirName << "\n";
      if (!algo.instance().empty())
        dirName += algo.instance() + "_";
      LogDebug("HGCalValidator") << dirName << "\n";

      if (!dirName.empty()) {
        dirName.resize(dirName.size() - 1);
      }

      LogDebug("HGCalValidator") << dirName << "\n";

      ibook.setCurrentFolder(dirName);

      histoProducerAlgo_->bookSimClusterAssociationHistos(
          ibook, histograms.histoProducerAlgo, totallayers_to_monitor_, thicknesses_to_monitor_);
    }  //end of loop over masks
  }  //if for simCluster plots

  //Booking histograms concerning with hgcal layer clusters
  if (doLayerClustersPlots_) {
    ibook.cd();
    ibook.setCurrentFolder(dirName_ + label_layerClustersPlots_.label() + "/ClusterLevel");
    histoProducerAlgo_->bookClusterHistos_ClusterLevel(ibook,
                                                       histograms.histoProducerAlgo,
                                                       totallayers_to_monitor_,
                                                       thicknesses_to_monitor_,
                                                       cummatbudinxo_.fullPath());
    ibook.cd();
    ibook.setCurrentFolder(dirName_ + label_layerClustersPlots_.label() + "/" + label_LCToCPLinking_.label());
    histoProducerAlgo_->bookClusterHistos_LCtoCP_association(
        ibook, histograms.histoProducerAlgo, totallayers_to_monitor_);

    ibook.cd();
    ibook.setCurrentFolder(dirName_ + label_layerClustersPlots_.label() + "/CellLevel");
    histoProducerAlgo_->bookClusterHistos_CellLevel(
        ibook, histograms.histoProducerAlgo, totallayers_to_monitor_, thicknesses_to_monitor_);
  }

  //Booking histograms for Tracksters
  for (unsigned int www = 0; www < label_tst.size(); www++) {
    ibook.cd();
    InputTag algo = label_tst[www];
    string dirName = dirName_;
    if (!algo.process().empty())
      dirName += algo.process() + "_";
    LogDebug("HGCalValidator") << dirName << "\n";
    if (!algo.label().empty())
      dirName += algo.label() + "_";
    LogDebug("HGCalValidator") << dirName << "\n";
    if (!algo.instance().empty())
      dirName += algo.instance() + "_";
    LogDebug("HGCalValidator") << dirName << "\n";

    if (!dirName.empty()) {
      dirName.resize(dirName.size() - 1);
    }

    ibook.setCurrentFolder(dirName);

    // Booking histograms concerning HGCal tracksters
    if (doTrackstersPlots_) {
      // Generic histos
      ibook.setCurrentFolder(dirName + "/" + label_TS_);
      histoProducerAlgo_->bookTracksterHistos(ibook, histograms.histoProducerAlgo, totallayers_to_monitor_);
      // CP Linking by Hits
      ibook.setCurrentFolder(dirName + "/" + label_TSbyHitsCP_);
      histoProducerAlgo_->bookTracksterSTSHistos(
          ibook, histograms.histoProducerAlgo, HGVHistoProducerAlgo::validationType::byHits_CP);
      // CP Linking by LCs
      ibook.setCurrentFolder(dirName + "/" + label_TSbyLCsCP_);

      histoProducerAlgo_->bookTracksterSTSHistos(
          ibook, histograms.histoProducerAlgo, HGVHistoProducerAlgo::validationType::byLCs_CP);
      // SimTracksters Linking by Hits
      ibook.setCurrentFolder(dirName + "/" + label_TSbyHits_);
      histoProducerAlgo_->bookTracksterSTSHistos(
          ibook, histograms.histoProducerAlgo, HGVHistoProducerAlgo::validationType::byHits);
      // SimTracksters Linking by LCs
      ibook.setCurrentFolder(dirName + "/" + label_TSbyLCs_);
      histoProducerAlgo_->bookTracksterSTSHistos(
          ibook, histograms.histoProducerAlgo, HGVHistoProducerAlgo::validationType::byLCs);
    }
  }  //end of booking Tracksters loop

  // Booking histograms concerning TICL candidates
  if (doCandidatesPlots_) {
    ibook.cd();
    ibook.setCurrentFolder(dirName_ + label_candidates_);
    candidateVal_->bookCandidatesHistos(ibook, histograms.histoTICLCandidates, dirName_ + label_candidates_);
  }
}

void HGCalValidator::cpParametersAndSelection(const Histograms& histograms,
                                              std::vector<CaloParticle> const& cPeff,
                                              std::vector<SimVertex> const& simVertices,
                                              std::vector<size_t>& selected_cPeff,
                                              unsigned int layers,
                                              std::unordered_map<DetId, const unsigned int> const& hitMap,
                                              MultiVectorManager<HGCRecHit> const& hits) const {
  selected_cPeff.reserve(cPeff.size());

  size_t j = 0;
  for (auto const& caloParticle : cPeff) {
    int id = caloParticle.pdgId();

    if (!doCaloParticleSelection_ || (doCaloParticleSelection_ && cpSelector(caloParticle, simVertices))) {
      selected_cPeff.push_back(j);
      if (doCaloParticlePlots_) {
        histoProducerAlgo_->fill_caloparticle_histos(
            histograms.histoProducerAlgo, id, caloParticle, simVertices, layers, hitMap, hits);
      }
    }
    ++j;
  }  //end of loop over caloparticles
}

void HGCalValidator::dqmAnalyze(const edm::Event& event,
                                const edm::EventSetup& setup,
                                const Histograms& histograms) const {
  using namespace reco;

  LogDebug("HGCalValidator") << "\n===================================================="
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

  edm::Handle<ticl::TracksterCollection> simTracksterHandle;
  event.getByToken(simTracksters_, simTracksterHandle);
  ticl::TracksterCollection const& simTracksters = *simTracksterHandle;

  edm::Handle<ticl::TracksterCollection> simTracksterFromCPHandle;
  event.getByToken(simTracksters_fromCPs_, simTracksterFromCPHandle);
  ticl::TracksterCollection const& simTrackstersFromCPs = *simTracksterFromCPHandle;

  edm::Handle<std::map<uint, std::vector<uint>>> simTrackstersMapHandle;
  event.getByToken(simTrackstersMap_, simTrackstersMapHandle);
  const std::map<uint, std::vector<uint>>& cpToSc_SimTrackstersMap = *simTrackstersMapHandle;

  edm::ESHandle<CaloGeometry> geom = setup.getHandle(caloGeomToken_);
  tools_->setGeometry(*geom);
  histoProducerAlgo_->setRecHitTools(tools_);

  edm::Handle<ticl::SimToRecoCollection> simtorecoCollectionH;
  event.getByToken(associatorMapStR, simtorecoCollectionH);
  const auto& simRecColl = *simtorecoCollectionH;
  edm::Handle<ticl::RecoToSimCollection> recotosimCollectionH;
  event.getByToken(associatorMapRtS, recotosimCollectionH);
  const auto& recSimColl = *recotosimCollectionH;

  edm::Handle<std::unordered_map<DetId, const unsigned int>> hitMapHandle;
  event.getByToken(hitMap_, hitMapHandle);
  const std::unordered_map<DetId, const unsigned int>& hitMap = *hitMapHandle;

  MultiVectorManager<HGCRecHit> rechitManager;
  for (const auto &token : hits_tokens_) {
    Handle<HGCRecHitCollection> hitsHandle;
    event.getByToken(token, hitsHandle);
    rechitManager.addVector(*hitsHandle);
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
  LogTrace("HGCalValidator") << "\n# of CaloParticles: " << caloParticles.size() << "\n" << std::endl;
  std::vector<size_t> selected_cPeff;
  cpParametersAndSelection(
      histograms, caloParticles, simVertices, selected_cPeff, totallayers_to_monitor_, hitMap, rechitManager);

  //get collections from the event
  //simClusters
  edm::Handle<std::vector<SimCluster>> simClustersHandle;
  event.getByToken(simClusters_, simClustersHandle);
  std::vector<SimCluster> const& simClusters = *simClustersHandle;

  //Layer clusters
  edm::Handle<reco::CaloClusterCollection> clusterHandle;
  event.getByToken(layerclusters_, clusterHandle);
  const reco::CaloClusterCollection& clusters = *clusterHandle;

  std::vector<edm::Handle<TracksterToTracksterMap>> tracksterToTracksterMapsHandles;
  for (auto& token : tracksterToTracksterAssociatorsTokens_) {
    edm::Handle<TracksterToTracksterMap> tracksterToTracksterMapHandle;
    event.getByToken(token, tracksterToTracksterMapHandle);
    tracksterToTracksterMapsHandles.push_back(tracksterToTracksterMapHandle);
  }

  std::vector<edm::Handle<TracksterToTracksterMap>> tracksterToTracksterByHitsMapsHandles;
  for (auto& token : tracksterToTracksterByHitsAssociatorsTokens_) {
    edm::Handle<TracksterToTracksterMap> tracksterToTracksterByHitsMapHandle;
    event.getByToken(token, tracksterToTracksterByHitsMapHandle);
    tracksterToTracksterByHitsMapsHandles.push_back(tracksterToTracksterByHitsMapHandle);
  }

  edm::Handle<SimClusterToCaloParticleMap> scToCpMapHandle;
  event.getByToken(scToCpMapToken_, scToCpMapHandle);
  const SimClusterToCaloParticleMap& scToCpMap = *scToCpMapHandle;

  auto nSimClusters = simClusters.size();
  std::vector<size_t> sCIndices;
  //There shouldn't be any SimTracks from different crossings, but maybe they will be added later.
  //At the moment there should be one SimTrack in each SimCluster.
  for (unsigned int scId = 0; scId < nSimClusters; ++scId) {
    if (simClusters[scId].g4Tracks()[0].eventId().event() != 0 or
        simClusters[scId].g4Tracks()[0].eventId().bunchCrossing() != 0) {
      LogDebug("HGCalValidator") << "Excluding SimClusters from event: "
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
    histoProducerAlgo_->fill_simCluster_histos(
        histograms.histoProducerAlgo, simClusters, totallayers_to_monitor_, thicknesses_to_monitor_);

    for (unsigned int ws = 0; ws < label_clustersmask.size(); ws++) {
      const auto& inputClusterMask = event.get(clustersMaskTokens_[ws]);

      edm::Handle<ticl::SimToRecoCollectionWithSimClusters> simtorecoCollectionH;
      event.getByToken(associatorMapSimtR, simtorecoCollectionH);
      auto simRecColl = *simtorecoCollectionH;
      edm::Handle<ticl::RecoToSimCollectionWithSimClusters> recotosimCollectionH;
      event.getByToken(associatorMapRtSim, recotosimCollectionH);
      auto recSimColl = *recotosimCollectionH;

      histoProducerAlgo_->fill_simClusterAssociation_histos(histograms.histoProducerAlgo,
                                                            ws,
                                                            clusterHandle,
                                                            clusters,
                                                            simClustersHandle,
                                                            simClusters,
                                                            sCIndices,
                                                            inputClusterMask,
                                                            hitMap,
                                                            totallayers_to_monitor_,
                                                            recSimColl,
                                                            simRecColl,
                                                            rechitManager);

      //General Info on simClusters
      LogTrace("HGCalValidator") << "\n# of SimClusters: " << nSimClusters
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
                                                    hitMap,
                                                    cumulative_material_budget,
                                                    totallayers_to_monitor_,
                                                    thicknesses_to_monitor_,
                                                    recSimColl,
                                                    simRecColl,
                                                    rechitManager);

    for (unsigned int layerclusterIndex = 0; layerclusterIndex < clusters.size(); layerclusterIndex++) {
      histoProducerAlgo_->fill_cluster_histos(histograms.histoProducerAlgo, w, clusters[layerclusterIndex]);
    }

    //General Info on hgcalLayerClusters
    LogTrace("HGCalValidator") << "\n# of layer clusters with " << label_lcl.process() << ":" << label_lcl.label()
                               << ":" << label_lcl.instance() << ": " << clusters.size() << "\n";
  }

  // ##############################################
  // Fill Trackster histograms
  // ##############################################
  for (unsigned int wml = 0; wml < label_tstTokens.size(); wml++) {
    if (doTrackstersPlots_) {
      edm::Handle<ticl::TracksterCollection> tracksterHandle;
      event.getByToken(label_tstTokens[wml], tracksterHandle);
      const ticl::TracksterCollection& tracksters = *tracksterHandle;
      if (tracksterHandle.id() == simTracksterHandle.id() or tracksterHandle.id() == simTracksterFromCPHandle.id())
        continue;
      edm::Handle<TracksterToTracksterMap> trackstersToSimTrackstersMapH, simTrackstersToTrackstersMapH,
          trackstersToSimTrackstersFromCPsMapH, simTrackstersFromCPsToTrackstersMapH,
          trackstersToSimTrackstersByHitsMapH, simTrackstersToTrackstersByHitsMapH,
          trackstersToSimTrackstersFromCPsByHitsMapH, simTrackstersFromCPsToTrackstersByHitsMapH;

      bool mapsFound = assignTracksterMaps(tracksterHandle,
                                           simTracksterHandle,
                                           simTracksterFromCPHandle,
                                           tracksterToTracksterMapsHandles,
                                           trackstersToSimTrackstersMapH,
                                           simTrackstersToTrackstersMapH,
                                           trackstersToSimTrackstersFromCPsMapH,
                                           simTrackstersFromCPsToTrackstersMapH);

      mapsFound = mapsFound and assignTracksterMaps(tracksterHandle,
                                                    simTracksterHandle,
                                                    simTracksterFromCPHandle,
                                                    tracksterToTracksterByHitsMapsHandles,
                                                    trackstersToSimTrackstersByHitsMapH,
                                                    simTrackstersToTrackstersByHitsMapH,
                                                    trackstersToSimTrackstersFromCPsByHitsMapH,
                                                    simTrackstersFromCPsToTrackstersByHitsMapH);

      histoProducerAlgo_->fill_trackster_histos(histograms.histoProducerAlgo,
                                                wml,
                                                tracksters,
                                                clusters,
                                                simTracksters,
                                                simTrackstersFromCPs,
                                                cpToSc_SimTrackstersMap,
                                                simClusters,
                                                caloParticleHandle.id(),
                                                caloParticles,
                                                cPIndices,
                                                selected_cPeff,
                                                hitMap,
                                                totallayers_to_monitor_,
                                                rechitManager,
                                                mapsFound,
                                                trackstersToSimTrackstersMapH,
                                                simTrackstersToTrackstersMapH,
                                                trackstersToSimTrackstersFromCPsMapH,
                                                simTrackstersFromCPsToTrackstersMapH,
                                                trackstersToSimTrackstersByHitsMapH,
                                                simTrackstersToTrackstersByHitsMapH,
                                                trackstersToSimTrackstersFromCPsByHitsMapH,
                                                simTrackstersFromCPsToTrackstersByHitsMapH,
                                                scToCpMap);
    }
  }  //end of loop over Trackster input labels

  // tracksters histograms
  if (doCandidatesPlots_) {
    candidateVal_->fillCandidateHistos(event, histograms.histoTICLCandidates, simTracksterFromCPHandle);
  }
}

void HGCalValidator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
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
    psd1.add<double>("maxTotNClsperlay", 50.0);
    psd1.add<int>("nintTotNClsperlay", 50);
    psd1.add<double>("minEneClperlay", 0.0);
    psd1.add<double>("maxEneClperlay", 110.0);
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
    psd1.add<double>("minTotNsimClsperthick", 0.0);
    psd1.add<double>("maxTotNsimClsperthick", 800.0);
    psd1.add<int>("nintTotNsimClsperthick", 100);
    psd1.add<double>("minTotNClsperthick", 0.0);
    psd1.add<double>("maxTotNClsperthick", 800.0);
    psd1.add<int>("nintTotNClsperthick", 100);
    psd1.add<double>("minTotNcellsperthickperlayer", 0.0);
    psd1.add<double>("maxTotNcellsperthickperlayer", 500.0);
    psd1.add<int>("nintTotNcellsperthickperlayer", 100);
    psd1.add<double>("minDisToSeedperthickperlayer", 0.0);
    psd1.add<double>("maxDisToSeedperthickperlayer", 300.0);
    psd1.add<int>("nintDisToSeedperthickperlayer", 100);
    psd1.add<double>("minDisToSeedperthickperlayerenewei", 0.0);
    psd1.add<double>("maxDisToSeedperthickperlayerenewei", 10.0);
    psd1.add<int>("nintDisToSeedperthickperlayerenewei", 50);
    psd1.add<double>("minDisToMaxperthickperlayer", 0.0);
    psd1.add<double>("maxDisToMaxperthickperlayer", 300.0);
    psd1.add<int>("nintDisToMaxperthickperlayer", 100);
    psd1.add<double>("minDisToMaxperthickperlayerenewei", 0.0);
    psd1.add<double>("maxDisToMaxperthickperlayerenewei", 50.0);
    psd1.add<int>("nintDisToMaxperthickperlayerenewei", 50);
    psd1.add<double>("minDisSeedToMaxperthickperlayer", 0.0);
    psd1.add<double>("maxDisSeedToMaxperthickperlayer", 300.0);
    psd1.add<int>("nintDisSeedToMaxperthickperlayer", 100);
    psd1.add<double>("minClEneperthickperlayer", 0.0);
    psd1.add<double>("maxClEneperthickperlayer", 10.0);
    psd1.add<int>("nintClEneperthickperlayer", 100);
    psd1.add<double>("minCellsEneDensperthick", 0.0);
    psd1.add<double>("maxCellsEneDensperthick", 100.0);
    psd1.add<int>("nintCellsEneDensperthick", 200);
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
  desc.add<std::vector<edm::InputTag>>("hits",
                                       {
                                           edm::InputTag("HGCalRecHit", "HGCEERecHits"),
                                           edm::InputTag("HGCalRecHit", "HGCHEFRecHits"),
                                           edm::InputTag("HGCalRecHit", "HGCHEBRecHits"),
                                       });
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
  desc.addUntracked<edm::InputTag>("associator", edm::InputTag("layerClusterCaloParticleAssociationProducer"));
  desc.addUntracked<edm::InputTag>("associatorSim", edm::InputTag("layerClusterSimClusterAssociationProducer"));
  desc.addUntracked<bool>("SaveGeneralInfo", true);
  desc.addUntracked<bool>("doCaloParticlePlots", true);
  desc.addUntracked<bool>("doCaloParticleSelection", true);
  desc.addUntracked<bool>("doSimClustersPlots", true);
  desc.add<edm::InputTag>("label_SimClusters", edm::InputTag("SimClusters"));
  desc.add<edm::InputTag>("label_SimClustersLevel", edm::InputTag("ClusterLevel"));
  desc.addUntracked<bool>("doLayerClustersPlots", true);
  desc.add<edm::InputTag>("label_layerClusterPlots", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<edm::InputTag>("label_LCToCPLinking", edm::InputTag("LCToCP_association"));
  desc.addUntracked<bool>("doTrackstersPlots", true);
  desc.add<std::string>("label_TS", "Morphology");
  desc.add<std::string>("label_TSbyHitsCP", "TSbyHits_CP");
  desc.add<std::string>("label_TSbyHits", "TSbyHits");
  desc.add<std::string>("label_TSbyLCs", "TSbyLCs");
  desc.add<std::string>("label_TSbyLCsCP", "TSbyLCs_CP");
  desc.add<edm::InputTag>("simClustersToCaloParticlesMap",
                          edm::InputTag("SimClusterToCaloParticleAssociation", "simClusterToCaloParticleMap"));
  desc.add<std::vector<edm::InputTag>>(
      "allTracksterTracksterAssociatorsLabels",
      {
          edm::InputTag("allTrackstersToSimTrackstersAssociationsByLCs", "ticlTrackstersCLUE3DHighToticlSimTracksters"),
          edm::InputTag("allTrackstersToSimTrackstersAssociationsByLCs", "ticlSimTrackstersToticlTrackstersCLUE3DHigh"),
          edm::InputTag("allTrackstersToSimTrackstersAssociationsByLCs",
                        "ticlTrackstersCLUE3DHighToticlSimTrackstersfromCPs"),
          edm::InputTag("allTrackstersToSimTrackstersAssociationsByLCs",
                        "ticlSimTrackstersfromCPsToticlTrackstersCLUE3DHigh"),
          edm::InputTag("allTrackstersToSimTrackstersAssociationsByLCs", "ticlTracksterLinksToticlSimTracksters"),
          edm::InputTag("allTrackstersToSimTrackstersAssociationsByLCs", "ticlSimTrackstersToticlTracksterLinks"),
          edm::InputTag("allTrackstersToSimTrackstersAssociationsByLCs",
                        "ticlTracksterLinksToticlSimTrackstersfromCPs"),
          edm::InputTag("allTrackstersToSimTrackstersAssociationsByLCs",
                        "ticlSimTrackstersfromCPsToticlTracksterLinks"),
          edm::InputTag("allTrackstersToSimTrackstersAssociationsByLCs", "ticlCandidateToticlSimTracksters"),
          edm::InputTag("allTrackstersToSimTrackstersAssociationsByLCs", "ticlSimTrackstersToticlCandidate"),
          edm::InputTag("allTrackstersToSimTrackstersAssociationsByLCs", "ticlCandidateToticlSimTrackstersfromCPs"),
          edm::InputTag("allTrackstersToSimTrackstersAssociationsByLCs", "ticlSimTrackstersfromCPsToticlCandidate"),
      });
  desc.add<std::vector<edm::InputTag>>(
      "allTracksterTracksterByHitsAssociatorsLabels",
      {
          edm::InputTag("allTrackstersToSimTrackstersAssociationsByHits",
                        "ticlTrackstersCLUE3DHighToticlSimTracksters"),
          edm::InputTag("allTrackstersToSimTrackstersAssociationsByHits",
                        "ticlSimTrackstersToticlTrackstersCLUE3DHigh"),
          edm::InputTag("allTrackstersToSimTrackstersAssociationsByHits",
                        "ticlTrackstersCLUE3DHighToticlSimTrackstersfromCPs"),
          edm::InputTag("allTrackstersToSimTrackstersAssociationsByHits",
                        "ticlSimTrackstersfromCPsToticlTrackstersCLUE3DHigh"),
          edm::InputTag("allTrackstersToSimTrackstersAssociationsByHits", "ticlTracksterLinksToticlSimTracksters"),
          edm::InputTag("allTrackstersToSimTrackstersAssociationsByHits", "ticlSimTrackstersToticlTracksterLinks"),
          edm::InputTag("allTrackstersToSimTrackstersAssociationsByHits",
                        "ticlTracksterLinksToticlSimTrackstersfromCPs"),
          edm::InputTag("allTrackstersToSimTrackstersAssociationsByHits",
                        "ticlSimTrackstersfromCPsToticlTracksterLinks"),
          edm::InputTag("allTrackstersToSimTrackstersAssociationsByHits", "ticlCandidateToticlSimTracksters"),
          edm::InputTag("allTrackstersToSimTrackstersAssociationsByHits", "ticlSimTrackstersToticlCandidate"),
          edm::InputTag("allTrackstersToSimTrackstersAssociationsByHits", "ticlCandidateToticlSimTrackstersfromCPs"),
          edm::InputTag("allTrackstersToSimTrackstersAssociationsByHits", "ticlSimTrackstersfromCPsToticlCandidate"),
      });
  desc.addUntracked<bool>("doCandidatesPlots", true);
  desc.add<std::string>("ticlCandidates", "ticlCandidates");
  desc.add<edm::InputTag>("ticlTrackstersMerge", edm::InputTag("ticlTrackstersMerge"));
  desc.add<edm::InputTag>("simTiclCandidates", edm::InputTag("ticlSimTracksters"));
  desc.add<edm::InputTag>("recoTracks", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>(
      "mergeRecoToSimAssociator",
      edm::InputTag("allTrackstersToSimTrackstersAssociationsByLCs", "ticlTrackstersMergeToticlSimTrackstersfromCPs"));
  desc.add<edm::InputTag>(
      "mergeSimToRecoAssociator",
      edm::InputTag("allTrackstersToSimTrackstersAssociationsByLCs", "ticlSimTrackstersfromCPsToticlTrackstersMerge"));
  desc.add<edm::FileInPath>("cummatbudinxo", edm::FileInPath("Validation/HGCalValidation/data/D41.cumulative.xo"));
  desc.add<edm::InputTag>("label_cp_effic", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("label_cp_fake", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("label_scl", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("simVertices", edm::InputTag("g4SimHits"));
  desc.add<std::vector<edm::InputTag>>("LayerClustersInputMask",
                                       {
                                           edm::InputTag("ticlTrackstersCLUE3DHigh"),
                                           edm::InputTag("ticlSimTracksters", "fromCPs"),
                                           edm::InputTag("ticlSimTracksters"),
                                       });
  desc.add<int>("totallayers_to_monitor", 52);
  desc.add<std::vector<int>>("thicknesses_to_monitor",
                             {
                                 120,
                                 200,
                                 300,
                                 -1,
                             });
  desc.add<std::string>("dirName", "HGCAL/HGCalValidator/");
  desc.addUntracked<bool>("isticlv5", false);
  descriptions.add("hgcalValidator", desc);
}
