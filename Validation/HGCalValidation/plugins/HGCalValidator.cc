#include "Validation/HGCalValidation/interface/HGCalValidator.h"

#include "SimCalorimetry/HGCalAssociatorProducers/interface/AssociatorTools.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace edm;

HGCalValidator::HGCalValidator(const edm::ParameterSet& pset)
    : caloGeomToken_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
      label_lcl(pset.getParameter<edm::InputTag>("label_lcl")),
      label_tst(pset.getParameter<std::vector<edm::InputTag>>("label_tst")),
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
      label_TSToCPLinking_(pset.getParameter<std::string>("label_TSToCPLinking")),
      label_TSToSTSPR_(pset.getParameter<std::string>("label_TSToSTSPR")),
      label_clustersmask(pset.getParameter<std::vector<edm::InputTag>>("LayerClustersInputMask")),
      doCandidatesPlots_(pset.getUntrackedParameter<bool>("doCandidatesPlots")),
      label_candidates_(pset.getParameter<std::string>("ticlCandidates")),
      cummatbudinxo_(pset.getParameter<edm::FileInPath>("cummatbudinxo")),
      isTICLv5_(pset.getUntrackedParameter<bool>("isticlv5")),
      hits_label_(pset.getParameter<std::vector<edm::InputTag>>("hits")) {
  //In this way we can easily generalize to associations between other objects also.
  const edm::InputTag& label_cp_effic_tag = pset.getParameter<edm::InputTag>("label_cp_effic");
  const edm::InputTag& label_cp_fake_tag = pset.getParameter<edm::InputTag>("label_cp_fake");

  for (auto& label : hits_label_) {
    hits_token_.push_back(consumes<HGCRecHitCollection>(label));
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

  if (doCandidatesPlots_) {
    edm::EDGetTokenT<std::vector<TICLCandidate>> TICLCandidatesToken =
        consumes<std::vector<TICLCandidate>>(pset.getParameter<edm::InputTag>("ticlTrackstersMerge"));
    edm::EDGetTokenT<std::vector<TICLCandidate>> simTICLCandidatesToken =
        consumes<std::vector<TICLCandidate>>(pset.getParameter<edm::InputTag>("simTiclCandidates"));
    edm::EDGetTokenT<std::vector<reco::Track>> recoTracksToken =
        consumes<std::vector<reco::Track>>(pset.getParameter<edm::InputTag>("recoTracks"));
    edm::EDGetTokenT<std::vector<ticl::Trackster>> trackstersToken =
        consumes<std::vector<ticl::Trackster>>(pset.getParameter<edm::InputTag>("ticlTrackstersMerge"));
    edm::EDGetTokenT<ticl::RecoToSimCollectionSimTracksters> associatorMapRtSToken =
        consumes<ticl::SimToRecoCollectionSimTracksters>(pset.getParameter<edm::InputTag>("mergeRecoToSimAssociator"));
    edm::EDGetTokenT<ticl::SimToRecoCollectionSimTracksters> associatorMapStRToken =
        consumes<ticl::SimToRecoCollectionSimTracksters>(pset.getParameter<edm::InputTag>("mergeSimToRecoAssociator"));
    edm::EDGetTokenT<ticl::RecoToSimCollectionSimTracksters> associatorMapRtSPUToken =
        consumes<ticl::SimToRecoCollectionSimTracksters>(
            pset.getParameter<edm::InputTag>("mergeRecoToSimAssociatorPU"));

    candidateVal_ = std::make_unique<TICLCandidateValidator>(TICLCandidatesToken,
                                                             simTICLCandidatesToken,
                                                             recoTracksToken,
                                                             trackstersToken,
                                                             associatorMapRtSToken,
                                                             associatorMapStRToken,
                                                             associatorMapRtSPUToken,
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
  }    //if for simCluster plots

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

    LogDebug("HGCalValidator") << dirName << "\n";

    ibook.setCurrentFolder(dirName);

    // Booking histograms concerning HGCal tracksters
    if (doTrackstersPlots_) {
      // Generic histos
      ibook.setCurrentFolder(dirName + "/" + label_TS_);
      histoProducerAlgo_->bookTracksterHistos(ibook, histograms.histoProducerAlgo, totallayers_to_monitor_);
      // CP Linking
      ibook.setCurrentFolder(dirName + "/" + label_TSToCPLinking_);
      histoProducerAlgo_->bookTracksterSTSHistos(
          ibook, histograms.histoProducerAlgo, HGVHistoProducerAlgo::validationType::Linking);
      // SimTracksters Pattern Recognition
      ibook.setCurrentFolder(dirName + "/" + label_TSToSTSPR_);
      histoProducerAlgo_->bookTracksterSTSHistos(
          ibook, histograms.histoProducerAlgo, HGVHistoProducerAlgo::validationType::PatternRecognition);
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
                                              std::vector<HGCRecHit> const& hits) const {
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
  const std::map<uint, std::vector<uint>> cpToSc_SimTrackstersMap = *simTrackstersMapHandle;

  edm::ESHandle<CaloGeometry> geom = setup.getHandle(caloGeomToken_);
  tools_->setGeometry(*geom);
  histoProducerAlgo_->setRecHitTools(tools_);

  edm::Handle<ticl::SimToRecoCollection> simtorecoCollectionH;
  event.getByToken(associatorMapStR, simtorecoCollectionH);
  auto simRecColl = *simtorecoCollectionH;
  edm::Handle<ticl::RecoToSimCollection> recotosimCollectionH;
  event.getByToken(associatorMapRtS, recotosimCollectionH);
  auto recSimColl = *recotosimCollectionH;

  edm::Handle<std::unordered_map<DetId, const unsigned int>> hitMapHandle;
  event.getByToken(hitMap_, hitMapHandle);
  const std::unordered_map<DetId, const unsigned int>* hitMap = &*hitMapHandle;

  std::vector<HGCRecHit> hits;
  for (auto& token : hits_token_) {
    edm::Handle<HGCRecHitCollection> hitsHandle;
    event.getByToken(token, hitsHandle);
    hits.insert(hits.end(), (*hitsHandle).begin(), (*hitsHandle).end());
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
      histograms, caloParticles, simVertices, selected_cPeff, totallayers_to_monitor_, *hitMap, hits);

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
                                                            *hitMap,
                                                            totallayers_to_monitor_,
                                                            recSimColl,
                                                            simRecColl,
                                                            hits);

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
                                                    *hitMap,
                                                    cumulative_material_budget,
                                                    totallayers_to_monitor_,
                                                    thicknesses_to_monitor_,
                                                    recSimColl,
                                                    simRecColl,
                                                    hits);

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

      //General Info on Tracksters
      LogTrace("HGCalValidator") << "\n# of Tracksters from " << label_tst[wml].process() << ":"
                                 << label_tst[wml].label() << ":" << label_tst[wml].instance() << ": "
                                 << tracksters.size() << "\n"
                                 << std::endl;

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
                                                *hitMap,
                                                totallayers_to_monitor_,
                                                hits);
    }
  }  //end of loop over Trackster input labels

  // tracksters histograms
  if (doCandidatesPlots_) {
    candidateVal_->fillCandidateHistos(event, histograms.histoTICLCandidates, simTracksterFromCPHandle);
  }
}
