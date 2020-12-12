#include "Validation/HGCalValidation/interface/HGCalValidator.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace edm;

HGCalValidator::HGCalValidator(const edm::ParameterSet& pset)
    : label_lcl(pset.getParameter<edm::InputTag>("label_lcl")),
      label_mcl(pset.getParameter<std::vector<edm::InputTag>>("label_mcl")),
      associator_(pset.getUntrackedParameter<edm::InputTag>("associator")),
      SaveGeneralInfo_(pset.getUntrackedParameter<bool>("SaveGeneralInfo")),
      doCaloParticlePlots_(pset.getUntrackedParameter<bool>("doCaloParticlePlots")),
      doCaloParticleSelection_(pset.getUntrackedParameter<bool>("doCaloParticleSelection")),
      dolayerclustersPlots_(pset.getUntrackedParameter<bool>("dolayerclustersPlots")),
      domulticlustersPlots_(pset.getUntrackedParameter<bool>("domulticlustersPlots")),
      cummatbudinxo_(pset.getParameter<edm::FileInPath>("cummatbudinxo")) {
  //In this way we can easily generalize to associations between other objects also.
  const edm::InputTag& label_cp_effic_tag = pset.getParameter<edm::InputTag>("label_cp_effic");
  const edm::InputTag& label_cp_fake_tag = pset.getParameter<edm::InputTag>("label_cp_fake");

  label_cp_effic = consumes<std::vector<CaloParticle>>(label_cp_effic_tag);
  label_cp_fake = consumes<std::vector<CaloParticle>>(label_cp_fake_tag);

  simVertices_ = consumes<std::vector<SimVertex>>(pset.getParameter<edm::InputTag>("simVertices"));

  hitMap_ = consumes<std::unordered_map<DetId, const HGCRecHit*>>(edm::InputTag("hgcalRecHitMapProducer"));

  density_ = consumes<Density>(edm::InputTag("hgcalLayerClusters"));

  layerclusters_ = consumes<reco::CaloClusterCollection>(label_lcl);

  for (auto& itag : label_mcl) {
    label_mclTokens.push_back(consumes<std::vector<reco::HGCalMultiCluster>>(itag));
  }

  associatorMapRtS = consumes<hgcal::RecoToSimCollection>(associator_);
  associatorMapStR = consumes<hgcal::SimToRecoCollection>(associator_);

  cpSelector = CaloParticleSelector(pset.getParameter<double>("ptMinCP"),
                                    pset.getParameter<double>("ptMaxCP"),
                                    pset.getParameter<double>("minRapidityCP"),
                                    pset.getParameter<double>("maxRapidityCP"),
                                    pset.getParameter<int>("minHitCP"),
                                    pset.getParameter<int>("maxSimClustersCP"),
                                    pset.getParameter<double>("tipCP"),
                                    pset.getParameter<double>("lipCP"),
                                    pset.getParameter<bool>("signalOnlyCP"),
                                    pset.getParameter<bool>("intimeOnlyCP"),
                                    pset.getParameter<bool>("chargedOnlyCP"),
                                    pset.getParameter<bool>("stableOnlyCP"),
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
    cummatbudg.insert(std::pair<double, double>(thelay, mbg));
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

  //Booking histograms concerning with hgcal layer clusters
  if (dolayerclustersPlots_) {
    ibook.cd();
    ibook.setCurrentFolder(dirName_ + "hgcalLayerClusters");
    histoProducerAlgo_->bookClusterHistos(ibook,
                                          histograms.histoProducerAlgo,
                                          totallayers_to_monitor_,
                                          thicknesses_to_monitor_,
                                          cummatbudinxo_.fullPath());
  }

  //Booking histograms for multiclusters
  for (unsigned int www = 0; www < label_mcl.size(); www++) {
    ibook.cd();
    InputTag algo = label_mcl[www];
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

    //Booking histograms concerning for hgcal multi clusters
    if (domulticlustersPlots_) {
      histoProducerAlgo_->bookMultiClusterHistos(ibook, histograms.histoProducerAlgo, totallayers_to_monitor_);
    }
  }  //end of booking multiclusters loop
}

void HGCalValidator::cpParametersAndSelection(const Histograms& histograms,
                                              std::vector<CaloParticle> const& cPeff,
                                              std::vector<SimVertex> const& simVertices,
                                              std::vector<size_t>& selected_cPeff,
                                              unsigned int layers,
                                              std::unordered_map<DetId, const HGCRecHit*> const& hitMap) const {
  selected_cPeff.reserve(cPeff.size());

  size_t j = 0;
  for (auto const& caloParticle : cPeff) {
    int id = caloParticle.pdgId();

    if (!doCaloParticleSelection_ || (doCaloParticleSelection_ && cpSelector(caloParticle, simVertices))) {
      selected_cPeff.push_back(j);
      if (doCaloParticlePlots_) {
        histoProducerAlgo_->fill_caloparticle_histos(
            histograms.histoProducerAlgo, id, caloParticle, simVertices, layers, hitMap);
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

  edm::ESHandle<CaloGeometry> geom;
  setup.get<CaloGeometryRecord>().get(geom);
  tools_->setGeometry(*geom);
  histoProducerAlgo_->setRecHitTools(tools_);

  edm::Handle<hgcal::SimToRecoCollection> simtorecoCollectionH;
  event.getByToken(associatorMapStR, simtorecoCollectionH);
  auto simRecColl = *simtorecoCollectionH;
  edm::Handle<hgcal::RecoToSimCollection> recotosimCollectionH;
  event.getByToken(associatorMapRtS, recotosimCollectionH);
  auto recSimColl = *recotosimCollectionH;

  edm::Handle<std::unordered_map<DetId, const HGCRecHit*>> hitMapHandle;
  event.getByToken(hitMap_, hitMapHandle);
  const std::unordered_map<DetId, const HGCRecHit*>* hitMap = &*hitMapHandle;

  //Some general info on layers etc.
  if (SaveGeneralInfo_) {
    histoProducerAlgo_->fill_info_histos(histograms.histoProducerAlgo, totallayers_to_monitor_);
  }

  auto nCaloParticles = caloParticles.size();
  std::vector<size_t> cPIndices;
  //Consider CaloParticles coming from the hard scatterer
  //excluding the PU contribution and save the indices.
  for (unsigned int cpId = 0; cpId < nCaloParticles; ++cpId) {
    if (caloParticles[cpId].g4Tracks()[0].eventId().event() != 0 or
        caloParticles[cpId].g4Tracks()[0].eventId().bunchCrossing() != 0) {
      LogDebug("HGCalValidator") << "Excluding CaloParticles from event: "
                                 << caloParticles[cpId].g4Tracks()[0].eventId().event()
                                 << " with BX: " << caloParticles[cpId].g4Tracks()[0].eventId().bunchCrossing()
                                 << std::endl;
      continue;
    }
    cPIndices.emplace_back(cpId);
  }

  // ##############################################
  // fill caloparticles histograms
  // ##############################################
  // HGCRecHit are given to select the SimHits which are also reconstructed
  LogTrace("HGCalValidator") << "\n# of CaloParticles: " << caloParticles.size() << "\n" << std::endl;
  std::vector<size_t> selected_cPeff;
  cpParametersAndSelection(histograms, caloParticles, simVertices, selected_cPeff, totallayers_to_monitor_, *hitMap);

  //get collections from the event
  //Layer clusters
  edm::Handle<reco::CaloClusterCollection> clusterHandle;
  event.getByToken(layerclusters_, clusterHandle);
  const reco::CaloClusterCollection& clusters = *clusterHandle;

  //Density
  edm::Handle<Density> densityHandle;
  event.getByToken(density_, densityHandle);
  const Density& densities = *densityHandle;

  // ##############################################
  // fill layercluster histograms
  // ##############################################
  int w = 0;  //counter counting the number of sets of histograms
  if (dolayerclustersPlots_) {
    histoProducerAlgo_->fill_generic_cluster_histos(histograms.histoProducerAlgo,
                                                    w,
                                                    clusterHandle,
                                                    clusters,
                                                    densities,
                                                    caloParticleHandle,
                                                    caloParticles,
                                                    cPIndices,
                                                    selected_cPeff,
                                                    *hitMap,
                                                    cummatbudg,
                                                    totallayers_to_monitor_,
                                                    thicknesses_to_monitor_,
                                                    recSimColl,
                                                    simRecColl);

    for (unsigned int layerclusterIndex = 0; layerclusterIndex < clusters.size(); layerclusterIndex++) {
      histoProducerAlgo_->fill_cluster_histos(histograms.histoProducerAlgo, w, clusters[layerclusterIndex]);
    }

    //General Info on hgcalLayerClusters
    LogTrace("HGCalValidator") << "\n# of layer clusters with " << label_lcl.process() << ":" << label_lcl.label()
                               << ":" << label_lcl.instance() << ": " << clusters.size() << "\n";
  }

  // ##############################################
  // fill multicluster histograms
  // ##############################################
  for (unsigned int wml = 0; wml < label_mclTokens.size(); wml++) {
    if (domulticlustersPlots_) {
      edm::Handle<std::vector<reco::HGCalMultiCluster>> multiClusterHandle;
      event.getByToken(label_mclTokens[wml], multiClusterHandle);
      const std::vector<reco::HGCalMultiCluster>& multiClusters = *multiClusterHandle;

      histoProducerAlgo_->fill_multi_cluster_histos(histograms.histoProducerAlgo,
                                                    wml,
                                                    multiClusters,
                                                    caloParticles,
                                                    cPIndices,
                                                    selected_cPeff,
                                                    *hitMap,
                                                    totallayers_to_monitor_);

      //General Info on multiclusters
      LogTrace("HGCalValidator") << "\n# of multi clusters with " << label_mcl[wml].process() << ":"
                                 << label_mcl[wml].label() << ":" << label_mcl[wml].instance() << ": "
                                 << multiClusters.size() << "\n"
                                 << std::endl;
    }
  }  //end of loop over multicluster input labels
}
