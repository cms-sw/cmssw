#include "Validation/EcalClusters/interface/EgammaBasicClusters.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/EmptyGroupDescription.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/MakerMacros.h"

EgammaBasicClusters::EgammaBasicClusters(const edm::ParameterSet &ps)
    : enableEndcaps_{ps.getParameter<bool>("enableEndcaps")},
      barrelBasicClusterCollection_{
          consumes<reco::BasicClusterCollection>(ps.getParameter<edm::InputTag>("barrelBasicClusterCollection"))},
      endcapBasicClusterCollection_{enableEndcaps_ ? consumes<reco::BasicClusterCollection>(
                                                         ps.getParameter<edm::InputTag>("endcapBasicClusterCollection"))
                                                   : edm::EDGetTokenT<reco::BasicClusterCollection>{}},
      hsSize_(ps, "Size"),
      hsNumRecHits_(ps, "NumRecHits"),
      hsET_(ps, "ET"),
      hsEta_(ps, "Eta"),
      hsPhi_(ps, "Phi"),
      hsR_(ps, "R"),
      hist_EB_BC_Size_(nullptr),
      hist_EE_BC_Size_(nullptr),
      hist_EB_BC_NumRecHits_(nullptr),
      hist_EE_BC_NumRecHits_(nullptr),
      hist_EB_BC_ET_(nullptr),
      hist_EE_BC_ET_(nullptr),
      hist_EB_BC_Eta_(nullptr),
      hist_EE_BC_Eta_(nullptr),
      hist_EB_BC_Phi_(nullptr),
      hist_EE_BC_Phi_(nullptr),
      hist_EB_BC_ET_vs_Eta_(nullptr),
      hist_EB_BC_ET_vs_Phi_(nullptr),
      hist_EE_BC_ET_vs_Eta_(nullptr),
      hist_EE_BC_ET_vs_Phi_(nullptr),
      hist_EE_BC_ET_vs_R_(nullptr) {}

EgammaBasicClusters::~EgammaBasicClusters() {}

void EgammaBasicClusters::fillDescriptions(edm::ConfigurationDescriptions &confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("barrelBasicClusterCollection",
                          edm::InputTag("hybridSuperClusters", "hybridBarrelBasicClusters"));
  desc.ifValue(edm::ParameterDescription<bool>("enableEndcaps", true, true),
               true >> edm::ParameterDescription<edm::InputTag>(
                           "endcapBasicClusterCollection",
                           edm::InputTag("multi5x5SuperClusters", "multi5x5EndcapBasicClusters"),
                           true) or
                   false >> edm::EmptyGroupDescription());
  desc.add<int>("hist_bins_Size", 20);
  desc.add<double>("hist_min_Size", 0.);
  desc.add<double>("hist_max_Size", 20.);
  desc.add<int>("hist_bins_Phi", 181);
  desc.add<double>("hist_min_Phi", -3.14159);
  desc.add<double>("hist_max_Phi", 3.14159);
  desc.add<int>("hist_bins_Eta", 91);
  desc.add<double>("hist_min_Eta", -2.5);
  desc.add<double>("hist_max_Eta", 2.5);
  desc.add<int>("hist_bins_ET", 200);
  desc.add<double>("hist_min_ET", 0.);
  desc.add<double>("hist_max_ET", 200.);
  desc.add<int>("hist_bins_NumRecHits", 50);
  desc.add<double>("hist_min_NumRecHits", 0.);
  desc.add<double>("hist_max_NumRecHits", 50.);
  desc.add<int>("hist_bins_R", 55);
  desc.add<double>("hist_min_R", 0.);
  desc.add<double>("hist_max_R", 175.);
  confDesc.add("egammaBasicClusterAnalyzer", desc);
}

void EgammaBasicClusters::bookHistograms(DQMStore::IBooker &_ibooker, edm::Run const &, edm::EventSetup const &) {
  _ibooker.setCurrentFolder("EcalClusterV/EcalBasicClusters/");

  hist_EB_BC_Size_ =
      _ibooker.book1D("hist_EB_BC_Size_", "# Basic Clusters in Barrel", hsSize_.bins, hsSize_.min, hsSize_.max);
  hist_EB_BC_NumRecHits_ = _ibooker.book1D("hist_EB_BC_NumRecHits_",
                                           "# of RecHits in Basic Clusters in Barrel",
                                           hsNumRecHits_.bins,
                                           hsNumRecHits_.min,
                                           hsNumRecHits_.max);

  if (enableEndcaps_) {
    hist_EE_BC_Size_ =
        _ibooker.book1D("hist_EE_BC_Size_", "# Basic Clusters in Endcap", hsSize_.bins, hsSize_.min, hsSize_.max);
    hist_EE_BC_NumRecHits_ = _ibooker.book1D("hist_EE_BC_NumRecHits_",
                                             "# of RecHits in Basic Clusters in Endcap",
                                             hsNumRecHits_.bins,
                                             hsNumRecHits_.min,
                                             hsNumRecHits_.max);
  }

  hist_EB_BC_ET_ =
      _ibooker.book1D("hist_EB_BC_ET_", "ET of Basic Clusters in Barrel", hsET_.bins, hsET_.min, hsET_.max);
  hist_EB_BC_Eta_ =
      _ibooker.book1D("hist_EB_BC_Eta_", "Eta of Basic Clusters in Barrel", hsEta_.bins, hsEta_.min, hsEta_.max);
  hist_EB_BC_Phi_ =
      _ibooker.book1D("hist_EB_BC_Phi_", "Phi of Basic Clusters in Barrel", hsPhi_.bins, hsPhi_.min, hsPhi_.max);

  if (enableEndcaps_) {
    hist_EE_BC_ET_ =
        _ibooker.book1D("hist_EE_BC_ET_", "ET of Basic Clusters in Endcap", hsET_.bins, hsET_.min, hsET_.max);

    hist_EE_BC_Eta_ =
        _ibooker.book1D("hist_EE_BC_Eta_", "Eta of Basic Clusters in Endcap", hsEta_.bins, hsEta_.min, hsEta_.max);

    hist_EE_BC_Phi_ =
        _ibooker.book1D("hist_EE_BC_Phi_", "Phi of Basic Clusters in Endcap", hsPhi_.bins, hsPhi_.min, hsPhi_.max);
  }

  hist_EB_BC_ET_vs_Eta_ = _ibooker.book2D("hist_EB_BC_ET_vs_Eta_",
                                          "Basic Cluster ET versus Eta in Barrel",
                                          hsET_.bins,
                                          hsET_.min,
                                          hsET_.max,
                                          hsEta_.bins,
                                          hsEta_.min,
                                          hsEta_.max);

  hist_EB_BC_ET_vs_Phi_ = _ibooker.book2D("hist_EB_BC_ET_vs_Phi_",
                                          "Basic Cluster ET versus Phi in Barrel",
                                          hsET_.bins,
                                          hsET_.min,
                                          hsET_.max,
                                          hsPhi_.bins,
                                          hsPhi_.min,
                                          hsPhi_.max);

  if (enableEndcaps_) {
    hist_EE_BC_ET_vs_Eta_ = _ibooker.book2D("hist_EE_BC_ET_vs_Eta_",
                                            "Basic Cluster ET versus Eta in Endcap",
                                            hsET_.bins,
                                            hsET_.min,
                                            hsET_.max,
                                            hsEta_.bins,
                                            hsEta_.min,
                                            hsEta_.max);

    hist_EE_BC_ET_vs_Phi_ = _ibooker.book2D("hist_EE_BC_ET_vs_Phi_",
                                            "Basic Cluster ET versus Phi in Endcap",
                                            hsET_.bins,
                                            hsET_.min,
                                            hsET_.max,
                                            hsPhi_.bins,
                                            hsPhi_.min,
                                            hsPhi_.max);

    hist_EE_BC_ET_vs_R_ = _ibooker.book2D("hist_EE_BC_ET_vs_R_",
                                          "Basic Cluster ET versus Radius in Endcap",
                                          hsET_.bins,
                                          hsET_.min,
                                          hsET_.max,
                                          hsR_.bins,
                                          hsR_.min,
                                          hsR_.max);
  }
}

void EgammaBasicClusters::analyze(const edm::Event &evt, const edm::EventSetup &) {
  edm::Handle<reco::BasicClusterCollection> pBarrelBasicClusters;
  evt.getByToken(barrelBasicClusterCollection_, pBarrelBasicClusters);
  if (!pBarrelBasicClusters.isValid()) {
    Labels l;
    labelsForToken(barrelBasicClusterCollection_, l);
    edm::LogError("EgammaBasicClusters") << "Error! can't get collection with label " << l.module;
  }

  const reco::BasicClusterCollection *barrelBasicClusters = pBarrelBasicClusters.product();
  hist_EB_BC_Size_->Fill(barrelBasicClusters->size());

  for (reco::BasicClusterCollection::const_iterator aClus = barrelBasicClusters->begin();
       aClus != barrelBasicClusters->end();
       aClus++) {
    hist_EB_BC_NumRecHits_->Fill(aClus->size());
    hist_EB_BC_ET_->Fill(aClus->energy() / std::cosh(aClus->position().eta()));
    hist_EB_BC_Eta_->Fill(aClus->position().eta());
    hist_EB_BC_Phi_->Fill(aClus->position().phi());

    hist_EB_BC_ET_vs_Eta_->Fill(aClus->energy() / std::cosh(aClus->position().eta()), aClus->eta());
    hist_EB_BC_ET_vs_Phi_->Fill(aClus->energy() / std::cosh(aClus->position().eta()), aClus->phi());
  }

  if (enableEndcaps_) {
    edm::Handle<reco::BasicClusterCollection> pEndcapBasicClusters;

    evt.getByToken(endcapBasicClusterCollection_, pEndcapBasicClusters);
    if (!pEndcapBasicClusters.isValid()) {
      Labels l;
      labelsForToken(endcapBasicClusterCollection_, l);
      edm::LogError("EgammaBasicClusters") << "Error! can't get collection with label " << l.module;
    }

    const reco::BasicClusterCollection *endcapBasicClusters = pEndcapBasicClusters.product();
    hist_EE_BC_Size_->Fill(endcapBasicClusters->size());

    for (reco::BasicClusterCollection::const_iterator aClus = endcapBasicClusters->begin();
         aClus != endcapBasicClusters->end();
         aClus++) {
      hist_EE_BC_NumRecHits_->Fill(aClus->size());
      hist_EE_BC_ET_->Fill(aClus->energy() / std::cosh(aClus->position().eta()));
      hist_EE_BC_Eta_->Fill(aClus->position().eta());
      hist_EE_BC_Phi_->Fill(aClus->position().phi());

      hist_EE_BC_ET_vs_Eta_->Fill(aClus->energy() / std::cosh(aClus->position().eta()), aClus->eta());
      hist_EE_BC_ET_vs_Phi_->Fill(aClus->energy() / std::cosh(aClus->position().eta()), aClus->phi());
      hist_EE_BC_ET_vs_R_->Fill(aClus->energy() / std::cosh(aClus->position().eta()),
                                std::sqrt(std::pow(aClus->x(), 2) + std::pow(aClus->y(), 2)));
    }
  }
}
