#include "Validation/EcalClusters/interface/EgammaBasicClusters.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/MakerMacros.h"

EgammaBasicClusters::EgammaBasicClusters(const edm::ParameterSet &ps)
    : barrelBasicClusterCollection_(
          consumes<reco::BasicClusterCollection>(ps.getParameter<edm::InputTag>("barrelBasicClusterCollection"))),
      endcapBasicClusterCollection_(
          consumes<reco::BasicClusterCollection>(ps.getParameter<edm::InputTag>("endcapBasicClusterCollection"))),
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

void EgammaBasicClusters::bookHistograms(DQMStore::IBooker &_ibooker, edm::Run const &, edm::EventSetup const &) {
  _ibooker.setCurrentFolder("EcalClusterV/EcalBasicClusters/");

  hist_EB_BC_Size_ =
      _ibooker.book1D("hist_EB_BC_Size_", "# Basic Clusters in Barrel", hsSize_.bins, hsSize_.min, hsSize_.max);
  hist_EE_BC_Size_ =
      _ibooker.book1D("hist_EE_BC_Size_", "# Basic Clusters in Endcap", hsSize_.bins, hsSize_.min, hsSize_.max);

  hist_EB_BC_NumRecHits_ = _ibooker.book1D("hist_EB_BC_NumRecHits_",
                                           "# of RecHits in Basic Clusters in Barrel",
                                           hsNumRecHits_.bins,
                                           hsNumRecHits_.min,
                                           hsNumRecHits_.max);
  hist_EE_BC_NumRecHits_ = _ibooker.book1D("hist_EE_BC_NumRecHits_",
                                           "# of RecHits in Basic Clusters in Endcap",
                                           hsNumRecHits_.bins,
                                           hsNumRecHits_.min,
                                           hsNumRecHits_.max);

  hist_EB_BC_ET_ =
      _ibooker.book1D("hist_EB_BC_ET_", "ET of Basic Clusters in Barrel", hsET_.bins, hsET_.min, hsET_.max);
  hist_EE_BC_ET_ =
      _ibooker.book1D("hist_EE_BC_ET_", "ET of Basic Clusters in Endcap", hsET_.bins, hsET_.min, hsET_.max);

  hist_EB_BC_Eta_ =
      _ibooker.book1D("hist_EB_BC_Eta_", "Eta of Basic Clusters in Barrel", hsEta_.bins, hsEta_.min, hsEta_.max);
  hist_EE_BC_Eta_ =
      _ibooker.book1D("hist_EE_BC_Eta_", "Eta of Basic Clusters in Endcap", hsEta_.bins, hsEta_.min, hsEta_.max);

  hist_EB_BC_Phi_ =
      _ibooker.book1D("hist_EB_BC_Phi_", "Phi of Basic Clusters in Barrel", hsPhi_.bins, hsPhi_.min, hsPhi_.max);
  hist_EE_BC_Phi_ =
      _ibooker.book1D("hist_EE_BC_Phi_", "Phi of Basic Clusters in Endcap", hsPhi_.bins, hsPhi_.min, hsPhi_.max);

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

  for (const auto &barrelBasicCluster : *barrelBasicClusters) {
    hist_EB_BC_NumRecHits_->Fill(barrelBasicCluster.size());
    hist_EB_BC_ET_->Fill(barrelBasicCluster.energy() / std::cosh(barrelBasicCluster.position().eta()));
    hist_EB_BC_Eta_->Fill(barrelBasicCluster.position().eta());
    hist_EB_BC_Phi_->Fill(barrelBasicCluster.position().phi());

    hist_EB_BC_ET_vs_Eta_->Fill(barrelBasicCluster.energy() / std::cosh(barrelBasicCluster.position().eta()),
                                barrelBasicCluster.eta());
    hist_EB_BC_ET_vs_Phi_->Fill(barrelBasicCluster.energy() / std::cosh(barrelBasicCluster.position().eta()),
                                barrelBasicCluster.phi());
  }

  edm::Handle<reco::BasicClusterCollection> pEndcapBasicClusters;

  evt.getByToken(endcapBasicClusterCollection_, pEndcapBasicClusters);
  if (!pEndcapBasicClusters.isValid()) {
    Labels l;
    labelsForToken(endcapBasicClusterCollection_, l);
    edm::LogError("EgammaBasicClusters") << "Error! can't get collection with label " << l.module;
  }

  const reco::BasicClusterCollection *endcapBasicClusters = pEndcapBasicClusters.product();
  hist_EE_BC_Size_->Fill(endcapBasicClusters->size());

  for (const auto &endcapBasicCluster : *endcapBasicClusters) {
    hist_EE_BC_NumRecHits_->Fill(endcapBasicCluster.size());
    hist_EE_BC_ET_->Fill(endcapBasicCluster.energy() / std::cosh(endcapBasicCluster.position().eta()));
    hist_EE_BC_Eta_->Fill(endcapBasicCluster.position().eta());
    hist_EE_BC_Phi_->Fill(endcapBasicCluster.position().phi());

    hist_EE_BC_ET_vs_Eta_->Fill(endcapBasicCluster.energy() / std::cosh(endcapBasicCluster.position().eta()),
                                endcapBasicCluster.eta());
    hist_EE_BC_ET_vs_Phi_->Fill(endcapBasicCluster.energy() / std::cosh(endcapBasicCluster.position().eta()),
                                endcapBasicCluster.phi());
    hist_EE_BC_ET_vs_R_->Fill(endcapBasicCluster.energy() / std::cosh(endcapBasicCluster.position().eta()),
                              std::sqrt(std::pow(endcapBasicCluster.x(), 2) + std::pow(endcapBasicCluster.y(), 2)));
  }
}
