#include "Validation/EcalClusters/interface/EgammaBasicClusters.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/MakerMacros.h"

EgammaBasicClusters::EgammaBasicClusters( const edm::ParameterSet& ps ) :
  barrelBasicClusterCollection_(consumes<reco::BasicClusterCollection>(ps.getParameter<edm::InputTag>("barrelBasicClusterCollection"))),
  endcapBasicClusterCollection_(consumes<reco::BasicClusterCollection>(ps.getParameter<edm::InputTag>("endcapBasicClusterCollection"))),
  hsSize_(ps, "Size"),
  hsNumRecHits_(ps, "NumRecHits"),
  hsET_(ps, "ET"),
  hsEta_(ps, "Eta"),
  hsPhi_(ps, "Phi"),
  hsR_(ps, "R"),
  hist_EB_BC_Size_(0),
  hist_EE_BC_Size_(0),
  hist_EB_BC_NumRecHits_(0),
  hist_EE_BC_NumRecHits_(0),
  hist_EB_BC_ET_(0),
  hist_EE_BC_ET_(0),
  hist_EB_BC_Eta_(0),
  hist_EE_BC_Eta_(0),
  hist_EB_BC_Phi_(0),
  hist_EE_BC_Phi_(0),
  hist_EB_BC_ET_vs_Eta_(0),
  hist_EB_BC_ET_vs_Phi_(0),
  hist_EE_BC_ET_vs_Eta_(0),
  hist_EE_BC_ET_vs_Phi_(0),
  hist_EE_BC_ET_vs_R_(0)
{
}

EgammaBasicClusters::~EgammaBasicClusters()
{
}

void
EgammaBasicClusters::bookHistograms(DQMStore::IBooker& _ibooker, edm::Run const&, edm::EventSetup const&)
{
  _ibooker.setCurrentFolder("EcalClusterV/EcalBasicClusters/");

  hist_EB_BC_Size_ 
    = _ibooker.book1D("hist_EB_BC_Size_","# Basic Clusters in Barrel",
                      hsSize_.bins, hsSize_.min, hsSize_.max);
  hist_EE_BC_Size_ 
    = _ibooker.book1D("hist_EE_BC_Size_","# Basic Clusters in Endcap",
                      hsSize_.bins, hsSize_.min, hsSize_.max);

  hist_EB_BC_NumRecHits_ 
    = _ibooker.book1D("hist_EB_BC_NumRecHits_","# of RecHits in Basic Clusters in Barrel",
                      hsNumRecHits_.bins, hsNumRecHits_.min, hsNumRecHits_.max);
  hist_EE_BC_NumRecHits_ 
    = _ibooker.book1D("hist_EE_BC_NumRecHits_","# of RecHits in Basic Clusters in Endcap",
                      hsNumRecHits_.bins, hsNumRecHits_.min, hsNumRecHits_.max);

  hist_EB_BC_ET_ 
    = _ibooker.book1D("hist_EB_BC_ET_","ET of Basic Clusters in Barrel",
                      hsET_.bins, hsET_.min, hsET_.max);
  hist_EE_BC_ET_ 
    = _ibooker.book1D("hist_EE_BC_ET_","ET of Basic Clusters in Endcap",
                      hsET_.bins, hsET_.min, hsET_.max);

  hist_EB_BC_Eta_ 
    = _ibooker.book1D("hist_EB_BC_Eta_","Eta of Basic Clusters in Barrel",
                      hsEta_.bins, hsEta_.min, hsEta_.max);
  hist_EE_BC_Eta_ 
    = _ibooker.book1D("hist_EE_BC_Eta_","Eta of Basic Clusters in Endcap",
                      hsEta_.bins, hsEta_.min, hsEta_.max);

  hist_EB_BC_Phi_
    = _ibooker.book1D("hist_EB_BC_Phi_","Phi of Basic Clusters in Barrel",
                      hsPhi_.bins, hsPhi_.min, hsPhi_.max);
  hist_EE_BC_Phi_ 
    = _ibooker.book1D("hist_EE_BC_Phi_","Phi of Basic Clusters in Endcap",
                      hsPhi_.bins, hsPhi_.min, hsPhi_.max);

	
  hist_EB_BC_ET_vs_Eta_
    = _ibooker.book2D( "hist_EB_BC_ET_vs_Eta_", "Basic Cluster ET versus Eta in Barrel", 
                       hsET_.bins, hsET_.min, hsET_.max,
                       hsEta_.bins, hsEta_.min, hsEta_.max );

  hist_EB_BC_ET_vs_Phi_
    = _ibooker.book2D( "hist_EB_BC_ET_vs_Phi_", "Basic Cluster ET versus Phi in Barrel", 
                       hsET_.bins, hsET_.min, hsET_.max,
                       hsPhi_.bins, hsPhi_.min, hsPhi_.max );

  hist_EE_BC_ET_vs_Eta_
    = _ibooker.book2D( "hist_EE_BC_ET_vs_Eta_", "Basic Cluster ET versus Eta in Endcap", 
                       hsET_.bins, hsET_.min, hsET_.max,
                       hsEta_.bins, hsEta_.min, hsEta_.max );

  hist_EE_BC_ET_vs_Phi_
    = _ibooker.book2D( "hist_EE_BC_ET_vs_Phi_", "Basic Cluster ET versus Phi in Endcap", 
                       hsET_.bins, hsET_.min, hsET_.max,
                       hsPhi_.bins, hsPhi_.min, hsPhi_.max );

  hist_EE_BC_ET_vs_R_
    = _ibooker.book2D( "hist_EE_BC_ET_vs_R_", "Basic Cluster ET versus Radius in Endcap", 
                       hsET_.bins, hsET_.min, hsET_.max,
                       hsR_.bins, hsR_.min, hsR_.max );
}

void
EgammaBasicClusters::analyze( const edm::Event& evt, const edm::EventSetup&)
{
  edm::Handle<reco::BasicClusterCollection> pBarrelBasicClusters;
  evt.getByToken(barrelBasicClusterCollection_, pBarrelBasicClusters);
  if (!pBarrelBasicClusters.isValid()) {

    Labels l;
    labelsForToken(barrelBasicClusterCollection_,l);
    edm::LogError("EgammaBasicClusters") << "Error! can't get collection with label " 
                                         << l.module;
  }

  const reco::BasicClusterCollection* barrelBasicClusters = pBarrelBasicClusters.product();
  hist_EB_BC_Size_->Fill(barrelBasicClusters->size());

  for(reco::BasicClusterCollection::const_iterator aClus = barrelBasicClusters->begin(); 
      aClus != barrelBasicClusters->end(); aClus++)
    {
      hist_EB_BC_NumRecHits_->Fill(aClus->size());
      hist_EB_BC_ET_->Fill(aClus->energy()/std::cosh(aClus->position().eta()));
      hist_EB_BC_Eta_->Fill(aClus->position().eta());
      hist_EB_BC_Phi_->Fill(aClus->position().phi());
		
      hist_EB_BC_ET_vs_Eta_->Fill( aClus->energy()/std::cosh(aClus->position().eta()), aClus->eta() );
      hist_EB_BC_ET_vs_Phi_->Fill( aClus->energy()/std::cosh(aClus->position().eta()), aClus->phi() );
    }

  edm::Handle<reco::BasicClusterCollection> pEndcapBasicClusters;

  evt.getByToken(endcapBasicClusterCollection_, pEndcapBasicClusters);
  if (!pEndcapBasicClusters.isValid()) {

    Labels l;
    labelsForToken(endcapBasicClusterCollection_,l);
    edm::LogError("EgammaBasicClusters") << "Error! can't get collection with label " 
                                         << l.module;
  }

  const reco::BasicClusterCollection* endcapBasicClusters = pEndcapBasicClusters.product();
  hist_EE_BC_Size_->Fill(endcapBasicClusters->size());

  for(reco::BasicClusterCollection::const_iterator aClus = endcapBasicClusters->begin(); 
      aClus != endcapBasicClusters->end(); aClus++)
    {
      hist_EE_BC_NumRecHits_->Fill(aClus->size());
      hist_EE_BC_ET_->Fill(aClus->energy()/std::cosh(aClus->position().eta()));
      hist_EE_BC_Eta_->Fill(aClus->position().eta());
      hist_EE_BC_Phi_->Fill(aClus->position().phi());

      hist_EE_BC_ET_vs_Eta_->Fill( aClus->energy()/std::cosh(aClus->position().eta()), aClus->eta() );
      hist_EE_BC_ET_vs_Phi_->Fill( aClus->energy()/std::cosh(aClus->position().eta()), aClus->phi() );
      hist_EE_BC_ET_vs_R_->Fill( aClus->energy()/std::cosh(aClus->position().eta()), 
                                 std::sqrt( std::pow(aClus->x(),2) + std::pow(aClus->y(),2) ) );

    }
}
