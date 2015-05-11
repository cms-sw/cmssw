#include "Validation/EcalClusters/interface/EgammaSuperClusters.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/MakerMacros.h"

EgammaSuperClusters::EgammaSuperClusters( const edm::ParameterSet& ps ) :
  MCTruthCollectionToken_(consumes<edm::HepMCProduct>(ps.getParameter<edm::InputTag>("MCTruthCollection"))),
  barrelRawSuperClusterCollectionToken_(consumes<reco::SuperClusterCollection>(ps.getParameter<edm::InputTag>("barrelRawSuperClusterCollection"))),
  barrelCorSuperClusterCollectionToken_(consumes<reco::SuperClusterCollection>(ps.getParameter<edm::InputTag>("barrelCorSuperClusterCollection"))),
  endcapRawSuperClusterCollectionToken_(consumes<reco::SuperClusterCollection>(ps.getParameter<edm::InputTag>("endcapRawSuperClusterCollection"))),
  endcapPreSuperClusterCollectionToken_(consumes<reco::SuperClusterCollection>(ps.getParameter<edm::InputTag>("endcapPreSuperClusterCollection"))),
  endcapCorSuperClusterCollectionToken_(consumes<reco::SuperClusterCollection>(ps.getParameter<edm::InputTag>("endcapCorSuperClusterCollection"))),
  barrelRecHitCollectionToken_(consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("barrelRecHitCollection"))),
  endcapRecHitCollectionToken_(consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("endcapRecHitCollection"))),
  hsSize_(ps, "Size"),
  hsNumBC_(ps, "NumBC"),
  hsET_(ps, "ET"),
  hsEta_(ps, "Eta"),
  hsPhi_(ps, "Phi"),
  hsS1toS9_(ps, "S1toS9"),
  hsS25toE_(ps, "S25toE"),
  hsEoverTruth_(ps, "EoverTruth"),
  hsdeltaR_(ps, "deltaR"),
  hsphiWidth_(ps, "phiWidth"),
  hsetaWidth_(ps, "etaWidth"),
  hspreshowerE_(ps, "preshowerE"),
  hsR_(ps, "R"),
  hist_EB_RawSC_Size_(0),
  hist_EE_RawSC_Size_(0),
  hist_EB_CorSC_Size_(0),
  hist_EE_CorSC_Size_(0),
  hist_EE_PreSC_Size_(0),
  hist_EB_RawSC_NumBC_(0),
  hist_EE_RawSC_NumBC_(0),
  hist_EB_CorSC_NumBC_(0),
  hist_EE_CorSC_NumBC_(0),
  hist_EE_PreSC_NumBC_(0),
  hist_EB_RawSC_ET_(0),
  hist_EE_RawSC_ET_(0),
  hist_EB_CorSC_ET_(0),
  hist_EE_CorSC_ET_(0),
  hist_EE_PreSC_ET_(0),
  hist_EB_RawSC_Eta_(0),
  hist_EE_RawSC_Eta_(0),
  hist_EB_CorSC_Eta_(0),
  hist_EE_CorSC_Eta_(0),
  hist_EE_PreSC_Eta_(0),
  hist_EB_RawSC_Phi_(0),
  hist_EE_RawSC_Phi_(0),
  hist_EB_CorSC_Phi_(0),
  hist_EE_CorSC_Phi_(0),
  hist_EE_PreSC_Phi_(0),
  hist_EB_RawSC_S1toS9_(0),
  hist_EE_RawSC_S1toS9_(0),
  hist_EB_CorSC_S1toS9_(0),
  hist_EE_CorSC_S1toS9_(0),
  hist_EE_PreSC_S1toS9_(0),
  hist_EB_RawSC_S25toE_(0),
  hist_EE_RawSC_S25toE_(0),
  hist_EB_CorSC_S25toE_(0),
  hist_EE_CorSC_S25toE_(0),
  hist_EE_PreSC_S25toE_(0),
  hist_EB_RawSC_EoverTruth_(0),
  hist_EE_RawSC_EoverTruth_(0),
  hist_EB_CorSC_EoverTruth_(0),
  hist_EE_CorSC_EoverTruth_(0),
  hist_EE_PreSC_EoverTruth_(0),
  hist_EB_RawSC_deltaR_(0),
  hist_EE_RawSC_deltaR_(0),
  hist_EB_CorSC_deltaR_(0),
  hist_EE_CorSC_deltaR_(0),
  hist_EE_PreSC_deltaR_(0),
  hist_EE_PreSC_preshowerE_(0),
  hist_EE_CorSC_preshowerE_(0),
  hist_EE_CorSC_phiWidth_(0),
  hist_EB_CorSC_phiWidth_(0),
  hist_EE_CorSC_etaWidth_(0),
  hist_EB_CorSC_etaWidth_(0),
  hist_EB_CorSC_ET_vs_Eta_(0),
  hist_EB_CorSC_ET_vs_Phi_(0),
  hist_EE_CorSC_ET_vs_Eta_(0),
  hist_EE_CorSC_ET_vs_Phi_(0),
  hist_EE_CorSC_ET_vs_R_(0)
{
}

EgammaSuperClusters::~EgammaSuperClusters()
{
}

void
EgammaSuperClusters::bookHistograms(DQMStore::IBooker& _ibooker, edm::Run const&, edm::EventSetup const&)
{
  _ibooker.setCurrentFolder("EcalClusterV/EcalSuperClusters/");

  // Number of SuperClusters
  //
  hist_EB_RawSC_Size_ 
    = _ibooker.book1D("hist_EB_RawSC_Size_","# Raw SuperClusters in Barrel",
                      hsSize_.bins, hsSize_.min, hsSize_.max);
  hist_EE_RawSC_Size_ 
    = _ibooker.book1D("hist_EE_RawSC_Size_","# Raw SuperClusters in Endcap",
                      hsSize_.bins, hsSize_.min, hsSize_.max);
  hist_EB_CorSC_Size_
    = _ibooker.book1D("hist_EB_CorSC_Size_","# Corrected SuperClusters in Barrel",
                      hsSize_.bins, hsSize_.min, hsSize_.max);
  hist_EE_CorSC_Size_
    = _ibooker.book1D("hist_EE_CorSC_Size_","# Corrected SuperClusters in Endcap",
                      hsSize_.bins, hsSize_.min, hsSize_.max);
  hist_EE_PreSC_Size_
    = _ibooker.book1D("hist_EE_PreSC_Size_","# SuperClusters with Preshower in Endcap",
                      hsSize_.bins, hsSize_.min, hsSize_.max);
  
  // Number of BasicClusters in SuperCluster
  //
  hist_EB_RawSC_NumBC_ 
    = _ibooker.book1D("hist_EB_RawSC_NumBC_","# of Basic Clusters in Raw Super Clusters in Barrel",
                      hsNumBC_.bins, hsNumBC_.min, hsNumBC_.max);
  hist_EE_RawSC_NumBC_ 
    = _ibooker.book1D("hist_EE_RawSC_NumBC_","# of Basic Clusters in Raw Super Clusters in Endcap",
                      hsNumBC_.bins, hsNumBC_.min, hsNumBC_.max);
  hist_EB_CorSC_NumBC_
    = _ibooker.book1D("hist_EB_CorSC_NumBC_","# of Basic Clusters in Corrected SuperClusters in Barrel",
                      hsNumBC_.bins, hsNumBC_.min, hsNumBC_.max);
  hist_EE_CorSC_NumBC_
    = _ibooker.book1D("hist_EE_CorSC_NumBC_","# of Basic Clusters in Corrected SuperClusters in Endcap",
                      hsNumBC_.bins, hsNumBC_.min, hsNumBC_.max);
  hist_EE_PreSC_NumBC_
    = _ibooker.book1D("hist_EE_PreSC_NumBC_","# of Basic Clusters in SuperClusters with Preshower in Endcap",
                      hsNumBC_.bins, hsNumBC_.min, hsNumBC_.max);
  
  // ET distribution of SuperClusters
  //
  hist_EB_RawSC_ET_ 
    = _ibooker.book1D("hist_EB_RawSC_ET_","ET of Raw SuperClusters in Barrel",
                      hsET_.bins, hsET_.min, hsET_.max);
  hist_EE_RawSC_ET_ 
    = _ibooker.book1D("hist_EE_RawSC_ET_","ET of Raw SuperClusters in Endcap",
                      hsET_.bins, hsET_.min, hsET_.max);
  hist_EB_CorSC_ET_
    = _ibooker.book1D("hist_EB_CorSC_ET_","ET of Corrected SuperClusters in Barrel",
                      hsET_.bins, hsET_.min, hsET_.max);
  hist_EE_CorSC_ET_
    = _ibooker.book1D("hist_EE_CorSC_ET_","ET of Corrected SuperClusters in Endcap",
                      hsET_.bins, hsET_.min, hsET_.max);
  hist_EE_PreSC_ET_
    = _ibooker.book1D("hist_EE_PreSC_ET_","ET of SuperClusters with Preshower in Endcap",
                      hsET_.bins, hsET_.min, hsET_.max);
  
  // Eta distribution of SuperClusters
  //
  hist_EB_RawSC_Eta_ 
    = _ibooker.book1D("hist_EB_RawSC_Eta_","Eta of Raw SuperClusters in Barrel",
                      hsEta_.bins, hsEta_.min, hsEta_.max);
  hist_EE_RawSC_Eta_ 
    = _ibooker.book1D("hist_EE_RawSC_Eta_","Eta of Raw SuperClusters in Endcap",
                      hsEta_.bins, hsEta_.min, hsEta_.max);
  hist_EB_CorSC_Eta_
    = _ibooker.book1D("hist_EB_CorSC_Eta_","Eta of Corrected SuperClusters in Barrel",
                      hsEta_.bins, hsEta_.min, hsEta_.max);
  hist_EE_CorSC_Eta_
    = _ibooker.book1D("hist_EE_CorSC_Eta_","Eta of Corrected SuperClusters in Endcap",
                      hsEta_.bins, hsEta_.min, hsEta_.max);
  hist_EE_PreSC_Eta_
    = _ibooker.book1D("hist_EE_PreSC_Eta_","Eta of SuperClusters with Preshower in Endcap",
                      hsEta_.bins, hsEta_.min, hsEta_.max);
  
  // Phi distribution of SuperClusters
  //
  hist_EB_RawSC_Phi_
    = _ibooker.book1D("hist_EB_RawSC_Phi_","Phi of Raw SuperClusters in Barrel",
                      hsPhi_.bins, hsPhi_.min, hsPhi_.max);
  hist_EE_RawSC_Phi_
    = _ibooker.book1D("hist_EE_RawSC_Phi_","Phi of Raw SuperClusters in Endcap",
                      hsPhi_.bins, hsPhi_.min, hsPhi_.max);
  hist_EB_CorSC_Phi_
    = _ibooker.book1D("hist_EB_CorSC_Phi_","Phi of Corrected SuperClusters in Barrel",
                      hsPhi_.bins, hsPhi_.min, hsPhi_.max);
  hist_EE_CorSC_Phi_
    = _ibooker.book1D("hist_EE_CorSC_Phi_","Phi of Corrected SuperClusters in Endcap",
                      hsPhi_.bins, hsPhi_.min, hsPhi_.max);
  hist_EE_PreSC_Phi_
    = _ibooker.book1D("hist_EE_PreSC_Phi_","Phi of SuperClusters with Preshower in Endcap",
                      hsPhi_.bins, hsPhi_.min, hsPhi_.max);
  
  // S1/S9 distribution of SuperClusters
  //
  hist_EB_RawSC_S1toS9_ 
    = _ibooker.book1D("hist_EB_RawSC_S1toS9_","S1/S9 of Raw Super Clusters in Barrel",
                      hsS1toS9_.bins, hsS1toS9_.min, hsS1toS9_.max);
  hist_EE_RawSC_S1toS9_ 
    = _ibooker.book1D("hist_EE_RawSC_S1toS9_","S1/S9 of Raw Super Clusters in Endcap",
                      hsS1toS9_.bins, hsS1toS9_.min, hsS1toS9_.max);
  hist_EB_CorSC_S1toS9_
    = _ibooker.book1D("hist_EB_CorSC_S1toS9_","S1/S9 of Corrected SuperClusters in Barrel",
                      hsS1toS9_.bins, hsS1toS9_.min, hsS1toS9_.max);
  hist_EE_CorSC_S1toS9_
    = _ibooker.book1D("hist_EE_CorSC_S1toS9_","S1/S9 of Corrected SuperClusters in Endcap",
                      hsS1toS9_.bins, hsS1toS9_.min, hsS1toS9_.max);
  hist_EE_PreSC_S1toS9_
    = _ibooker.book1D("hist_EE_PreSC_S1toS9_","S1/S9 of SuperClusters with Preshower in Endcap",
                      hsS1toS9_.bins, hsS1toS9_.min, hsS1toS9_.max);
  
  // S25/E distribution of SuperClusters
  //
  hist_EB_RawSC_S25toE_ 
    = _ibooker.book1D("hist_EB_RawSC_S25toE_","S25/E of Raw Super Clusters in Barrel",
                      hsS25toE_.bins, hsS25toE_.min, hsS25toE_.max);
  hist_EE_RawSC_S25toE_ 
    = _ibooker.book1D("hist_EE_RawSC_S25toE_","S25/E of Raw Super Clusters in Endcap",
                      hsS25toE_.bins, hsS25toE_.min, hsS25toE_.max);
  hist_EB_CorSC_S25toE_
    = _ibooker.book1D("hist_EB_CorSC_S25toE_","S25/E of Corrected SuperClusters in Barrel",
                      hsS25toE_.bins, hsS25toE_.min, hsS25toE_.max);
  hist_EE_CorSC_S25toE_
    = _ibooker.book1D("hist_EE_CorSC_S25toE_","S25/E of Corrected SuperClusters in Endcap",
                      hsS25toE_.bins, hsS25toE_.min, hsS25toE_.max);
  hist_EE_PreSC_S25toE_
    = _ibooker.book1D("hist_EE_PreSC_S25toE_","S25/E of SuperClusters with Preshower in Endcap",
                      hsS25toE_.bins, hsS25toE_.min, hsS25toE_.max);
  
  // E/E(true) distribution of SuperClusters
  //
  hist_EB_RawSC_EoverTruth_ 
    = _ibooker.book1D("hist_EB_RawSC_EoverTruth_","E/True E of Raw SuperClusters in Barrel",	
                      hsEoverTruth_.bins, hsEoverTruth_.min, hsEoverTruth_.max);
  hist_EE_RawSC_EoverTruth_ 
    = _ibooker.book1D("hist_EE_RawSC_EoverTruth_","E/True E of Raw SuperClusters in Endcap",
                      hsEoverTruth_.bins, hsEoverTruth_.min, hsEoverTruth_.max);
  hist_EB_CorSC_EoverTruth_
    = _ibooker.book1D("hist_EB_CorSC_EoverTruth_","E/True E of Corrected SuperClusters in Barrel",
                      hsEoverTruth_.bins, hsEoverTruth_.min, hsEoverTruth_.max);
  hist_EE_CorSC_EoverTruth_
    = _ibooker.book1D("hist_EE_CorSC_EoverTruth_","E/True E of Corrected SuperClusters in Endcap",
                      hsEoverTruth_.bins, hsEoverTruth_.min, hsEoverTruth_.max);
  hist_EE_PreSC_EoverTruth_
    = _ibooker.book1D("hist_EE_PreSC_EoverTruth_","E/True E of SuperClusters with Preshower in Endcap",
                      hsEoverTruth_.bins, hsEoverTruth_.min, hsEoverTruth_.max);
  
  // dR distribution of SuperClusters from truth
  //
  hist_EB_RawSC_deltaR_ 
    = _ibooker.book1D("hist_EB_RawSC_deltaR_","dR to MC truth of Raw Super Clusters in Barrel",
                      hsdeltaR_.bins, hsdeltaR_.min, hsdeltaR_.max);
  hist_EE_RawSC_deltaR_ 
    = _ibooker.book1D("hist_EE_RawSC_deltaR_","dR to MC truth of Raw Super Clusters in Endcap",
                      hsdeltaR_.bins, hsdeltaR_.min, hsdeltaR_.max);
  hist_EB_CorSC_deltaR_
    = _ibooker.book1D("hist_EB_CorSC_deltaR_","dR to MC truth of Corrected SuperClusters in Barrel",
                      hsdeltaR_.bins, hsdeltaR_.min, hsdeltaR_.max);
  hist_EE_CorSC_deltaR_
    = _ibooker.book1D("hist_EE_CorSC_deltaR_","dR to MC truth of Corrected SuperClusters in Endcap",
                      hsdeltaR_.bins, hsdeltaR_.min, hsdeltaR_.max);
  hist_EE_PreSC_deltaR_
    = _ibooker.book1D("hist_EE_PreSC_deltaR_","dR to MC truth of SuperClusters with Preshower in Endcap",
                      hsdeltaR_.bins, hsdeltaR_.min, hsdeltaR_.max);
  
  // phi width stored in corrected SuperClusters
  hist_EB_CorSC_phiWidth_
    = _ibooker.book1D("hist_EB_CorSC_phiWidth_","phiWidth of Corrected Super Clusters in Barrel",
                      hsphiWidth_.bins, hsphiWidth_.min, hsphiWidth_.max);
  hist_EE_CorSC_phiWidth_
    = _ibooker.book1D("hist_EE_CorSC_phiWidth_","phiWidth of Corrected Super Clusters in Endcap",
                      hsphiWidth_.bins, hsphiWidth_.min, hsphiWidth_.max);
  
  // eta width stored in corrected SuperClusters
  hist_EB_CorSC_etaWidth_
    = _ibooker.book1D("hist_EB_CorSC_etaWidth_","etaWidth of Corrected Super Clusters in Barrel",
                      hsetaWidth_.bins, hsetaWidth_.min, hsetaWidth_.max);
  hist_EE_CorSC_etaWidth_
    = _ibooker.book1D("hist_EE_CorSC_etaWidth_","etaWidth of Corrected Super Clusters in Endcap",
                      hsetaWidth_.bins, hsetaWidth_.min, hsetaWidth_.max);
  
  
  // preshower energy
  hist_EE_PreSC_preshowerE_
    = _ibooker.book1D("hist_EE_PreSC_preshowerE_","preshower energy in Super Clusters with Preshower in Endcap",
                      hspreshowerE_.bins, hspreshowerE_.min, hspreshowerE_.max);
  hist_EE_CorSC_preshowerE_
    = _ibooker.book1D("hist_EE_CorSC_preshowerE_","preshower energy in Corrected Super Clusters with Preshower in Endcap",
                      hspreshowerE_.bins, hspreshowerE_.min, hspreshowerE_.max);
  
  
  //
  hist_EB_CorSC_ET_vs_Eta_
    = _ibooker.book2D( "hist_EB_CorSC_ET_vs_Eta_", "Corr Super Cluster ET versus Eta in Barrel", 
                       hsET_.bins, hsET_.min, hsET_.max,
                       hsEta_.bins, hsEta_.min, hsEta_.max);
  
  hist_EB_CorSC_ET_vs_Phi_
    = _ibooker.book2D( "hist_EB_CorSC_ET_vs_Phi_", "Corr Super Cluster ET versus Phi in Barrel", 
                       hsET_.bins, hsET_.min, hsET_.max,
                       hsPhi_.bins, hsPhi_.min, hsPhi_.max);
  
  hist_EE_CorSC_ET_vs_Eta_
    = _ibooker.book2D( "hist_EE_CorSC_ET_vs_Eta_", "Corr Super Cluster ET versus Eta in Endcap", 
                       hsET_.bins, hsET_.min, hsET_.max,
                       hsEta_.bins, hsEta_.min, hsEta_.max);
  
  hist_EE_CorSC_ET_vs_Phi_
    = _ibooker.book2D( "hist_EE_CorSC_ET_vs_Phi_", "Corr Super Cluster ET versus Phi in Endcap", 
                       hsET_.bins, hsET_.min, hsET_.max,
                       hsPhi_.bins, hsPhi_.min, hsPhi_.max);
  
  hist_EE_CorSC_ET_vs_R_
    = _ibooker.book2D( "hist_EE_CorSC_ET_vs_R_", "Corr Super Cluster ET versus Radius in Endcap", 
                       hsET_.bins, hsET_.min, hsET_.max,
                       hsR_.bins, hsR_.min, hsR_.max);
  
}

void
EgammaSuperClusters::analyze( const edm::Event& evt, const edm::EventSetup& es )
{
  
  bool skipMC = false;
  bool skipBarrel = false;
  bool skipEndcap = false;
  
  //
  // Get MCTRUTH
  //
  edm::Handle<edm::HepMCProduct> pMCTruth ;
  evt.getByToken(MCTruthCollectionToken_, pMCTruth);
  if (!pMCTruth.isValid()) {
    edm::LogError("EgammaSuperClusters") << "Error! can't get MC collection ";
    skipMC = true;
  }
  const HepMC::GenEvent* genEvent = pMCTruth->GetEvent();
  
  if( skipMC ) return;
  
  //
  // Get the BARREL products 
  //
  edm::Handle<reco::SuperClusterCollection> pBarrelRawSuperClusters;
  evt.getByToken(barrelRawSuperClusterCollectionToken_, pBarrelRawSuperClusters);
  if (!pBarrelRawSuperClusters.isValid()) {
    edm::LogError("EgammaSuperClusters") << "Error! can't get collection Raw SC";
    skipBarrel = true;
  }
  
  edm::Handle<reco::SuperClusterCollection> pBarrelCorSuperClusters;
  evt.getByToken(barrelCorSuperClusterCollectionToken_, pBarrelCorSuperClusters);
  if (!pBarrelCorSuperClusters.isValid()) {
    edm::LogError("EgammaSuperClusters") << "Error! can't get collection Cor SC";
    skipBarrel = true;
  }
  
  edm::Handle< EBRecHitCollection > pBarrelRecHitCollection;
  evt.getByToken( barrelRecHitCollectionToken_, pBarrelRecHitCollection );
  if ( ! pBarrelRecHitCollection.isValid() ) {
    skipBarrel = true;
  }
  edm::Handle< EERecHitCollection > pEndcapRecHitCollection;
  evt.getByToken( endcapRecHitCollectionToken_, pEndcapRecHitCollection );
  if ( ! pEndcapRecHitCollection.isValid() ) {
    skipEndcap = true;
  }
  
  if( skipBarrel || skipEndcap ) return;
  
  EcalClusterLazyTools lazyTool( evt, es, barrelRecHitCollectionToken_, endcapRecHitCollectionToken_ );
  
  // Get the BARREL collections        
  const reco::SuperClusterCollection* barrelRawSuperClusters = pBarrelRawSuperClusters.product();
  const reco::SuperClusterCollection* barrelCorSuperClusters = pBarrelCorSuperClusters.product();
  
  // Number of entries in collections
  hist_EB_RawSC_Size_->Fill(barrelRawSuperClusters->size());
  hist_EB_CorSC_Size_->Fill(barrelCorSuperClusters->size());
  
  // Do RAW BARREL SuperClusters
  for(reco::SuperClusterCollection::const_iterator aClus = barrelRawSuperClusters->begin(); 
      aClus != barrelRawSuperClusters->end(); aClus++)
    {
      // kinematics
      hist_EB_RawSC_NumBC_->Fill(aClus->clustersSize());
      hist_EB_RawSC_ET_->Fill(aClus->energy()/std::cosh(aClus->position().eta()));
      hist_EB_RawSC_Eta_->Fill(aClus->position().eta());
      hist_EB_RawSC_Phi_->Fill(aClus->position().phi());
      
      // cluster shape
      const reco::CaloClusterPtr seed = aClus->seed();
      hist_EB_RawSC_S1toS9_->Fill( lazyTool.eMax( *seed ) / lazyTool.e3x3( *seed ) );
      hist_EB_RawSC_S25toE_->Fill( lazyTool.e5x5( *seed ) / aClus->energy() );
      
      // truth
      double dRClosest = 999.9;
      double energyClosest = 0;
      closestMCParticle(genEvent, *aClus, dRClosest, energyClosest);
      
      if (dRClosest < 0.1)
	{
	  hist_EB_RawSC_EoverTruth_->Fill(aClus->energy()/energyClosest);		
	  hist_EB_RawSC_deltaR_->Fill(dRClosest);
	  
	}
      
    }
  
  // Do CORRECTED BARREL SuperClusters
  for(reco::SuperClusterCollection::const_iterator aClus = barrelCorSuperClusters->begin();
      aClus != barrelCorSuperClusters->end(); aClus++)
    {
      // kinematics
      hist_EB_CorSC_NumBC_->Fill(aClus->clustersSize());
      hist_EB_CorSC_ET_->Fill(aClus->energy()/std::cosh(aClus->position().eta()));
      hist_EB_CorSC_Eta_->Fill(aClus->position().eta());
      hist_EB_CorSC_Phi_->Fill(aClus->position().phi());
      
      hist_EB_CorSC_ET_vs_Eta_->Fill( aClus->energy()/std::cosh(aClus->position().eta()), aClus->eta() );
      hist_EB_CorSC_ET_vs_Phi_->Fill( aClus->energy()/std::cosh(aClus->position().eta()), aClus->phi() );
      
      
      // cluster shape
      const reco::CaloClusterPtr seed = aClus->seed();
      hist_EB_CorSC_S1toS9_->Fill( lazyTool.eMax( *seed ) / lazyTool.e3x3( *seed ) );
      hist_EB_CorSC_S25toE_->Fill( lazyTool.e5x5( *seed ) / aClus->energy() );
      
      // correction variables
      hist_EB_CorSC_phiWidth_->Fill(aClus->phiWidth());
      hist_EB_CorSC_etaWidth_->Fill(aClus->etaWidth());
      
      // truth
      double dRClosest = 999.9;
      double energyClosest = 0;
      closestMCParticle(genEvent, *aClus, dRClosest, energyClosest);
      
      if (dRClosest < 0.1)
	{
	  hist_EB_CorSC_EoverTruth_->Fill(aClus->energy()/energyClosest);
	  hist_EB_CorSC_deltaR_->Fill(dRClosest);
	  
	}
      
    }
  
  //
  // Get the ENDCAP products
  //
  edm::Handle<reco::SuperClusterCollection> pEndcapRawSuperClusters;
  evt.getByToken(endcapRawSuperClusterCollectionToken_, pEndcapRawSuperClusters);
  if (!pEndcapRawSuperClusters.isValid()) {
    edm::LogError("EgammaSuperClusters") << "Error! can't get collection Raw EE SC";
  }
  
  edm::Handle<reco::SuperClusterCollection> pEndcapPreSuperClusters;
  evt.getByToken(endcapPreSuperClusterCollectionToken_, pEndcapPreSuperClusters);
  if (!pEndcapPreSuperClusters.isValid()) {
    edm::LogError("EgammaSuperClusters") << "Error! can't get collection Pre EE SC";
  }
  
  edm::Handle<reco::SuperClusterCollection> pEndcapCorSuperClusters;
  evt.getByToken(endcapCorSuperClusterCollectionToken_, pEndcapCorSuperClusters);
  if (!pEndcapCorSuperClusters.isValid()) {
    edm::LogError("EgammaSuperClusters") << "Error! can't get collection Cor EE SC";
  }
  
  // Get the ENDCAP collections
  const reco::SuperClusterCollection* endcapRawSuperClusters = pEndcapRawSuperClusters.product();
  const reco::SuperClusterCollection* endcapPreSuperClusters = pEndcapPreSuperClusters.product();
  const reco::SuperClusterCollection* endcapCorSuperClusters = pEndcapCorSuperClusters.product();
  
  // Number of entries in collections
  hist_EE_RawSC_Size_->Fill(endcapRawSuperClusters->size());
  hist_EE_PreSC_Size_->Fill(endcapPreSuperClusters->size());
  hist_EE_CorSC_Size_->Fill(endcapCorSuperClusters->size());
  
  // Do RAW ENDCAP SuperClusters
  for(reco::SuperClusterCollection::const_iterator aClus = endcapRawSuperClusters->begin(); 
      aClus != endcapRawSuperClusters->end(); aClus++)
    {
      hist_EE_RawSC_NumBC_->Fill(aClus->clustersSize());
      hist_EE_RawSC_ET_->Fill(aClus->energy()/std::cosh(aClus->position().eta()));
      hist_EE_RawSC_Eta_->Fill(aClus->position().eta());
      hist_EE_RawSC_Phi_->Fill(aClus->position().phi());
      
      const reco::CaloClusterPtr seed = aClus->seed();
      hist_EE_RawSC_S1toS9_->Fill( lazyTool.eMax( *seed ) / lazyTool.e3x3( *seed ) );
      hist_EE_RawSC_S25toE_->Fill( lazyTool.e5x5( *seed ) / aClus->energy() );
      
      // truth
      double dRClosest = 999.9;
      double energyClosest = 0;
      closestMCParticle(genEvent, *aClus, dRClosest, energyClosest);
      
      if (dRClosest < 0.1)
	{
	  hist_EE_RawSC_EoverTruth_->Fill(aClus->energy()/energyClosest);
	  hist_EE_RawSC_deltaR_->Fill(dRClosest);
	}
      
    }
  
  // Do ENDCAP SuperClusters with PRESHOWER
  for(reco::SuperClusterCollection::const_iterator aClus = endcapPreSuperClusters->begin();
      aClus != endcapPreSuperClusters->end(); aClus++)
    {
      hist_EE_PreSC_NumBC_->Fill(aClus->clustersSize());
      hist_EE_PreSC_ET_->Fill(aClus->energy()/std::cosh(aClus->position().eta()));
      hist_EE_PreSC_Eta_->Fill(aClus->position().eta());
      hist_EE_PreSC_Phi_->Fill(aClus->position().phi());
      hist_EE_PreSC_preshowerE_->Fill(aClus->preshowerEnergy());
      
      const reco::CaloClusterPtr seed = aClus->seed();
      hist_EE_PreSC_S1toS9_->Fill( lazyTool.eMax( *seed ) / lazyTool.e3x3( *seed ) );
      hist_EE_PreSC_S25toE_->Fill( lazyTool.e5x5( *seed ) / aClus->energy() );
      
      // truth
      double dRClosest = 999.9;
      double energyClosest = 0;
      closestMCParticle(genEvent, *aClus, dRClosest, energyClosest);
      
      if (dRClosest < 0.1)
	{
	  hist_EE_PreSC_EoverTruth_->Fill(aClus->energy()/energyClosest);
	  hist_EE_PreSC_deltaR_->Fill(dRClosest);
	}
      
    }
  
  // Do CORRECTED ENDCAP SuperClusters
  for(reco::SuperClusterCollection::const_iterator aClus = endcapCorSuperClusters->begin();
      aClus != endcapCorSuperClusters->end(); aClus++)
    {
      hist_EE_CorSC_NumBC_->Fill(aClus->clustersSize());
      hist_EE_CorSC_ET_->Fill(aClus->energy()/std::cosh(aClus->position().eta()));
      hist_EE_CorSC_Eta_->Fill(aClus->position().eta());
      hist_EE_CorSC_Phi_->Fill(aClus->position().phi());
      hist_EE_CorSC_preshowerE_->Fill(aClus->preshowerEnergy());
      
      hist_EE_CorSC_ET_vs_Eta_->Fill( aClus->energy()/std::cosh(aClus->position().eta()), aClus->eta() );
      hist_EE_CorSC_ET_vs_Phi_->Fill( aClus->energy()/std::cosh(aClus->position().eta()), aClus->phi() );
      hist_EE_CorSC_ET_vs_R_->Fill( aClus->energy()/std::cosh(aClus->position().eta()), 
				    std::sqrt( std::pow(aClus->x(),2) + std::pow(aClus->y(),2) ) );
      
      
      // correction variables
      hist_EE_CorSC_phiWidth_->Fill(aClus->phiWidth());
      hist_EE_CorSC_etaWidth_->Fill(aClus->etaWidth());
      
      const reco::CaloClusterPtr seed = aClus->seed();
      hist_EE_CorSC_S1toS9_->Fill( lazyTool.eMax( *seed ) / lazyTool.e3x3( *seed ) );
      hist_EE_CorSC_S25toE_->Fill( lazyTool.e5x5( *seed ) / aClus->energy() );
      
      // truth
      double dRClosest = 999.9;
      double energyClosest = 0;
      closestMCParticle(genEvent, *aClus, dRClosest, energyClosest);
      
      if (dRClosest < 0.1)
	{
	  hist_EE_CorSC_EoverTruth_->Fill(aClus->energy()/energyClosest);
	  hist_EE_CorSC_deltaR_->Fill(dRClosest);
	}
      
    }
  
}

//
// Closest MC Particle
//
void
EgammaSuperClusters::closestMCParticle(const HepMC::GenEvent *genEvent, const reco::SuperCluster &sc, 
                                       double &dRClosest, double &energyClosest) const
{
  
  // SuperCluster eta, phi
  double scEta = sc.eta();
  double scPhi = sc.phi();
  
  // initialize dRClosest to a large number
  dRClosest = 999.9;
  
  // loop over the MC truth particles to find the
  // closest to the superCluster in dR space
  for(HepMC::GenEvent::particle_const_iterator currentParticle = genEvent->particles_begin();
      currentParticle != genEvent->particles_end(); currentParticle++ )
    {
      if((*currentParticle)->status() == 1)
	{
	  // need GenParticle in ECAL co-ordinates
	  HepMC::FourVector vtx = (*currentParticle)->production_vertex()->position();
	  double phiTrue = (*currentParticle)->momentum().phi();
	  double etaTrue = ecalEta((*currentParticle)->momentum().eta(), vtx.z()/10., vtx.perp()/10.);
	  
	  double dPhi = reco::deltaPhi(phiTrue, scPhi);
	  double dEta = scEta - etaTrue;
	  double deltaR = std::sqrt(dPhi*dPhi + dEta*dEta);
	  
	  if(deltaR < dRClosest)
	    {
	      dRClosest = deltaR;
	      energyClosest = (*currentParticle)->momentum().e();
	    }
	  
	} // end if stable particle	
      
    } // end loop on get particles
  
}


//
// Compute Eta in the ECAL co-ordinate system
//
float
EgammaSuperClusters::ecalEta(float EtaParticle , float Zvertex, float plane_Radius) const
{  
  const float R_ECAL           = 136.5;
  const float Z_Endcap         = 328.0;
  const float etaBarrelEndcap  = 1.479;
  
  if(EtaParticle != 0.)
    {
      float Theta = 0.0  ;
      float ZEcal = (R_ECAL-plane_Radius)*sinh(EtaParticle)+Zvertex;
      
      if(ZEcal != 0.0) Theta = atan(R_ECAL/ZEcal);
      if(Theta<0.0) Theta = Theta+Geom::pi() ;
      
      float ETA = - log(tan(0.5*Theta));
      
      if( fabs(ETA) > etaBarrelEndcap )
	{
	  float Zend = Z_Endcap ;
	  if(EtaParticle<0.0 )  Zend = -Zend ;
	  float Zlen = Zend - Zvertex ;
	  float RR = Zlen/sinh(EtaParticle);
	  Theta = atan((RR+plane_Radius)/Zend);
	  if(Theta<0.0) Theta = Theta+Geom::pi() ;
	  ETA = - log(tan(0.5*Theta));
	}
      
      return ETA;
    }
  else
    {
      edm::LogWarning("")  << "[EgammaSuperClusters::ecalEta] Warning: Eta equals to zero, not correcting" ;
      return EtaParticle;
    }
}

