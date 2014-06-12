// -*- C++ -*-
//
//
// dummy producer for a L1TkTauParticle
// The code simply match the L1CaloTaus with the closest L1Track.
// 

// system include files
#include <memory>
#include <string>
#include "TMath.h"
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkTauParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkTauParticleFwd.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkPrimaryVertex.h" // new
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h" // new
// for L1Tracks:
#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h" //for 'L1TkTrack_PixelDigi_Collection', etc..
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

using namespace l1extra ;

// ---------- class declaration  ---------- //
class L1TkTauFromCaloProducer : public edm::EDProducer {
public:
  
  typedef TTTrack< Ref_PixelDigi_ > L1TkTrackType;
  typedef edm::Ptr< L1TkTrackType > L1TkTrackRefPtr;
  typedef std::vector< L1TkTrackType > L1TkTrackCollectionType;
  typedef edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > L1TkStubRef; //new
  
  explicit L1TkTauFromCaloProducer(const edm::ParameterSet&);
  ~L1TkTauFromCaloProducer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions); 
  
  struct TrackPtComparator{
    bool operator() (const L1TkTrackRefPtr trackA, L1TkTrackRefPtr trackB ) const
    {
      return ( trackA->getMomentum().perp() > trackB->getMomentum().perp() );
    }
  };

  struct IsBelowTrackPtThreshold{
    float ptMinTrack;
    bool operator()(const L1TkTrackRefPtr track) { return track->getMomentum().perp() < ptMinTrack; }
  };
  
  struct IsAboveChi2RedMaxThreshold{
    float chi2RedMaxNoEndcapStub;
    float chi2RedMaxWithEndcapStub;
    float chi2RedMax;
    bool operator()(const L1TkTrackRefPtr track) { 
      
      /// Get the number of the L1TkTracks_Stubs
      std::vector< L1TkStubRef >  theStubs = track -> getStubRefs();
      unsigned int nStubs = (int) theStubs.size(); //new
      bool bHasEndcap     = false;
      
      /// For-loop: L1TkTracks Stubs
      for ( unsigned int jStub = 0; jStub < nStubs; jStub++ ){
	
	/// Get the detector id for the stub (barrel / endcap)
	StackedTrackerDetId thisDetId( theStubs.at(jStub)->getDetId() ); //new
	
	if ( thisDetId.isEndcap() ){ 
	  bHasEndcap = true; 
	  break;
	}
	
      } //For-loop: L1TkTracks Stubs
      
      /// Apply stub quality criteria for the L1TkTracks. 
      if ( bHasEndcap ){ chi2RedMax = chi2RedMaxWithEndcapStub; }
      else{ chi2RedMax = chi2RedMaxNoEndcapStub; }
      return track->getChi2Red() > chi2RedMax;
    }
  };
  
  struct IsBelowNStubsMinThreshold{
    unsigned int nStubsMin;
    bool operator()(const L1TkTrackRefPtr track) { return track->getStubRefs().size() < nStubsMin; } //new
  };
  
  struct IsAboveVtxZ0MaxThreshold{
    float vtxZ0Max;
    bool operator()(const L1TkTrackRefPtr track) { return fabs(track->getPOCA().z()) > vtxZ0Max; } //new 
  };

  struct TrackFoundInTkClusterMap{
    std::map< L1TkTrackRefPtr, unsigned int > m_TrackToCluster;
    bool operator()(const L1TkTrackRefPtr track) { return m_TrackToCluster.find( track ) != m_TrackToCluster.end(); }
  };


private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  // ---------- member data  ---------- //
  edm::InputTag L1TausInputTag;
  edm::InputTag L1TrackInputTag;	 
  unsigned int L1TkTrack_NStubsMin;       // Min number of stubs per Track (unitless)
  double L1TkTrack_PtMin_AllTracks;       // Min pT of tracks to consider [GeV]
  double L1TkTrack_PtMin_SignalTracks;    // Min pT applied on signal L1TkTracks [GeV]
  double L1TkTrack_PtMin_IsoTracks;       // Min pT applied on isolation L1TkTracks [GeV]
  double L1TkTrack_RedChiSquareEndcapMax; // Max chi squared for L1TkTracks in Endcap [unitless]
  double L1TkTrack_RedChiSquareBarrelMax; // Max chi squared for L1TkTracks in Barrel [unitless]
  bool L1TkTrack_ApplyVtxIso;             // Produce isolated L1TkTaus (True) or just L1TkTaus (False).
  double L1TkTrack_VtxIsoZ0Max;           // Max vertex z for L1TkTracks for VtxIsolation [cm] 
  double L1TkTrack_VtxZ0Max;              // Max vertex z for L1TkTracks [cm]
  double DeltaR_L1TkTau_L1TkTrack;        // Cone size for assigning L1TkTracks to L1TkTau candidate [unitless]
  double DeltaR_L1TkTauIsolation;         // Isolation cone size for L1TkTau [unitless]
  double DeltaR_L1TkTau_L1CaloTau;        // Cone size for matching L1TkTau to L1CaloTau [unitless]
  double L1CaloTau_EtMin;                 // Min eT applied on all L1CaloTaus [GeV]
  bool RemoveL1TkTauTracksFromIsoCalculation;  // Remove tracks used in L1TkTau construction from isolation calculation?
  
} ;


// ------------ constructor  ------------ //
L1TkTauFromCaloProducer::L1TkTauFromCaloProducer(const edm::ParameterSet& iConfig){

  L1TausInputTag                        = iConfig.getParameter<edm::InputTag>("L1TausInputTag");
  L1TrackInputTag                       = iConfig.getParameter<edm::InputTag>("L1TrackInputTag");  
  L1TkTrack_NStubsMin                   = (unsigned int)iConfig.getParameter<uint32_t>("L1TkTrack_NStubsMin");
  L1TkTrack_PtMin_AllTracks             = iConfig.getParameter<double>("L1TkTrack_PtMin_AllTracks");
  L1TkTrack_PtMin_SignalTracks          = iConfig.getParameter<double>("L1TkTrack_PtMin_SignalTracks");
  L1TkTrack_PtMin_IsoTracks             = iConfig.getParameter<double>("L1TkTrack_PtMin_IsoTracks");
  L1TkTrack_RedChiSquareEndcapMax       = iConfig.getParameter< double >("L1TkTrack_RedChiSquareEndcapMax");
  L1TkTrack_RedChiSquareBarrelMax       = iConfig.getParameter< double >("L1TkTrack_RedChiSquareBarrelMax");
  L1TkTrack_ApplyVtxIso                 = iConfig.getParameter< bool >("L1TkTrack_ApplyVtxIso");
  L1TkTrack_VtxIsoZ0Max                 = iConfig.getParameter< double >("L1TkTrack_VtxIsoZ0Max");
  L1TkTrack_VtxZ0Max                    = iConfig.getParameter< double >("L1TkTrack_VtxZ0Max");
  DeltaR_L1TkTau_L1TkTrack              = iConfig.getParameter< double >("DeltaR_L1TkTau_L1TkTrack");
  DeltaR_L1TkTauIsolation               = iConfig.getParameter< double >("DeltaR_L1TkTauIsolation");
  DeltaR_L1TkTau_L1CaloTau              = iConfig.getParameter< double >("DeltaR_L1TkTau_L1CaloTau");
  L1CaloTau_EtMin                       = iConfig.getParameter< double >("L1CaloTau_EtMin");
  RemoveL1TkTauTracksFromIsoCalculation = iConfig.getParameter< bool >("RemoveL1TkTauTracksFromIsoCalculation");
  produces<L1TkTauParticleCollection>();
}


// ------------ destructor  ------------ //
L1TkTauFromCaloProducer::~L1TkTauFromCaloProducer(){}

// ------------ method called to produce the data  ------------ //
void L1TkTauFromCaloProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){
  
  using namespace edm;
  using namespace std;
  std::auto_ptr<L1TkTauParticleCollection> result(new L1TkTauParticleCollection);
  
  /// Collection Handles: The L1TkTracks the PixelDigi collection
  typedef std::vector< L1TkTrackType >  L1TkTrack_PixelDigi_MyCollection; 
  edm::Handle< L1TkTrack_PixelDigi_MyCollection > h_PixelDigi_L1TkTrack; //new
  iEvent.getByLabel( L1TrackInputTag, h_PixelDigi_L1TkTrack );
  L1TkTrack_PixelDigi_MyCollection::const_iterator L1TkTrack;

  /// Collection Handles: The L1CaloTau from the L1 ExtraParticles
  edm::Handle< std::vector< l1extra::L1JetParticle > > h_L1CaloTau;
  iEvent.getByLabel( L1TausInputTag, h_L1CaloTau );

  /// Define maps and their iterators
  std::map< L1TkTrackRefPtr, unsigned int > m_L1TkTrackToL1TkCluster; // L1TkTrackRefPtr --> Cluster index it belongs to  
  std::map< L1TkTrackRefPtr, unsigned int >::const_iterator it_L1TkTrackToL1TkCluster;
  
  std::map< unsigned int, std::vector< L1TkTrackRefPtr > > m_L1TkClusterToL1TkTracks; // Reverse of m_L1TkTrackToL1TkCluster
  std::map< unsigned int, std::vector< L1TkTrackRefPtr > >::const_iterator it_L1TkClusterToL1TkTracks;


  /// L1TkTracks: Quality Criteria
  ////////////////////////////////
  std::vector< L1TkTrackRefPtr > v_L1TkTracks_wQuality;
  unsigned int iL1TkTrack = 0;

  /// For-loop: L1TkTracks
  for ( L1TkTrack = h_PixelDigi_L1TkTrack->begin(); L1TkTrack != h_PixelDigi_L1TkTrack->end(); L1TkTrack++ ){

    /// Make a pointer to the L1TkTracks
    L1TkTrackRefPtr L1TkTrack_Ptr( h_PixelDigi_L1TkTrack, iL1TkTrack++);
    
    /// Get the number of stubs of the L1TkTracks
    std::vector< L1TkStubRef >  L1TkTrackStubs = L1TkTrack -> getStubRefs() ;
    unsigned int L1TkTrack_NStubs = L1TkTrackStubs.size();
    double L1TkTrack_Pt           = L1TkTrack->getMomentum().perp();
    double L1TkTrack_VtxZ0        = L1TkTrack->getPOCA().z();
    bool L1TkTrack_HasEndcap      = false;
    bool L1TkTrack_HasBarrel      = false;

    /// Apply user-defined L1TkTracks quality criteria
    if ( L1TkTrack_NStubs < L1TkTrack_NStubsMin ){ continue; }
    if ( L1TkTrack_Pt < L1TkTrack_PtMin_AllTracks ){ continue; }
    if ( fabs( L1TkTrack_VtxZ0 ) > L1TkTrack_VtxZ0Max ){ continue; }

    /// Nested for-loop: L1TkTrack Stubs
    for ( unsigned int iStub = 0; iStub < L1TkTrack_NStubs; iStub++ ){

      /// Get the detector id for the stub (barrel / endcap)
      StackedTrackerDetId thisDetId( L1TkTrackStubs.at(iStub)->getDetId() );	
      if ( thisDetId.isEndcap() ){ L1TkTrack_HasEndcap = true; }
      if ( thisDetId.isBarrel() ){ L1TkTrack_HasBarrel = true; }	

    } // nested for-loop: L1TkTrack Stubs

    /// Apply user-defined L1TkTracks stub quality criteria
    if ( ( L1TkTrack_HasEndcap ) && ( L1TkTrack->getChi2Red() > L1TkTrack_RedChiSquareEndcapMax ) ){ continue; }
    if ( ( L1TkTrack_HasBarrel ) && ( L1TkTrack->getChi2Red() > L1TkTrack_RedChiSquareBarrelMax ) ){ continue; }

    v_L1TkTracks_wQuality.push_back( L1TkTrack_Ptr );
        
  } // For-loop: L1TkTracks


  /// Sort the quality L1TkTrack with pT 
  std::sort( v_L1TkTracks_wQuality.begin(), v_L1TkTracks_wQuality.end(), TrackPtComparator() );


  /// L1TkTracks: Isolation Cone
  //////////////////////////////
  std::vector< L1TkTrackRefPtr > v_L1TkTracks_IsoCone = v_L1TkTracks_wQuality;

  /// Apply min pt requirement
  IsBelowTrackPtThreshold L1TkTracks_PtFilterIso;
  L1TkTracks_PtFilterIso.ptMinTrack = L1TkTrack_PtMin_IsoTracks;
  v_L1TkTracks_IsoCone.erase( remove_if(v_L1TkTracks_IsoCone.begin(), v_L1TkTracks_IsoCone.end(), L1TkTracks_PtFilterIso ), v_L1TkTracks_IsoCone.end() );

  // /// Apply max reduced chi squared requirement
  // IsAboveChi2RedMaxThreshold L1TkTracks_Chi2RedFilter;
  // L1TkTracks_Chi2RedFilter.chi2RedMaxNoEndcapStub   = L1TkTrack_RedChiSquareBarrelMax;
  // L1TkTracks_Chi2RedFilter.chi2RedMaxWithEndcapStub = L1TkTrack_RedChiSquareEndcapMax;
  // v_L1TkTracks_IsoCone.erase( remove_if(v_L1TkTracks_IsoCone.begin(), v_L1TkTracks_IsoCone.end(), L1TkTracks_Chi2RedFilter ), v_L1TkTracks_IsoCone.end());

  // /// Apply min number of stubs requirement
  // IsBelowNStubsMinThreshold L1TkTracks_NStubsMinFilter;
  // L1TkTracks_NStubsMinFilter.nStubsMin = L1TkTrack_NStubsMin;
  // v_L1TkTracks_IsoCone.erase( remove_if(v_L1TkTracks_IsoCone.begin(), v_L1TkTracks_IsoCone.end(), L1TkTracks_NStubsMinFilter ), v_L1TkTracks_IsoCone.end());

  // /// Apply max vertex-z requirement
  // IsAboveVtxZ0MaxThreshold L1TkTracks_VtxZ0MaxFilter;
  // L1TkTracks_VtxZ0MaxFilter.vtxZ0Max = L1TkTrack_VtxZ0Max;
  // v_L1TkTracks_IsoCone.erase( remove_if(v_L1TkTracks_IsoCone.begin(), v_L1TkTracks_IsoCone.end(), L1TkTracks_VtxZ0MaxFilter ), v_L1TkTracks_IsoCone.end());

  /// L1TkTracks: Signal Cone
  ///////////////////////////
  std::vector< L1TkTrackRefPtr > v_L1TkTracks_Signal = v_L1TkTracks_wQuality;
  
  /// Apply min pt requirement
  IsBelowTrackPtThreshold L1TkTracks_PtFilterSignal;
  L1TkTracks_PtFilterSignal.ptMinTrack = L1TkTrack_PtMin_SignalTracks;
  v_L1TkTracks_Signal.erase( remove_if(v_L1TkTracks_Signal.begin(), v_L1TkTracks_Signal.end(), L1TkTracks_PtFilterSignal ), v_L1TkTracks_Signal.end() );  


  /// L1TkTau: Cluster Construction
  /////////////////////////////////
  unsigned int nL1TkTauClusters  = 0;
  /// For-loop: L1TkTracks
  for ( unsigned int iTrack = 0; iTrack < v_L1TkTracks_Signal.size(); iTrack++ ){
    
    L1TkTrackRefPtr ldgTk = v_L1TkTracks_Signal.at(iTrack); 

    /// Determine if track has already been assigned to a cluster
    bool bLdgTkFoundInTkClusterMap = m_L1TkTrackToL1TkCluster.find( ldgTk ) != m_L1TkTrackToL1TkCluster.end();
    if ( bLdgTkFoundInTkClusterMap == true ){ continue; }

    /// Add L1TkTrack to the L1TkTau Cluster
    nL1TkTauClusters++;
    m_L1TkTrackToL1TkCluster.insert( std::make_pair( ldgTk, nL1TkTauClusters ) );

    /// Get the L1TkTrack properties
    double ldgTk_Eta = ldgTk->getMomentum().eta();
    double ldgTk_Phi = ldgTk->getMomentum().phi();

    /// Nested for-loop: L1TkTracks
    for ( unsigned int jTrack = 0; jTrack < v_L1TkTracks_Signal.size(); jTrack++ ){
      
      L1TkTrackRefPtr tkToAdd = v_L1TkTracks_Signal.at(jTrack);

      /// Skip if L1TkTrack is identical to ldgTk or if previously used in another L1TkTau Cluster
      if ( ldgTk.get() == tkToAdd.get() ){ continue; }
      bool bTKFoundInTkClusterMap = m_L1TkTrackToL1TkCluster.find( tkToAdd ) != m_L1TkTrackToL1TkCluster.end();
      if ( bTKFoundInTkClusterMap == true ){ continue; }
      
      /// Check if the L1TkTrack is inside L1TkTau signal cone (around ldgTk). If yes add it to the L1TkTau Cluster
      double tkToAdd_Eta = tkToAdd->getMomentum().eta();
      double tkToAdd_Phi = tkToAdd->getMomentum().phi();
      double deltaR      = reco::deltaR(ldgTk_Eta, ldgTk_Phi, tkToAdd_Eta, tkToAdd_Phi);
      if ( deltaR > DeltaR_L1TkTau_L1TkTrack ){	continue; }
      m_L1TkTrackToL1TkCluster.insert( std::make_pair( tkToAdd, nL1TkTauClusters ) );
    
    } /// Nested for-loop: L1TkTracks

  } /// For-loop: L1TkTracks



  /// L1TkTau: Cluster Construction (reverse map)
  ///////////////////////////////////////////////
  
  /// For-loop: m_L1TkTrackToL1TkTracks elements 
  for ( it_L1TkTrackToL1TkCluster = m_L1TkTrackToL1TkCluster.begin(); it_L1TkTrackToL1TkCluster != m_L1TkTrackToL1TkCluster.end(); it_L1TkTrackToL1TkCluster++ ){
    
    /// Check if the cluster is already put in the reverse-map. If not, create the vector and put into the map, if yes, add to the vector.
    bool bTkClusterFoundInTkMap = m_L1TkClusterToL1TkTracks.find( it_L1TkTrackToL1TkCluster->second ) != m_L1TkClusterToL1TkTracks.end();

    if ( !bTkClusterFoundInTkMap ){
      std::vector< L1TkTrackRefPtr > temp;
      temp.push_back( it_L1TkTrackToL1TkCluster->first );
      m_L1TkClusterToL1TkTracks.insert( std::make_pair( it_L1TkTrackToL1TkCluster->second, temp ) );
    }
    else{ m_L1TkClusterToL1TkTracks.find( it_L1TkTrackToL1TkCluster->second )->second.push_back( it_L1TkTrackToL1TkCluster->first ); }
  
  } // For-loop: m_L1TkTrackToL1TkTracks elements



  /// L1TkTau: Isolation
  //////////////////////
  
  /// Exclude L1TkTracks that have been used in the construction of the L1TkTaus from isolation calculation?
  TrackFoundInTkClusterMap L1TkTracks_TkClusterFilter;
  L1TkTracks_TkClusterFilter.m_TrackToCluster = m_L1TkTrackToL1TkCluster;
  if( RemoveL1TkTauTracksFromIsoCalculation == true ) {
    v_L1TkTracks_IsoCone.erase( remove_if(v_L1TkTracks_IsoCone.begin(), v_L1TkTracks_IsoCone.end(), L1TkTracks_TkClusterFilter ), v_L1TkTracks_IsoCone.end());
  }

  unsigned int nTksIsoCone_VtxIso = 0;
  bool bL1TkTauPassedVtxIso       = true;
  /// For-loop: m_L1TkClusterToL1TkTracks elements
  for ( it_L1TkClusterToL1TkTracks = m_L1TkClusterToL1TkTracks.begin(); it_L1TkClusterToL1TkTracks != m_L1TkClusterToL1TkTracks.end(); it_L1TkClusterToL1TkTracks++ ){

    nTksIsoCone_VtxIso   = 0;
    bL1TkTauPassedVtxIso = true;

    /// Get the L1TkTracks that compose the L1TkTau and sort by pT
    std::vector< L1TkTrackRefPtr > theseTracks = it_L1TkClusterToL1TkTracks->second;
    
    /// Get direction of the L1TkTau LdgTk 
    double L1TkTau_LdgTk_Eta      = theseTracks.at(0)->getMomentum().eta();
    double L1TkTau_LdgTk_Phi      = theseTracks.at(0)->getMomentum().phi(); 
    double L1TkTau_LdgTk_VtxZ0    = theseTracks.at(0)->getPOCA().z(); //new
    
    /// Nested for-loop: L1TkTracks (IsoCone)
    for ( unsigned int iTrack = 0; iTrack < v_L1TkTracks_IsoCone.size(); iTrack++ ){

      L1TkTrackRefPtr isoTk = v_L1TkTracks_IsoCone.at(iTrack); 

      /// If this L1TkTrack belongs to the L1TkTau, or if outside the isolation cone skip it
      bool bSkipThisTrack = false;

      /// Nested for-loop: L1TkTracks (Cluster)
      for ( unsigned int jTrack = 0; jTrack < theseTracks.size(); jTrack++ ){

	/// Skip identical tracks
	L1TkTrackRefPtr L1TkTau_ClusterTk = theseTracks.at(jTrack); 
	if ( L1TkTau_ClusterTk.get() == isoTk.get() ){ bSkipThisTrack = true; }

      } /// Nested for-loop: L1TkTracks in Cluster
      if ( bSkipThisTrack == true ){ continue; }
    
      /// Is the L1TkTrack within the isolation cone of the L1TkTau?
      float isoTk_Eta       = isoTk-> getMomentum().eta();
      float isoTk_Phi       = isoTk-> getMomentum().phi();
      double deltaR         = reco::deltaR(L1TkTau_LdgTk_Eta, L1TkTau_LdgTk_Phi, isoTk_Eta, isoTk_Phi);
      if ( deltaR > DeltaR_L1TkTauIsolation ){ continue; }    
    
      /// Calculate number of L1TkTracks inside isolation cone of L1TkTau, that are within a distance "L1TkTrack_VtxZ0Max" in vertex-z position
      double isoTk_VtxZ0    = isoTk->getPOCA().z(); //new
      double deltaVtxIso_Z0 = fabs ( L1TkTau_LdgTk_VtxZ0 - isoTk_VtxZ0 );
      if ( deltaVtxIso_Z0 <= L1TkTrack_VtxIsoZ0Max ){ nTksIsoCone_VtxIso++; }

    } /// Nested for-loop: L1TkTracks (Signal)
  
    if ( nTksIsoCone_VtxIso > 0 && L1TkTrack_ApplyVtxIso == true){ bL1TkTauPassedVtxIso = false; }

  
    /// L1CaloTau: Matching with L1TkTau
    ////////////////////////////////////
    unsigned int iL1CaloTauIndex   = -1;
    unsigned int iMatchedL1CaloTau = 0;
    double minDeltaR               = 99999;
    bool bFoundL1TkTauCaloMatch    = false;
    std::vector< L1TkTrackRefPtr > L1TkTau_TkPtrs;
    
    /// Nested for-loop: L1CaloTaus
    for ( std::vector< l1extra::L1JetParticle >::const_iterator L1CaloTau = h_L1CaloTau->begin();  L1CaloTau != h_L1CaloTau->end(); L1CaloTau++){
      
      iL1CaloTauIndex++; //starts at 0

      /// Match L1CaloTau with L1TkTau
      double L1CaloTau_Et  = L1CaloTau->et();
      double L1CaloTau_Eta = L1CaloTau->eta();
      double L1CaloTau_Phi = L1CaloTau->phi();
      if ( L1CaloTau_Et < L1CaloTau_EtMin){ continue; }

      /// Calculate distance of L1CaloTau from the L1TkTau Leading Track
      double deltaR = reco::deltaR(L1TkTau_LdgTk_Eta, L1TkTau_LdgTk_Phi, L1CaloTau_Eta, L1CaloTau_Phi);
      if ( deltaR < minDeltaR ){ 
	minDeltaR = deltaR; 
	iMatchedL1CaloTau = iL1CaloTauIndex;
      }
      
    } /// Nested for-Loop: L1CaloTaus 

     /// Ensuse that the L1CaloTau closest to the L1TkTau is at least within a certain matching cone size. Create a L1CaloTau Ref Pointer
    if ( minDeltaR < DeltaR_L1TkTau_L1CaloTau ){
      bFoundL1TkTauCaloMatch = true;
      L1TkTau_TkPtrs         = it_L1TkClusterToL1TkTracks->second;
    }

    /// Construct 4-Momentum of L1TkTau Candidate
    math::XYZTLorentzVector L1TkTau_P4(0, 0, 0, 0);
    float mPionPlus = 0.140; // GeV

    /// Nested for-loop: L1TkTracks of L1TkTau  //fixme: use calo Et 
    for ( unsigned int iTk = 0; iTk < L1TkTau_TkPtrs.size(); iTk++ ){
      
      L1TkTrackRefPtr tk = L1TkTau_TkPtrs.at(iTk);
      
      float L1TkTauTk_Px  = tk-> getMomentum().x();
      float L1TkTauTk_Py  = tk-> getMomentum().y();
      float L1TkTauTk_Pz  = tk-> getMomentum().z();
      float L1TkTauTk_E   = sqrt( L1TkTauTk_Px*L1TkTauTk_Px + L1TkTauTk_Py*L1TkTauTk_Py + L1TkTauTk_Pz*L1TkTauTk_Pz + mPionPlus * mPionPlus );
      math::XYZTLorentzVector tmp_P4(L1TkTauTk_Px, L1TkTauTk_Py, L1TkTauTk_Pz, L1TkTauTk_E);
      L1TkTau_P4 += tmp_P4;

    } /// Nested for-loop: L1TkTracks of L1TkTau 

    /// Apply Vtx-Based isolation on L1TkTaus. Also ensure a L1CaloTau has been matched to a L1TkTau
    bool bFillEvent =  bL1TkTauPassedVtxIso * bFoundL1TkTauCaloMatch;
    if(  bFillEvent == false ) { continue; }
    
    /// Save the L1TkTau candidate, its tracks, isolation variable (dumbie) and its L1CaloTau reference.
    float L1TkTau_TkIsol = -999.9;
    edm::Ref< L1JetParticleCollection > L1TauCaloRef( h_L1CaloTau, iMatchedL1CaloTau ); 

    /// Since the L1TkTauParticle does not support a vector of Track Ptrs yet, use instead a "hack" to simplify the code.
    edm::Ptr< L1TkTrackType > L1TrackPtrNull;    
    if (L1TkTau_TkPtrs.size() == 1) { 
      L1TkTau_TkPtrs.push_back(L1TrackPtrNull); 
      L1TkTau_TkPtrs.push_back(L1TrackPtrNull); 
    }
    else if (L1TkTau_TkPtrs.size() == 2) { 
      L1TkTau_TkPtrs.push_back(L1TrackPtrNull); 
    }
    else{}

    /// Fill The L1TkTauParticle object:
    // L1TkTauParticle L1TkTauFromCalo( L1TkTau_P4, L1TauCaloRef, L1TkTau_TkPtrs, L1TkTau_TkIsol );
    L1TkTauParticle L1TkTauFromCalo( L1TkTau_P4,
			    L1TauCaloRef,
			    L1TkTau_TkPtrs[0],
			    L1TkTau_TkPtrs[1],
			    L1TkTau_TkPtrs[2],
			    L1TkTau_TkIsol );

    result -> push_back( L1TkTauFromCalo );
    
  } /// For-loop: m_L1TkClusterToL1TkTracks elements

  iEvent.put( result );

}


// ------------ method called once each job just before starting event loop  ------------ //
void L1TkTauFromCaloProducer::beginJob(){}

// ------------ method called once each job just after ending the event loop  ------------ //
void L1TkTauFromCaloProducer::endJob() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------ //
void L1TkTauFromCaloProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TkTauFromCaloProducer);



