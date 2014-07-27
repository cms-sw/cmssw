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
#include "DataFormats/L1TrackTrigger/interface/L1TkPrimaryVertex.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
// for L1Tracks:
#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/L1TkTauEtComparator.h"


using namespace l1extra ;

// ---------- class declaration  ---------- //
class L1TkTauFromCaloProducer : public edm::EDProducer {
public:
  
  typedef TTTrack< Ref_PixelDigi_ > L1TkTrackType;
  typedef edm::Ptr< L1TkTrackType > L1TkTrackRefPtr;
  typedef std::vector< L1TkTrackType >   L1TkTrack_Collection;
  typedef std::vector< L1TkTrackRefPtr > L1TkTrackRefPtr_Collection; 
  typedef edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > L1TkStubRef;

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
    float ptMin;
    bool operator()(const L1TkTrackRefPtr track) { return track->getMomentum().perp() < ptMin; }
  };
  
  struct IsAboveTrackPtThreshold{
    float ptMax;
    bool operator()(const L1TkTrackRefPtr track) { return track->getMomentum().perp() > ptMax; }
  };

  struct TrackFoundInTkClusterMap{
    std::map< L1TkTrackRefPtr, unsigned int > m_TrackToCluster;
    bool operator()(const L1TkTrackRefPtr track) { return m_TrackToCluster.find( track ) != m_TrackToCluster.end(); }
  };
  
  struct CaloTauEtComparator{ 
    bool operator() (const l1extra::L1JetParticle tauA, const l1extra::L1JetParticle tauB) const
    {
      return ( tauA.et() > tauB.et() );
    }
  };

private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  // ---------- member data  ---------- //
  /// Parameters whose values are given through the python cfg file
  edm::InputTag cfg_L1TausInputTag;                 // Tag of Collection to be used for L1 Calorimeter Taus
  edm::InputTag cfg_L1TrackInputTag;                // Tag of Collection to be used for L1 Track Trigger Tracks
  unsigned int cfg_L1TkTrack_NStubsMin;             // Min number of stubs per L1TkTrack [unitless]
  double cfg_L1TkTrack_PtMin_AllTracks;             // Min pT applied on all L1TkTracks [GeV]
  double cfg_L1TkTrack_PtMin_SignalTracks;          // Min pT applied on signal L1TkTracks [GeV]
  double cfg_L1TkTrack_PtMin_IsoTracks;             // Min pT applied on isolation L1TkTracks [GeV]
  double cfg_L1TkTrack_RedChiSquareEndcapMax;       // Max chi squared for L1TkTracks in Endcap [unitless]
  double cfg_L1TkTrack_RedChiSquareBarrelMax;       // Max chi squared for L1TkTracks in Barrel [unitless]
  bool   cfg_L1TkTrack_ApplyVtxIso;                 // Produce isolated L1TkTaus (True) or just L1TkTaus (False).
  double cfg_L1TkTrack_VtxIsoZ0Max;                 // Max vertex z for L1TkTracks for VtxIsolation [cm] 
  double cfg_L1TkTrack_VtxZ0Max;                    // Max vertex z for L1TkTracks [cm]
  double cfg_DeltaR_L1TkTau_L1TkTrack;              // Cone size for assigning L1TkTracks to L1TkTau candidate [unitless]
  double cfg_DeltaR_L1TkTau_L1CaloTau;              // Matching cone for L1TkTau and L1CaloTau [unitless]
  double cfg_DeltaR_L1TkTau_Isolation_Min;          // Isolation cone size for L1TkTau [unitless]. If ">0" the isolation cone becomes an isolation annulus (default=0.1)
  double cfg_DeltaR_L1TkTau_Isolation_Max;          // Isolation cone size for L1TkTau [unitless] (default=0.4)
  bool   cfg_RemoveL1TkTauTracksFromIsoCalculation; // Remove tracks used in L1TkTau construction from isolation calculation?

  double L1CaloTau_EtMin;                 // Min eT applied on all L1CaloTaus [GeV]

  
} ;


// ------------ constructor  ------------ //
L1TkTauFromCaloProducer::L1TkTauFromCaloProducer(const edm::ParameterSet& iConfig){

  cfg_L1TausInputTag                        = iConfig.getParameter<edm::InputTag>("L1TausInputTag");
  cfg_L1TrackInputTag                       = iConfig.getParameter<edm::InputTag>("L1TrackInputTag");
  cfg_L1TkTrack_NStubsMin                   = (unsigned int)iConfig.getParameter< uint32_t >("L1TkTrack_NStubsMin");
  cfg_L1TkTrack_PtMin_AllTracks             = iConfig.getParameter< double >("L1TkTrack_PtMin_AllTracks");
  cfg_L1TkTrack_PtMin_SignalTracks          = iConfig.getParameter< double >("L1TkTrack_PtMin_SignalTracks");
  cfg_L1TkTrack_PtMin_IsoTracks             = iConfig.getParameter< double >("L1TkTrack_PtMin_IsoTracks");
  cfg_L1TkTrack_RedChiSquareEndcapMax       = iConfig.getParameter< double >("L1TkTrack_RedChiSquareEndcapMax"); 
  cfg_L1TkTrack_RedChiSquareBarrelMax       = iConfig.getParameter< double >("L1TkTrack_RedChiSquareBarrelMax");
  cfg_L1TkTrack_ApplyVtxIso                 = iConfig.getParameter< bool   >("L1TkTrack_ApplyVtxIso");
  cfg_L1TkTrack_VtxIsoZ0Max                 = iConfig.getParameter< double >("L1TkTrack_VtxIsoZ0Max");
  cfg_L1TkTrack_VtxZ0Max                    = iConfig.getParameter< double >("L1TkTrack_VtxZ0Max");
  cfg_DeltaR_L1TkTau_L1TkTrack              = iConfig.getParameter< double >("DeltaR_L1TkTau_L1TkTrack");
  cfg_DeltaR_L1TkTau_L1CaloTau              = iConfig.getParameter< double >("DeltaR_L1TkTau_L1CaloTau");
  cfg_DeltaR_L1TkTau_Isolation_Min          = iConfig.getParameter< double >("DeltaR_L1TkTau_Isolation_Min");
  cfg_DeltaR_L1TkTau_Isolation_Max          = iConfig.getParameter< double >("DeltaR_L1TkTau_Isolation_Max");
  cfg_RemoveL1TkTauTracksFromIsoCalculation = iConfig.getParameter< bool   >("RemoveL1TkTauTracksFromIsoCalculation");

  L1CaloTau_EtMin                       = iConfig.getParameter< double >("L1CaloTau_EtMin");

  produces<L1TkTauParticleCollection>();
  // produces<L1TkTauParticleCollection>("Tau").setBranchAlias( "Tau");
}


// ------------ destructor  ------------ //
L1TkTauFromCaloProducer::~L1TkTauFromCaloProducer(){}

// ------------ method called to produce the data  ------------ //
void L1TkTauFromCaloProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){
  
  using namespace edm;
  using namespace std;
  std::auto_ptr<L1TkTauParticleCollection> result(new L1TkTauParticleCollection);
  

  // ------------ Declaratations  ------------ //
  // double nan = std::numeric_limits<double>::quiet_NaN();

  /// First element is a L1TkTrack, second element is the index of the cluster it belongs to
  std::map< L1TkTrackRefPtr, unsigned int > m_L1TkTrackToL1TkCluster;

  /// First element is an index of the cluster a L1TkTrack belongs to, second element is a vector of L1TkTrack_Ptrs (reverse of m_L1TkTrackToL1TkCluster)
  std::map< unsigned int, std::vector< L1TkTrackRefPtr > > m_L1TkClusterToL1TkTracks;

  /// First element is the L1 Calo Tau, second element is the index of the closest (if matched, if any) L1 Tracker Track Cluster
  std::map< edm::Ptr< l1extra::L1JetParticle >, unsigned int > m_L1CaloTauToMatchedL1TkCluster;

  /// L1CaloTau
  edm::Handle< std::vector< l1extra::L1JetParticle > > h_L1CaloTau;
  iEvent.getByLabel( cfg_L1TausInputTag, h_L1CaloTau );


  // ------------ L1TkTracks  ------------ //
  edm::Handle< L1TkTrack_Collection > h_PixelDigi_L1TkTrack;
  iEvent.getByLabel( cfg_L1TrackInputTag, h_PixelDigi_L1TkTrack );

  L1TkTrackRefPtr_Collection c_L1TkTracks_Quality;
  unsigned int track_counter = 0;

  /// For-loop: L1TkTracks
  for ( L1TkTrack_Collection::const_iterator track = h_PixelDigi_L1TkTrack->begin(); track != h_PixelDigi_L1TkTrack->end(); track++ ){
  
    /// Make a pointer to the L1TkTracks
    L1TkTrackRefPtr track_RefPtr( h_PixelDigi_L1TkTrack, track_counter++);
    
    /// Declare for-loop variables
    std::vector< L1TkStubRef > track_Stubs = track -> getStubRefs();
    unsigned int track_NStubs              = track_Stubs.size();
    double track_Pt                        = track->getMomentum().perp();
    double track_VtxZ0                     = track->getPOCA().z();
    double track_RedChiSq                  = track->getChi2Red();
    bool track_HasEndcapStub               = false;
    bool track_HasBarrelStub               = false;

    /// Apply L1TkTracks quality criteria
    if ( track_Pt < cfg_L1TkTrack_PtMin_AllTracks ){ continue; }
    if ( track_NStubs < cfg_L1TkTrack_NStubsMin ){ continue; }
    if ( fabs( track_VtxZ0 ) > cfg_L1TkTrack_VtxZ0Max ){ continue; }

    /// For-loop (nested): L1TkTrack Stubs
    for ( unsigned int iStub = 0; iStub < track_NStubs; iStub++ ){
      
      /// Get the detector id for the stub (barrel / endcap)
      StackedTrackerDetId thisDetId( track_Stubs.at(iStub)->getDetId() );
      if ( thisDetId.isEndcap() ){ track_HasEndcapStub = true; }
      if ( thisDetId.isBarrel() ){ track_HasBarrelStub = true; }	

    } /// For-loop (nested): L1TkTrack Stubs

    /// Apply L1TkTracks Stubs quality criteria
    if ( ( track_HasEndcapStub ) && ( track_RedChiSq > cfg_L1TkTrack_RedChiSquareEndcapMax ) ){ continue; }
    if ( ( track_HasBarrelStub ) && ( track_RedChiSq > cfg_L1TkTrack_RedChiSquareBarrelMax ) ){ continue; }
    
    /// Beyond this point we should only have the user-defined quality L1TkTracks
    c_L1TkTracks_Quality.push_back( track_RefPtr );
    
  } // For-loop: L1TkTracks

  /// Sort by pT all selected L1TkTracks
  std::sort( c_L1TkTracks_Quality.begin(), c_L1TkTracks_Quality.end(), TrackPtComparator() );


  // ------------ L1TkTracks (Isolation Cone)  ------------ //
  IsAboveTrackPtThreshold filter_IsAboveTrackPtThreshold;
  L1TkTrackRefPtr_Collection c_L1TkTracks_IsoCone = c_L1TkTracks_Quality;  
  filter_IsAboveTrackPtThreshold.ptMax            = cfg_L1TkTrack_PtMin_SignalTracks;
  c_L1TkTracks_IsoCone.erase( remove_if( c_L1TkTracks_IsoCone.begin(), c_L1TkTracks_IsoCone.end(), filter_IsAboveTrackPtThreshold ), c_L1TkTracks_IsoCone.end() );
  
  /// Sort tracks by pT
  std::sort( c_L1TkTracks_IsoCone.begin(), c_L1TkTracks_IsoCone.end(), TrackPtComparator() );


  // ------------ L1TkTracks (Signal Cone)  ------------ //
  IsBelowTrackPtThreshold filter_IsBelowTrackPtThreshold;
  L1TkTrackRefPtr_Collection c_L1TkTracks_SignalCone = c_L1TkTracks_Quality;  
  filter_IsBelowTrackPtThreshold.ptMin                         = cfg_L1TkTrack_PtMin_SignalTracks;
  c_L1TkTracks_SignalCone.erase( remove_if( c_L1TkTracks_SignalCone.begin(), c_L1TkTracks_SignalCone.end(), filter_IsBelowTrackPtThreshold ), c_L1TkTracks_SignalCone.end() );

  /// Sort tracks by pT
  std::sort( c_L1TkTracks_SignalCone.begin(), c_L1TkTracks_SignalCone.end(), TrackPtComparator() );


  // ------------ L1TkTau Cluster: Construction ------------ //
  unsigned int nL1TkTaus = 0;

  /// For-loop: L1TkTracks (Signal Cone)
  for ( unsigned int i=0; i < c_L1TkTracks_SignalCone.size(); i++ ){
    
    L1TkTrackRefPtr iTrack = c_L1TkTracks_SignalCone.at(i);

    /// Determine if track has already been assigned to a cluster
    bool bTrackFoundInTkClusterMap = m_L1TkTrackToL1TkCluster.find( iTrack ) != m_L1TkTrackToL1TkCluster.end();
    if ( bTrackFoundInTkClusterMap  == true ){ continue; }

    /// If track has not already been assigned to a cluster then this is a new cluster
    nL1TkTaus++;

    /// Add this (leading) L1TkTrack to the L1TkTau Cluster. Increment L1TkTau Cluster counter
    m_L1TkTrackToL1TkCluster.insert( std::make_pair( iTrack, nL1TkTaus ) );    

    /// Get the L1TkTrack properties
    double ldgTrack_Eta = iTrack->getMomentum().eta();
    double ldgTrack_Phi = iTrack->getMomentum().phi();

    /// For-loop (nested): L1TkTracks (Signal Cone)
    for ( unsigned int j=0; j < c_L1TkTracks_SignalCone.size(); j++ ){
      
      /// Skip identical L1TkTracks
      L1TkTrackRefPtr jTrack = c_L1TkTracks_SignalCone.at(j);
      if ( iTrack.get() == jTrack.get() ){ continue; }
      
      /// If this track was used already in another L1TkTau Cluster, skip it. 
      bool bTrackFoundInTkClusterMap = m_L1TkTrackToL1TkCluster.find( jTrack ) != m_L1TkTrackToL1TkCluster.end();
      if ( bTrackFoundInTkClusterMap == true ){ continue; }

      /// Get the L1TkTrack direction and distance to the ldg L1TkTrack of the L1TkTau
      double track_Eta = jTrack->getMomentum().eta();
      double track_Phi = jTrack->getMomentum().phi();
      double deltaR   = reco::deltaR(ldgTrack_Eta, ldgTrack_Phi, track_Eta, track_Phi);

      /// If L1TkTrack is outside the signal cone, skip it. Otherwise, add it to the L1TkTau Cluster.
      if ( deltaR > cfg_DeltaR_L1TkTau_L1TkTrack ){ continue; }
      m_L1TkTrackToL1TkCluster.insert( std::make_pair( jTrack, nL1TkTaus ) );
    
    } /// For-loop (nested): L1TkTracks (Signal Cone)
  }  /// For-loop: L1TkTracks (Signal Cone)

  /// Remove L1TkTracks from the Isolation-Cone L1TkTrack-Collection if they have been used in L1TkTau Clusters
  TrackFoundInTkClusterMap filter_TkFoundInTkCluster;
  filter_TkFoundInTkCluster.m_TrackToCluster = m_L1TkTrackToL1TkCluster;
  if( cfg_RemoveL1TkTauTracksFromIsoCalculation == true ) {
    c_L1TkTracks_IsoCone.erase( remove_if( c_L1TkTracks_IsoCone.begin(), c_L1TkTracks_IsoCone.end(), filter_TkFoundInTkCluster ), c_L1TkTracks_IsoCone.end());
  }

  /// For-loop: m_L1TkTrackToL1TkTracks
  for ( std::map< L1TkTrackRefPtr, unsigned int >::const_iterator it_map = m_L1TkTrackToL1TkCluster.begin(); it_map != m_L1TkTrackToL1TkCluster.end(); ++it_map ){
    
    /// Check if the cluster is already put in the reverse-map. If not, create the vector and put into the map, if yes, add to the vector.
    bool bTkClusterFoundInTkMap = m_L1TkClusterToL1TkTracks.find( it_map->second ) != m_L1TkClusterToL1TkTracks.end();

    if ( !bTkClusterFoundInTkMap ){
      std::vector< L1TkTrackRefPtr > temp;
      temp.push_back( it_map->first );
      m_L1TkClusterToL1TkTracks.insert( std::make_pair( it_map->second, temp ) );
    }
    else{ m_L1TkClusterToL1TkTracks.find( it_map->second )->second.push_back( it_map->first ); }
  
  } /// For-loop: m_L1TkTrackToL1TkTracks


  // ------------ L1TkTau Cluster: Isolation ------------ //
  unsigned int nCloseTracks_Xcm = 0;
  bool bPassedVtxIso            = true;

  /// For-loop: m_L1TkClusterToL1TkTracks elements
  for ( std::map< unsigned int, std::vector< L1TkTrackRefPtr > >::const_iterator it = m_L1TkClusterToL1TkTracks.begin(); it != m_L1TkClusterToL1TkTracks.end(); it++){

    nCloseTracks_Xcm = 0;
    bPassedVtxIso    = true;

    /// Get the L1TkTracks that compose the L1TkTracks_Cluster
    std::vector< L1TkTrackRefPtr > theseTracks = it->second;
    
    /// Sort track in pT
    std::sort( theseTracks.begin(), theseTracks.end(), TrackPtComparator() );

    /// Get the L1TkTrack properties
    L1TkTrackRefPtr ldgTrack = theseTracks.at(0);
    double ldgTrack_Eta      = (ldgTrack)->getMomentum().eta();
    double ldgTrack_Phi      = (ldgTrack)->getMomentum().phi();
    double ldgTrack_VtxZ0    = (ldgTrack)->getPOCA().z();
        
    /// For-loop (nested): L1TkTracks (Isolation Cone)
    for ( unsigned int index = 0; index < c_L1TkTracks_IsoCone.size(); index++ ){ 

      /// Skip tracks that are identical to the one under investigation
      L1TkTrackRefPtr thisTrack = c_L1TkTracks_IsoCone.at(index);
      if ( thisTrack.get() == theseTracks.at(0).get() ){ continue; }

      /// Use only tracks close in eta/phi
      double thisEta   = thisTrack->getMomentum().eta();
      double thisPhi   = thisTrack->getMomentum().phi();
      double deltaR    = reco::deltaR(ldgTrack_Eta, ldgTrack_Phi, thisEta, thisPhi);

      /// Ensure that this "isolation" track is within the signal cone (annulus) defined by the user
      if ( (deltaR > cfg_DeltaR_L1TkTau_Isolation_Max) || (deltaR < cfg_DeltaR_L1TkTau_Isolation_Min) ){ continue; }

      /// Calculate the vertex-z distance of the "isolation" track from the ldg track
      double thisVtxZ0      = c_L1TkTracks_IsoCone.at(index)->getPOCA().z();
      double deltaVtxIso_Z0 = fabs ( ldgTrack_VtxZ0 - thisVtxZ0 );

      /// Calculate Vertex Isolation for L1TkTau
      if ( deltaVtxIso_Z0 <= cfg_L1TkTrack_VtxIsoZ0Max ){ nCloseTracks_Xcm++;  }
      
    } /// For-loop (nested): L1TkTracks (Isolation Cone)

    if ( nCloseTracks_Xcm > 0 && cfg_L1TkTrack_ApplyVtxIso == true){ bPassedVtxIso = false; } 
  
    /// L1CaloTau: Matching with L1TkTau
    double minDeltaR               = 99999.9;
    unsigned int iMatchedL1CaloTau = 0; 
    unsigned int iL1CaloTauIndex   = -1;
    std::vector< l1extra::L1JetParticle >::const_iterator L1CaloTau_Closest = h_L1CaloTau->end();
    bool bFoundCaloMatch = false;
    std::vector< L1TkTrackRefPtr > L1TkTau_TkPtrs; 


    // ------------ L1TkTau Cluster: Calo-Matching ------------ //
    /// For-Loop (nested): L1CaloTaus 
    for( std::vector< l1extra::L1JetParticle >::const_iterator L1CaloTau = h_L1CaloTau->begin(); L1CaloTau != h_L1CaloTau->end(); L1CaloTau++){

      iL1CaloTauIndex++; //starts at zero
      
      /// Calculate distance of L1CaloTau from the Leading Track of the L1TkTau Cluster
      double L1CaloTau_Eta = L1CaloTau->eta();
      double L1CaloTau_Phi = L1CaloTau->phi();
      double deltaR   = reco::deltaR(ldgTrack_Eta, ldgTrack_Phi, L1CaloTau_Eta, L1CaloTau_Phi);

      double L1CaloTau_Et  = L1CaloTau->et();
      if ( L1CaloTau_Et < L1CaloTau_EtMin){ continue; }


      /// Update closest L1TkTrack
      if ( deltaR < minDeltaR ){
	minDeltaR         = deltaR;
	L1CaloTau_Closest = L1CaloTau;
	iMatchedL1CaloTau = iL1CaloTauIndex; 
      }

    } /// For-Loop (nested): L1CaloTaus
    
    /// Ensuse that the L1CaloTau closest to the L1TkTau is at least within a certain matching cone size. Create a L1CaloTau Ref Pointer                                         
    if ( minDeltaR < cfg_DeltaR_L1TkTau_L1CaloTau ){                                                                                                                                  
      bFoundCaloMatch = true;
      L1TkTau_TkPtrs  = it->second;
    }

    /// Apply Vtx-Based isolation on L1TkTaus. Also ensure a L1CaloTau has been matched to a L1TkTau
    bool bFillEvent =  bPassedVtxIso * bFoundCaloMatch;
    if(  bFillEvent == false ) { continue; }
    
    /// Save the L1TkTau candidate's L1CaloTau P4, its L1CaloTau reference, its associated tracks and a dumbie isolation variable
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
    L1TkTauParticle L1TkTauFromCalo( L1TauCaloRef->p4(),
				     L1TauCaloRef,
				     L1TkTau_TkPtrs[0],
				     L1TkTau_TkPtrs[1],
				     L1TkTau_TkPtrs[2],
				     L1TkTau_TkIsol );

    result -> push_back( L1TkTauFromCalo );
    
  } /// For-loop: m_L1TkClusterToL1TkTracks elements
  
  sort( result->begin(), result->end(), L1TkTau::EtComparator() );

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
