// -*- C++ -*-
//
//
// Producer for a L1TkTauParticle from matching L1 Tracks to L1 Calo Taus
// The code matches the highest Et L1CaloTaus in an event with high pT L1TkTracks.
// A signal cone is created around the matching track and more L1TkTracks can be added 
// to this cluster if the user-defined criteria are satisfied. An isolation annulus is built
// around the signal cone and the L1TkTkracks found inside it are used to determine whether 
// the user-defined isolation criteria are satisfied.

// system include files
#include <memory>
#include <string>
#include "TMath.h"
#include <vector>
#include <TLorentzVector.h>

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

//#define DEBUG

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
  
  class TrackPtComparator{
    
    unsigned int nFitParams_;
  public:
    TrackPtComparator(unsigned int nFitParams){ nFitParams_ = nFitParams;}
    bool operator() (const L1TkTrackRefPtr trackA, L1TkTrackRefPtr trackB ) const {
      return ( trackA->getMomentum(nFitParams_).perp() > trackB->getMomentum(nFitParams_).perp() );
    }
  };
  
  struct EtComparator{ 
    bool operator() (const l1extra::L1JetParticle tauA, const l1extra::L1JetParticle tauB) const { return ( tauA.et() > tauB.et() ); }
  };

private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

bool GetTkTauFromCaloIsolation(const L1TkTrackRefPtr_Collection c_L1Tks,
			       const std::string iso_type, 
			       const double isolation_cut, 
			       const std::vector<int> sigTks_index, 
			       const std::vector<int> isoTks_index);

  void GetTracksInIsolationCone(const L1TkTrackRefPtr_Collection c_L1Tks,
				const double matchingTk_index, 
				const double deltaR_Min,
				const double deltaR_Max,
				const std::vector<int> sigTks_Index,
				std::vector<int> &isoTks_Index);

  void GetTracksInSignalCone(const L1TkTrackRefPtr_Collection c_L1Tks,
			     const double matchingTk_index, 
			     const double deltaR_Min,
			     const double deltaR_Max,
			     const double invMass_max,
			     const double deltaPOCAz_max,
			     std::vector<int> usedTks_Index,
			     std::vector<int> &sigTks_Index);

  
  // ---------- member data  ---------- //
  /// Parameters whose values are given through the python cfg file
  edm::InputTag cfg_L1CaloTaus_InputTag;            // Tag of Collection to be used for L1 Calorimeter Taus
  edm::InputTag cfg_L1TkTracks_InputTag;            // Tag of Collection to be used for L1 Track Trigger Tracks
  edm::InputTag cfg_L1TkPV_InputTag;                // Tag of Collection to be used for the L1Tk PV Collection
  unsigned int cfg_L1TkTracks_NFitParameters;       // Number of Fit Parameters: 4 or 5 ? (pT, eta, phi, z0, d0)
  double cfg_L1TkTracks_PtMin;                      // Min pT applied on all L1TkTracks [GeV]
  double cfg_L1TkTracks_AbsEtaMax;                  // Max |eta| applied on all L1TkTracks [unitless]
  double cfg_L1TkTracks_POCAzMax;                   // Max POCA-z for L1TkTracks [cm]
  unsigned int cfg_L1TkTracks_NStubsMin;            // Min number of stubs per L1TkTrack [unitless]
  double cfg_L1TkTracks_RedChiSquareBarrelMax;      // Max chi squared for L1TkTracks in Barrel [unitless]
  double cfg_L1TkTracks_RedChiSquareEndcapMax;      // Max chi squared for L1TkTracks in Endcap [unitless]
  double cfg_L1TkTracks_DeltaPOCAzFromPV;           // Max |POCA-z - Vertex-z| [cm]
  double cfg_L1TkTau_MatchingTk_PtMin;              // Min pT of L1CaloTau-matching L1TkTracks [GeV]
  double cfg_L1TkTau_MatchingTk_DeltaRMax;          // Cone size for matching L1TkTracks to L1CaloTaus [unitless]
  double cfg_L1TkTau_SignalTks_PtMin;               // Min pT of L1TkTau signal-cone L1TkTracks [GeV]
  double cfg_L1TkTau_SignalTks_DeltaRMax;           // Max opening of the L1TkTau signal-cone for adding L1TkTracks [unitless]
  double cfg_L1TkTau_SignalTks_InvMassMax;          // Max invariant mass of the L1TkTau when adding L1TkTracks [GeV/c^2]
  double cfg_L1TkTau_SignalTks_DeltaPOCAzMax;       // Max POCAz difference between MatchingTk and additional L1TkTau signal-cone L1TkTracks [cm]
  double cfg_L1TkTau_IsolationTks_PtMin;            // Min pT of L1TkTau isolation-annulus L1TkTracks [GeV]
  double cfg_L1TkTau_IsolationTks_DeltaRMax;        // Max opening of the L1TkTau isolation-annulus [unitless]
  double cfg_L1TkTau_IsolationTks_DeltaPOCAzMax;    // Max POCAz difference between MatchingTk and L1TkTracks in isolation cone [cm]
  
} ;


// ------------ constructor  ------------ //
L1TkTauFromCaloProducer::L1TkTauFromCaloProducer(const edm::ParameterSet& iConfig){

  cfg_L1CaloTaus_InputTag                   = iConfig.getParameter<edm::InputTag>("L1CaloTaus_InputTag");
  cfg_L1TkTracks_InputTag                   = iConfig.getParameter<edm::InputTag>("L1TkTracks_InputTag");
  cfg_L1TkPV_InputTag                       = iConfig.getParameter<edm::InputTag>("L1TkPV_InputTag");
  cfg_L1TkTracks_NFitParameters             = iConfig.getParameter< uint32_t >("L1TkTracks_NFitParameters");
  cfg_L1TkTracks_PtMin                      = iConfig.getParameter< double >("L1TkTracks_PtMin");
  cfg_L1TkTracks_AbsEtaMax                  = iConfig.getParameter< double >("L1TkTracks_AbsEtaMax");
  cfg_L1TkTracks_POCAzMax                   = iConfig.getParameter< double >("L1TkTracks_POCAzMax");
  cfg_L1TkTracks_NStubsMin                  = (unsigned int)iConfig.getParameter< uint32_t >("L1TkTracks_NStubsMin");
  cfg_L1TkTracks_RedChiSquareBarrelMax      = iConfig.getParameter< double >("L1TkTracks_RedChiSquareBarrelMax");
  cfg_L1TkTracks_RedChiSquareEndcapMax      = iConfig.getParameter< double >("L1TkTracks_RedChiSquareEndcapMax"); 
  cfg_L1TkTracks_DeltaPOCAzFromPV           = iConfig.getParameter< double >("L1TkTracks_DeltaPOCAzFromPV");
  cfg_L1TkTau_MatchingTk_PtMin              = iConfig.getParameter< double >("L1TkTau_MatchingTk_PtMin");
  cfg_L1TkTau_MatchingTk_DeltaRMax          = iConfig.getParameter< double >("L1TkTau_MatchingTk_DeltaRMax");
  cfg_L1TkTau_SignalTks_PtMin               = iConfig.getParameter< double >("L1TkTau_SignalTks_PtMin");
  cfg_L1TkTau_SignalTks_DeltaRMax           = iConfig.getParameter< double >("L1TkTau_SignalTks_DeltaRMax");
  cfg_L1TkTau_SignalTks_InvMassMax          = iConfig.getParameter< double >("L1TkTau_SignalTks_InvMassMax");
  cfg_L1TkTau_SignalTks_DeltaPOCAzMax       = iConfig.getParameter< double >("L1TkTau_SignalTks_DeltaPOCAzMax");
  cfg_L1TkTau_IsolationTks_PtMin            = iConfig.getParameter< double >("L1TkTau_IsolationTks_PtMin");
  cfg_L1TkTau_IsolationTks_DeltaRMax        = iConfig.getParameter< double >("L1TkTau_IsolationTks_DeltaRMax");
  cfg_L1TkTau_IsolationTks_DeltaPOCAzMax    = iConfig.getParameter< double >("L1TkTau_IsolationTks_DeltaPOCAzMax");
  
  produces<L1TkTauParticleCollection>();
}


// ------------ destructor  ------------ //
L1TkTauFromCaloProducer::~L1TkTauFromCaloProducer(){}

// ------------ method called to produce the data  ------------ //
void L1TkTauFromCaloProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){
  
  using namespace edm;
  using namespace std;
  std::auto_ptr<L1TkTauParticleCollection> result(new L1TkTauParticleCollection);
  
  // ------------ Primary Vertex using L1TkTracks  ------------ //
  edm::Handle< std::vector< L1TkPrimaryVertex > > h_L1TkPVs;
  iEvent.getByLabel( cfg_L1TkPV_InputTag, h_L1TkPVs );
  std::vector< L1TkPrimaryVertex > c_L1TkPVs;

  // double PV_SumPt   = 0.0;
  double PV_ZVertex = 0.0;

  if ( cfg_L1TkTracks_DeltaPOCAzFromPV < 10000) {   // such that the code can run
						    // without any vertex collection when this
						    // cut is actually not applied.

  /// For-loop: PVs (size should be one)
  for ( std::vector< L1TkPrimaryVertex >::const_iterator PV = h_L1TkPVs->begin();  PV != h_L1TkPVs->end(); PV++){
    
    // PV_SumPt   = PV->getSum();
    PV_ZVertex = PV->getZvertex();
    break;

  } // For-Loop: PVs

  } // endif

  // ------------ L1TkTracks  ------------ //
  edm::Handle< L1TkTrack_Collection > h_PixelDigi_L1TkTrack;
  iEvent.getByLabel( cfg_L1TkTracks_InputTag, h_PixelDigi_L1TkTrack );

  L1TkTrackRefPtr_Collection c_L1TkTracks;
  std::vector< L1TkTrackRefPtr > L1TkTau_TkPtrs;
							
  unsigned int track_counter = 0;
  /// For-loop: L1TkTracks
  for ( L1TkTrack_Collection::const_iterator track = h_PixelDigi_L1TkTrack->begin(); track != h_PixelDigi_L1TkTrack->end(); track++){
  
    /// Make a pointer to the L1TkTracks
    L1TkTrackRefPtr track_RefPtr( h_PixelDigi_L1TkTrack, track_counter++);
    
    /// Declare for-loop variables
    std::vector< L1TkStubRef > track_Stubs = track -> getStubRefs();
    unsigned int track_NStubs              = track_Stubs.size();
    double track_Pt                        = track->getMomentum(cfg_L1TkTracks_NFitParameters).perp();
    double track_Eta                       = track->getMomentum(cfg_L1TkTracks_NFitParameters).eta();
    double track_POCAz                     = track->getPOCA(cfg_L1TkTracks_NFitParameters).z();
    double track_RedChiSq                  = track->getChi2Red(cfg_L1TkTracks_NFitParameters);
    bool track_HasEndcapStub = false;
    bool track_HasBarrelStub = false;

    /// Apply L1TkTracks quality criteria
    const bool bPassStubCut = (track_NStubs >= cfg_L1TkTracks_NStubsMin);
    if (!bPassStubCut) continue;

    const bool bPassPtCut = (track_Pt >= cfg_L1TkTracks_PtMin);
    if (!bPassPtCut) continue; 

    const bool bPassEtaCut = (fabs(track_Eta) <= fabs(cfg_L1TkTracks_AbsEtaMax) );
    if (!bPassEtaCut) continue;

    const bool bPassPOCAzCut = (fabs(track_POCAz) <= fabs(cfg_L1TkTracks_POCAzMax) );
    if (!bPassPOCAzCut) continue; 

    /// For-loop: Stubs
    for ( unsigned int iStub = 0; iStub < track_NStubs; iStub++){
      
      /// Get the detector id for the stub (barrel / endcap)
      StackedTrackerDetId thisDetId( track_Stubs.at(iStub)->getDetId() );
      if ( thisDetId.isEndcap() ){ track_HasEndcapStub = true; }
      if ( thisDetId.isBarrel() ){ track_HasBarrelStub = true; }	

    } /// For-loop: Stubs

    /// Apply stubs quality criteria
    bool bPassRedChiSqCut = false;
    if ( ( track_HasBarrelStub ) && ( track_RedChiSq <= cfg_L1TkTracks_RedChiSquareBarrelMax ) ) bPassRedChiSqCut = true;
    if ( ( track_HasEndcapStub ) && ( track_RedChiSq <= cfg_L1TkTracks_RedChiSquareEndcapMax ) ) bPassRedChiSqCut = true;
    if(!bPassRedChiSqCut) continue;


    /// Apply PV distance requirement
    bool bPassPVzDistanceCut = false;
    if ( fabs(PV_ZVertex - track_POCAz) <= cfg_L1TkTracks_DeltaPOCAzFromPV) bPassPVzDistanceCut = true;
    if(!bPassPVzDistanceCut) continue;

    /// Save the user-defined quality L1TkTracks
    c_L1TkTracks.push_back( track_RefPtr );
    
  } // For-loop: L1TkTracks

  /// Sort by pT all selected L1TkTracks
  std::sort( c_L1TkTracks.begin(), c_L1TkTracks.end(), TrackPtComparator(cfg_L1TkTracks_NFitParameters) );

#ifdef DEBUG
  for (unsigned int i = 0; i < c_L1TkTracks.size(); i++) { std::cout << "c_L1TkTracks.at(" << i << ").pt() = " << c_L1TkTracks.at(i)->getMomentum(cfg_L1TkTracks_NFitParameters).perp() << std::endl;}
  std::cout << "\n" << std::endl; 
#endif

  // ------------ L1CaloTaus (First 12 leading) ------------ //
  edm::Handle< std::vector< l1extra::L1JetParticle > > h_L1CaloTau;
  iEvent.getByLabel( cfg_L1CaloTaus_InputTag, h_L1CaloTau );
  std::vector< l1extra::L1JetParticle > c_L1CaloTaus;
  
  /// Sort L1CaloTau collections in descending Et and truncate the collection to keep only the first 12 leading jets
  for ( std::vector< l1extra::L1JetParticle >::const_iterator L1CaloTau = h_L1CaloTau->begin();  L1CaloTau != h_L1CaloTau->end(); L1CaloTau++){
    c_L1CaloTaus.push_back( *L1CaloTau );
  }
  std::sort( c_L1CaloTaus.begin(), c_L1CaloTaus.end(), EtComparator() );
  c_L1CaloTaus.resize(12);



  // ------------ TkConfirmed L1CaloTaus ------------ //
  unsigned int calo_counter = 0;
  std::vector<int> usedSigTks_index;

  /// For-loop: L1CaloTaus (First 12 Leading L1CaloTaus)
  for( std::vector< l1extra::L1JetParticle >::const_iterator L1CaloTau = c_L1CaloTaus.begin(); L1CaloTau != c_L1CaloTaus.end(); L1CaloTau++ , calo_counter++){

    /// Get basic properties
    double L1CaloTau_Et  = L1CaloTau->et();
    double L1CaloTau_Eta = L1CaloTau->eta();
    double L1CaloTau_Phi = L1CaloTau->phi();

    // Initialise variables
    double deltaR_min    = 9999.9;
    int MatchingTk_index = -1.0;
    std::vector<int> sigTks_index;
    std::vector<int> isoTks_index;

    /// For-loop: Selected L1TkTracks
    for ( unsigned int i=0; i < c_L1TkTracks.size(); i++ ){

      L1TkTrackRefPtr iTk       = c_L1TkTracks.at(i);
      double MatchingTkCand_Pt  = iTk->getMomentum(cfg_L1TkTracks_NFitParameters).perp();
      double MatchingTkCand_Eta = iTk->getMomentum(cfg_L1TkTracks_NFitParameters).eta();
      double MatchingTkCand_Phi = iTk->getMomentum(cfg_L1TkTracks_NFitParameters).phi();
      
      // Apply pT cut for matching L1TkTracks
      const bool bPassPtCut = (MatchingTkCand_Pt >= cfg_L1TkTau_MatchingTk_PtMin);
      if (!bPassPtCut) continue; 
      
      // Check if track has already being matched to a L1CaloTau
      bool bIsUsedTk = std::find(usedSigTks_index.begin(), usedSigTks_index.end(), i) != usedSigTks_index.end();
      if (bIsUsedTk) continue;
      
      // Calculate distance between MatchingTkCand and L1CaloTau
      double deltaR = reco::deltaR(MatchingTkCand_Eta, MatchingTkCand_Phi, L1CaloTau_Eta, L1CaloTau_Phi);
      
      // Find track which is closest to the calo
      if (deltaR < deltaR_min && deltaR < cfg_L1TkTau_MatchingTk_DeltaRMax) {
	deltaR_min       = deltaR;
	MatchingTk_index = i;
      }

    } /// For-loop: Selected L1TkTracks


    // Check if Calo is TkConfirmed. If not go to the next L1CaloTau
    bool bIsTkConfirmed = MatchingTk_index > -1;
    if (!bIsTkConfirmed) continue;
    

    // Determine the L1TkTau size of the shrinking signal cone
    double signalCone_deltaR_max = (3.5)/(L1CaloTau_Et);
    if (signalCone_deltaR_max > cfg_L1TkTau_SignalTks_DeltaRMax) signalCone_deltaR_max = cfg_L1TkTau_SignalTks_DeltaRMax;


    // Get the signal/isolation tracks
    GetTracksInSignalCone(c_L1TkTracks, MatchingTk_index, 0.0, signalCone_deltaR_max, cfg_L1TkTau_SignalTks_InvMassMax, cfg_L1TkTau_SignalTks_DeltaPOCAzMax, usedSigTks_index, sigTks_index);
    GetTracksInIsolationCone(c_L1TkTracks, MatchingTk_index, signalCone_deltaR_max, cfg_L1TkTau_IsolationTks_DeltaRMax, sigTks_index, isoTks_index);

    /// Save all the signal-cone tracks and keep track of them through usedSigTks
    for ( unsigned int i=0; i < sigTks_index.size(); i++){ L1TkTau_TkPtrs.push_back( c_L1TkTracks.at( sigTks_index.at(i) ) ); }
    usedSigTks_index.insert(usedSigTks_index.end(), sigTks_index.begin(), sigTks_index.end());

#ifdef DEBUG
    for (unsigned int i = 0; i < sigTks_index.size(); i++) { std::cout << "s) c_L1TkTracks.at(" << sigTks_index.at(i) << ").pt() = " << c_L1TkTracks.at(sigTks_index.at(i))->getMomentum(cfg_L1TkTracks_NFitParameters).perp() << std::endl;}
    for (unsigned int i = 0; i < isoTks_index.size(); i++) { std::cout << "iso) c_L1TkTracks.at(" << isoTks_index.at(i) << ").pt() = " << c_L1TkTracks.at(isoTks_index.at(i))->getMomentum(cfg_L1TkTracks_NFitParameters).perp() << std::endl;}
    std::cout << "\n" << std::endl; 
#endif

    // Apply isolation criteria
    bool bPassedIso = GetTkTauFromCaloIsolation(c_L1TkTracks, "Vertex", cfg_L1TkTau_IsolationTks_DeltaPOCAzMax, sigTks_index, isoTks_index);
    if(!bPassedIso) continue;

    /// Save the L1TkTau candidate's L1CaloTau P4, its L1CaloTau reference, its associated tracks and a dumbie isolation variable
    float L1TkTau_TkIsol = -999.9;
    edm::Ref< L1JetParticleCollection > L1CaloTauRef( h_L1CaloTau, calo_counter );
        
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

    /// Fill The L1TkTauParticle object
    L1TkTauParticle L1TkTauFromCalo( L1CaloTauRef->p4(), L1CaloTauRef, L1TkTau_TkPtrs[0], L1TkTau_TkPtrs[1], L1TkTau_TkPtrs[2], L1TkTau_TkIsol );
    result -> push_back( L1TkTauFromCalo );

  } /// For-Loop: L1CaloTaus
  
  // Sort by eT before saving to the event (should be unnecessary since L1CaloTaus are already eT-sorted)
  sort( result->begin(), result->end(), L1TkTau::EtComparator() );
  iEvent.put( result );

}


// ------------ get indices of tracks in signal cone ------------ //
void L1TkTauFromCaloProducer::GetTracksInSignalCone(const L1TkTrackRefPtr_Collection c_L1Tks,
						    const double matchingTk_index, 
						    const double deltaR_min,
						    const double deltaR_max,
						    const double invMass_max,
						    const double deltaPOCAz_max,
						    std::vector<int> usedSigTks_Index,
						    std::vector<int> &sigTks_Index)
{

  // Add the matchingTk to the signal tracks
  if (matchingTk_index < 0) return;
  else sigTks_Index.push_back(matchingTk_index);

  // Get matchingTk properties
  L1TkTrackRefPtr matchingTk    = c_L1Tks.at(matchingTk_index);
  const double matchingTk_Pt    = matchingTk->getMomentum(cfg_L1TkTracks_NFitParameters).perp();
  const double matchingTk_Eta   = matchingTk->getMomentum(cfg_L1TkTracks_NFitParameters).eta();
  const double matchingTk_Phi   = matchingTk->getMomentum(cfg_L1TkTracks_NFitParameters).phi();
  const double matchingTk_Mass  = 0.13957018; // assume charged pion
  const double matchingTk_POCAz = matchingTk->getPOCA(cfg_L1TkTracks_NFitParameters).z();
  TLorentzVector tau_p4;
  tau_p4.SetPtEtaPhiM(matchingTk_Pt, matchingTk_Eta, matchingTk_Phi, matchingTk_Mass);

  // For-Loop: L1Tks
  for ( unsigned int i=0; i < c_L1Tks.size(); i++){

    // Check if track has already being used
    bool bIsOwnSigTk  = std::find(sigTks_Index.begin(), sigTks_Index.end(), i) != sigTks_Index.end();
    bool bIsUsedSigTk = std::find(usedSigTks_Index.begin(), usedSigTks_Index.end(), i) != usedSigTks_Index.end();
    if(bIsOwnSigTk) continue;
    if(bIsUsedSigTk) continue;

    // Is this track within the signal cone?
    L1TkTrackRefPtr iTk   = c_L1Tks.at(i);
    const double tk_Pt    = iTk->getMomentum(cfg_L1TkTracks_NFitParameters).perp();
    const double tk_Eta   = iTk->getMomentum(cfg_L1TkTracks_NFitParameters).eta();
    const double tk_Phi   = iTk->getMomentum(cfg_L1TkTracks_NFitParameters).phi();
    const double tk_Mass  = 0.13957018; // assume charged pion
    const double tk_POCAz = iTk->getPOCA(cfg_L1TkTracks_NFitParameters).z();
    double deltaR = reco::deltaR(matchingTk_Eta, matchingTk_Phi, tk_Eta, tk_Phi);

    TLorentzVector tmp_p4;
    tmp_p4.SetPtEtaPhiM(tk_Pt, tk_Eta, tk_Phi, tk_Mass);
    
    bool bIsInsideCone = ( (deltaR <= deltaR_max) && (deltaR >= deltaR_min) );
    if(!bIsInsideCone) continue;

    // Apply deltaPOCAz requirement
    double deltaPOCAz    = matchingTk_POCAz - tk_POCAz;
    double absDeltaPOCAz = fabs(deltaPOCAz);
    bool bPassPOCAzCut   = (absDeltaPOCAz <= deltaPOCAz_max);
    if (!bPassPOCAzCut) continue;

    // Apply InvMass requirement
    tau_p4 += tmp_p4;
    if (tau_p4.M() > invMass_max){
      tau_p4 -= tmp_p4;
      continue;
    }

    // Save the track as a signal-cone track
    sigTks_Index.push_back(i);
    
  } // For-Loop: L1Tks 
  
  return;

}


// ------------ get indices of tracks in isolation cone ------------ //
void L1TkTauFromCaloProducer::GetTracksInIsolationCone(const L1TkTrackRefPtr_Collection c_L1Tks,
						       const double matchingTk_index, 
						       const double deltaR_min,
						       const double deltaR_max,
						       const std::vector<int> sigTks_Index,
						       std::vector<int> &isoTks_Index)
{
  
  // If L1CaloTau is not TkConfirmed no point of applying isolation criteria
  if (matchingTk_index < 0) return;

  // For-Loop: L1Tks
  for ( unsigned int j=0; j < c_L1Tks.size(); j++){
    
    L1TkTrackRefPtr matchingTk = c_L1Tks.at(matchingTk_index);
    double matchingTk_Eta      = matchingTk->getMomentum(cfg_L1TkTracks_NFitParameters).eta();
    double matchingTk_Phi      = matchingTk->getMomentum(cfg_L1TkTracks_NFitParameters).phi();
    
    bool bIsOwnSigTk  = std::find(sigTks_Index.begin(), sigTks_Index.end(), j) != sigTks_Index.end();
    bool bIsUsedIsoTk = std::find(isoTks_Index.begin(), isoTks_Index.end(), j) != isoTks_Index.end();
    if (bIsOwnSigTk) continue;
    if (bIsUsedIsoTk) continue;
    
    // Is tk within a signal cone (or annulus)?
    L1TkTrackRefPtr iTk = c_L1Tks.at(j);
    double tk_Eta       = iTk->getMomentum(cfg_L1TkTracks_NFitParameters).eta();
    double tk_Phi       = iTk->getMomentum(cfg_L1TkTracks_NFitParameters).phi();      
    double deltaR       = reco::deltaR(tk_Eta, tk_Phi, matchingTk_Eta, matchingTk_Phi);
    bool bIsInsideCone  = (deltaR <= deltaR_max) && (deltaR >= deltaR_min);
    if (!bIsInsideCone) continue;
    
    // Save the track as an isolation-cone track
    isoTks_Index.push_back(j);
    
  } // For-Loop: L1Tks
  
  return;
}


// ------------ get indices of tracks in isolation cone ------------ //
bool L1TkTauFromCaloProducer::GetTkTauFromCaloIsolation(const L1TkTrackRefPtr_Collection c_L1Tks,
							const std::string iso_type,
							const double isolation_cut, 
							const std::vector<int> sigTks_index, 
							const std::vector<int> isoTks_index)
{
  
  if (sigTks_index.size() < 1) return false;
  if (isoTks_index.size() < 1) return true;
  if (isolation_cut < 0.0) return true;
  
  // Initialise variables
  bool bIsIsolated         = false;
  bool bIsVtxIsolated      = false;
  bool bIsRelIsolated      = false;
  double isoTk_scalarSumPt = 0.0; 
  int nIsoTksWithinPOCAz   = 0; 
  
  // Get L1CaloTau matching track properties
  const int matchingTk_index    = sigTks_index.at(0);
  L1TkTrackRefPtr matchingTk    = c_L1Tks.at(matchingTk_index);
  const double matchingTk_Pt    = matchingTk->getMomentum(cfg_L1TkTracks_NFitParameters).perp();
  const double matchingTk_POCAz = matchingTk->getPOCA(cfg_L1TkTracks_NFitParameters).z();
  
  // For-loop: IsoTks
  for (Size_t iTk = 0; iTk < isoTks_index.size(); iTk++) { 
    
    int isoTk_Index       = isoTks_index.at(iTk);
    L1TkTrackRefPtr isoTk = c_L1Tks.at(isoTk_Index);
    double isoTk_Pt       = isoTk->getMomentum(cfg_L1TkTracks_NFitParameters).perp();
    double isoTk_POCAz    = isoTk->getPOCA(cfg_L1TkTracks_NFitParameters).z();    

    // Calculate isolation variables: VtxIso (no isoTks within X cm of POCAz of matchingTk)
    double deltaPOCAz     = matchingTk_POCAz - isoTk_POCAz;
    double absDeltaPOCAz  = fabs(deltaPOCAz);

    // Calculate isolation variables: RelIso (add up the pT of all tracks in isolation annulus)
    isoTk_scalarSumPt += isoTk_Pt;

    // If the isoTk is close to the ldgtk then the TkTau is not isolated
    if ( absDeltaPOCAz <= isolation_cut) nIsoTksWithinPOCAz++;        
    
  }// For-loop: IsoTks

  
  // Calculated relative isolation
  double relIso = isoTk_scalarSumPt/matchingTk_Pt;

  // Determine if TkTau is relatively-isolated
  if (relIso <= isolation_cut) bIsRelIsolated = true;

  // Determine if TkTau is vertex-isolated
  if (nIsoTksWithinPOCAz == 0) bIsVtxIsolated = true;
  
  // Determine return value based on "Isolation Type"
  if (iso_type.compare("Vertex") == 0)        bIsIsolated = bIsVtxIsolated;
  else if (iso_type.compare("Relative") == 0) bIsIsolated = bIsRelIsolated;
  else {
    throw cms::Exception("Logic") << "Invalid isolation type selected \"" <<  iso_type << "\". Please choose from \"Vertex\" and \"Relative\"." << std::endl;
  }
  
  return bIsIsolated;
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


// Define this as a plug-in
DEFINE_FWK_MODULE(L1TkTauFromCaloProducer);
