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

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/L1TkTauEtComparator.h"

using namespace l1extra ;

// ---------- class declaration  ---------- //
class L1CaloTausToTkTausTranslator : public edm::EDProducer {
public:
  
  typedef TTTrack< Ref_PixelDigi_ > L1TkTrackType;
  typedef edm::Ptr< L1TkTrackType > L1TkTrackRefPtr;
  typedef std::vector< L1TkTrackType > L1TkTrackCollectionType;
  typedef edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > L1TkStubRef; //new
  
  explicit L1CaloTausToTkTausTranslator(const edm::ParameterSet&);
  ~L1CaloTausToTkTausTranslator();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions); 
  
private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  // ---------- member data  ---------- //
  edm::InputTag L1TausInputTag;
  
} ;


// ------------ constructor  ------------ //
L1CaloTausToTkTausTranslator::L1CaloTausToTkTausTranslator(const edm::ParameterSet& iConfig){

  L1TausInputTag                        = iConfig.getParameter<edm::InputTag>("L1TausInputTag");

  produces<L1TkTauParticleCollection>();

}


// ------------ destructor  ------------ //
L1CaloTausToTkTausTranslator::~L1CaloTausToTkTausTranslator(){}

// ------------ method called to produce the data  ------------ //
void L1CaloTausToTkTausTranslator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){
  
  using namespace edm;
  using namespace std;
  std::auto_ptr<L1TkTauParticleCollection> result(new L1TkTauParticleCollection);
  
  // Collection Handles: The L1CaloTau from the L1 ExtraParticles
  edm::Handle< std::vector< l1extra::L1JetParticle > > h_L1CaloTau;
  iEvent.getByLabel( L1TausInputTag, h_L1CaloTau );

  int iL1CaloTauIndex=0;

  // Nested for-loop: L1CaloTaus
  for ( std::vector< l1extra::L1JetParticle >::const_iterator L1CaloTau = h_L1CaloTau->begin();  L1CaloTau != h_L1CaloTau->end(); L1CaloTau++){
    
    edm::Ref< L1JetParticleCollection > L1TauCaloRef( h_L1CaloTau, iL1CaloTauIndex ); 
    
    edm::Ptr< L1TkTrackType > L1TrackPtrNull;    
    
    L1TkTauParticle L1TkTauFromCalo( L1CaloTau->p4(),
				     L1TauCaloRef,
				     L1TrackPtrNull,
				     L1TrackPtrNull,
				     L1TrackPtrNull,
				     0.0);
    
    result -> push_back( L1TkTauFromCalo );
    
    iL1CaloTauIndex++; //starts at 0
    
  }
  
  sort( result->begin(), result->end(), L1TkTau::EtComparator() );

  iEvent.put( result );

}


// ------------ method called once each job just before starting event loop  ------------ //
void L1CaloTausToTkTausTranslator::beginJob(){}

// ------------ method called once each job just after ending the event loop  ------------ //
void L1CaloTausToTkTausTranslator::endJob() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------ //
void L1CaloTausToTkTausTranslator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1CaloTausToTkTausTranslator);



