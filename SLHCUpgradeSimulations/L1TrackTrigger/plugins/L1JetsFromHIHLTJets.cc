// -*- C++ -*-
//
//
// Produces a collection of L1JetParticles starting from a
// collection of reco:CaloJets, as created by the HLT
// Heavy Ion jet algorithm.
// 

// system include files
#include <memory>

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

#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

#include "Math/LorentzVector.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/JetReco/interface/CaloJet.h"

#include <string>
#include "TMath.h"


using namespace l1extra ;
using namespace math;
using namespace reco;

//
// class declaration
//

class L1JetsFromHIHLTJets : public edm::EDProducer {
   public:

      explicit L1JetsFromHIHLTJets(const edm::ParameterSet&);
      ~L1JetsFromHIHLTJets();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      //virtual void beginRun(edm::Run&, edm::EventSetup const&);
      //virtual void endRun(edm::Run&, edm::EventSetup const&);
      //virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
      //virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);

      // ----------member data ---------------------------
	
	 edm::InputTag HIJetsInputTag;

	 float ETAMIN;
	 float ETAMAX;


} ;


//
// constructors and destructor
//
L1JetsFromHIHLTJets::L1JetsFromHIHLTJets(const edm::ParameterSet& iConfig)
{

   HIJetsInputTag = iConfig.getParameter<edm::InputTag>("HIJetsInputTag");

   ETAMIN = (float)iConfig.getParameter<double>("ETAMIN");
   ETAMAX = (float)iConfig.getParameter<double>("ETAMAX");

   produces<L1JetParticleCollection>();
}

L1JetsFromHIHLTJets::~L1JetsFromHIHLTJets() {
}

// ------------ method called to produce the data  ------------
void
L1JetsFromHIHLTJets::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;

 std::auto_ptr<L1JetParticleCollection> result(new L1JetParticleCollection);

 edm::Handle< vector<reco::CaloJet> > HLTJetsHandle;
 iEvent.getByLabel(HIJetsInputTag,HLTJetsHandle);

 vector<reco::CaloJet>::const_iterator hltIterj;

  for (hltIterj = HLTJetsHandle->begin(); hltIterj != HLTJetsHandle->end(); ++hltIterj) {

   float etajet = hltIterj -> eta();
   if (fabs(etajet) < ETAMIN || fabs(etajet) > ETAMAX)  continue;

   //LorentzVector P4 = hltIterj -> detectorP4();

   edm::Ref< L1GctJetCandCollection > dummyRef;

   int bx = 0;
   L1JetParticle jet( hltIterj -> detectorP4() , dummyRef, bx); 
   result -> push_back( jet );

 }

 iEvent.put( result );

}

// --------------------------------------------------------------------------------------


// ------------ method called once each job just before starting event loop  ------------
void
L1JetsFromHIHLTJets::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1JetsFromHIHLTJets::endJob() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
L1JetsFromHIHLTJets::beginRun(edm::Run& iRun, edm::EventSetup const& iSetup)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
L1JetsFromHIHLTJets::endRun(edm::Run&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
L1JetsFromHIHLTJets::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
L1JetsFromHIHLTJets::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1JetsFromHIHLTJets::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1JetsFromHIHLTJets);



