// -*- C++ -*-
//
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

#include "DataFormats/L1TrackTrigger/interface/L1TkMuonParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkMuonParticleFwd.h"

#include "DataFormats/L1TrackTrigger/interface/L1TkPrimaryVertex.h"

#include "DataFormats/Math/interface/LorentzVector.h"


#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include <string>
#include "TMath.h"


using namespace l1extra ;
using namespace edm;
using namespace std;

//
// class declaration
//

class L1TkMuonMerger : public edm::EDProducer {
   public:

   typedef TTTrack< Ref_PixelDigi_ >  L1TkTrackType;
   typedef std::vector< L1TkTrackType >  L1TkTrackCollectionType;

      explicit L1TkMuonMerger(const edm::ParameterSet&);
      ~L1TkMuonMerger();

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
	
	std::vector<edm::InputTag> TkMuonCollections;
	std::vector<double> absEtaMin;
	std::vector<double> absEtaMax;

} ;


//
// constructors and destructor
//
L1TkMuonMerger::L1TkMuonMerger(const edm::ParameterSet& iConfig)
{

   TkMuonCollections = iConfig.getParameter< std::vector<edm::InputTag> >("TkMuonCollections");
   absEtaMin = iConfig.getParameter< std::vector<double> >("absEtaMin");
   absEtaMax = iConfig.getParameter< std::vector<double> >("absEtaMax");
 
   int n1 = TkMuonCollections.size();
   int n2 = absEtaMin.size();
   int n3 = absEtaMax.size();
   if ( (n1 != n2) || (n1 != n3) ) {
          edm::LogError("L1TkMuonMerger")
	    << "\n Error in configuration. TkMuonCollections, absEtaMin and absEtaMax should have the sme size. Exit. "
            << std::endl;
   }

   produces<L1TkMuonParticleCollection>();
}

L1TkMuonMerger::~L1TkMuonMerger() {
}

// ------------ method called to produce the data  ------------
void
L1TkMuonMerger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;


 std::auto_ptr<L1TkMuonParticleCollection> result(new L1TkMuonParticleCollection);

  int ncollections = TkMuonCollections.size();
  //cout << " There are " << ncollections << " collections to merge " << std::endl;

  for (int icol=0; icol < ncollections; icol++) {

   edm::InputTag MuonInputTag = TkMuonCollections.at( icol );
   //cout << " Collection: " << MuonInputTag << endl;

   edm::Handle< vector<l1extra::L1TkMuonParticle>  > MuonHandle ;
   iEvent.getByLabel( MuonInputTag, MuonHandle );

   if (! MuonHandle.isValid() ) {
          LogError("L1TkMuonMerger")
            << "\nWarning: TkMuon collection  " << MuonInputTag
            << "\nrequested in configuration, but not found in the event. Skip it."
            << std::endl;
   }

   vector<l1extra::L1TkMuonParticle>::const_iterator l1MuIter;

  for (l1MuIter = MuonHandle->begin(); l1MuIter != MuonHandle->end(); ++l1MuIter) {
	float eta = l1MuIter -> eta();
	float feta = fabs(eta);
	if ( feta < absEtaMin[icol] || feta > absEtaMax[icol] ) continue;
	

	L1TkMuonParticle tkmuon = *l1MuIter;

	result -> push_back( tkmuon );
  }   // end collection
  }



 iEvent.put( result );

}

// --------------------------------------------------------------------------------------


// ------------ method called once each job just before starting event loop  ------------
void
L1TkMuonMerger::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1TkMuonMerger::endJob() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
L1TkMuonMerger::beginRun(edm::Run& iRun, edm::EventSetup const& iSetup)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
L1TkMuonMerger::endRun(edm::Run&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
L1TkMuonMerger::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
L1TkMuonMerger::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TkMuonMerger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TkMuonMerger);



