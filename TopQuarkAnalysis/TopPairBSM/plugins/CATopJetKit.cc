#include "TopQuarkAnalysis/TopPairBSM/interface/CATopJetKit.h"

using namespace std;
using namespace pat;


//
// constructors and destructor
//
CATopJetKit::CATopJetKit(const edm::ParameterSet& iConfig)
  :
  verboseLevel_(0),
  helper_(iConfig)
{
  helper_.bookHistos(this);
}


CATopJetKit::~CATopJetKit()
{
}


//
// member functions
//

// ------------ method called to for each event  ------------
// void CATopJetKit::analyze(const edm::Event& evt, const edm::EventSetup& iSetup)
void CATopJetKit::produce( edm::Event & evt, const edm::EventSetup & es )
{
  using namespace edm;
  using namespace std;

  if ( verboseLevel_ > 10 )
    std::cout << "CATopJetKit:: in analyze()." << std::endl;

  // --------------------------------------------------
  //    Step 1: Retrieve objects from data stream
  // --------------------------------------------------
  helper_.getHandles( evt,
		      muonHandle_,
		      electronHandle_,
		      tauHandle_,
		      jetHandle_,
		      METHandle_,
		      photonHandle_,
		      trackHandle_,
		      genParticles_);

  
  cout << "Processing " << jetHandle_->size() << " jets" << endl;

  // --------------------------------------------------
  //    Step 2: invoke PhysicsHistograms to deal with all this.
  //
  //    Note that each handle will dereference into a vector<>,
  //    however the fillCollection() method takes a reference,
  //    so the collections are not copied...
  // --------------------------------------------------
  if ( verboseLevel_ > 10 )
    std::cout << "CATopJetKit::analyze: calling fillCollection()." << std::endl;
  helper_.fillHistograms( evt,
			  muonHandle_,
			  electronHandle_,
			  tauHandle_,
			  jetHandle_,
			  METHandle_,
			  photonHandle_,
			  trackHandle_,
			  genParticles_);
}






// ------------ method called once each job just before starting event loop  ------------
void
CATopJetKit::beginJob(const edm::EventSetup&)
{
}



// ------------ method called once each job just after ending the event loop  ------------
void
CATopJetKit::endJob() {
}



//define this as a plug-in
DEFINE_FWK_MODULE(CATopJetKit);
