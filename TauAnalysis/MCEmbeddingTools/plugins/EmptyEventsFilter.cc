// -*- C++ -*-
//
// Package:    EmptyEventsFilter
// Class:      EmptyEventsFilter
// 
/**\class EmptyEventsFilter EmptyEventsFilter.cc z2tautau/EmptyEventsFilter/src/EmptyEventsFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Manuel Zeise
//         Created:  Wed Oct 17 10:06:52 CEST 2007
// $Id: EmptyEventsFilter.cc,v 1.8 2011/04/27 15:54:05 fruboes Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <vector>
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

//#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

//
// class declaration
//

class EmptyEventsFilter : public edm::EDFilter {
   public:
      explicit EmptyEventsFilter(const edm::ParameterSet&);
      ~EmptyEventsFilter();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      edm::InputTag src_;
      int target_;

      int evTotal_;
      int evSelected_;
      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
EmptyEventsFilter::EmptyEventsFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
   src_ = iConfig.getUntrackedParameter<edm::InputTag>("src",edm::InputTag("generatorSmeared"));
   target_ = iConfig.getUntrackedParameter<int>("target",0);
   evTotal_ = 0;
   evSelected_ = 0;
}


EmptyEventsFilter::~EmptyEventsFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
EmptyEventsFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  ++evTotal_;
	using namespace edm;
	using namespace std;
//	using namespace HepMC;

	bool found = false;
	switch (target_)
	{
		case 1:
		{
			Handle<edm::HepMCProduct> dataHandle ;
			iEvent.getByLabel(src_, dataHandle ) ;
			if (dataHandle.isValid())
				found = true;	
			break;
		}
		case 0:
		default:
		{	
			Handle<std::vector<reco::Muon> > dataHandle;
			iEvent.getByLabel(src_, dataHandle ) ;
			if (dataHandle.isValid())
				found = true;
		}
	}

// 	Handle<edm::HepMCProduct> HepMCHandle;
// 	iEvent.getByLabel("newSource",HepMCHandle);
// 
// 	HepMC::GenEvent * evt = new HepMC::GenEvent(*(HepMCHandle->GetEvent()));
// 	evt->print(std::cout);

        if (!found) {
                return false;
        }
        else {
                ++evSelected_;
                return true;
        }

}

// ------------ method called once each job just before starting event loop  ------------
void 
EmptyEventsFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EmptyEventsFilter::endJob() {
  std::cout << "EmptyEventsFilter:: " 
      << double(evSelected_)/evTotal_
      << " " << evSelected_
      << " " << evTotal_
      << std::endl;

}

DEFINE_FWK_MODULE(EmptyEventsFilter);
