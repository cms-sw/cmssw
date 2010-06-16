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
// $Id: EmptyEventsFilter.cc,v 1.5 2010/05/26 10:19:43 fruboes Exp $
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

      int minEvents_;
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
   minEvents_ = iConfig.getUntrackedParameter<int>("minEvents",1);
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

	int num = 0;
	switch (target_)
	{
		case 1:
		{
			std::vector< Handle<edm::HepMCProduct> > dataHandles ;
			iEvent.getManyByType( dataHandles ) ;
			num = dataHandles.size();
 //	  	cout << dataHandles.size() << " Produkte gefunden ******************* ^_^ ***\n";
			break;
		}
		case 0:
		default:
		{	
			std::vector< Handle<std::vector<reco::Muon> >  > dataHandles;
			iEvent.getManyByType( dataHandles ) ;
			num = dataHandles.size();
	//  	cout << dataHandles.size() << " Produkte gefunden ******************* ^_^ ***\n";
		}
	}

// 	Handle<edm::HepMCProduct> HepMCHandle;
// 	iEvent.getByLabel("newSource",HepMCHandle);
// 
// 	HepMC::GenEvent * evt = new HepMC::GenEvent(*(HepMCHandle->GetEvent()));
// 	evt->print(std::cout);

        if (num<minEvents_)
                return false;
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
