// -*- C++ -*-
//
// Package:    HSCPFilter
// Class:      HSCPFilter
// 
/**\class HSCPFilter HSCPFilter.cc SUSYBSMAnalysis/HSCPFilter/src/HSCPFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Rizzi
//         Created:  Tue Jun 26 11:37:21 CEST 2007
// $Id: HSCPFilter.cc,v 1.1 2007/06/27 13:35:36 arizzi Exp $
//
//


// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "SUSYBSMAnalysis/HSCP/interface/HSCPFilter.h"

#include "DataFormats/TrackReco/interface/TrackDeDxEstimate.h"
#include "DataFormats/TrackReco/interface/Track.h"
using namespace reco;
using namespace std;

HSCPFilter::HSCPFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed

}


HSCPFilter::~HSCPFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HSCPFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   Handle<TrackDeDxEstimateCollection> dedxH;
   iEvent.getByLabel("dedxTruncated40",dedxH);
   const TrackDeDxEstimateCollection & dedx = *dedxH.product();
   for(size_t i=0; i<dedx.size() ; i++)
    {
      if(dedx[i].first->pt() > 10) cout << dedx[i].second << endl;
      if(dedx[i].first->pt() > 10 && dedx[i].second> 4. ) return true;
    }
 
  return false;
}

// ------------ method called once each job just before starting event loop  ------------
void 
HSCPFilter::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HSCPFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HSCPFilter);
