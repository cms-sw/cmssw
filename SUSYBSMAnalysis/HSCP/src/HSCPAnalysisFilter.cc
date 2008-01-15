// -*- C++ -*-
//
// Package:    HSCPAnalysisFilter
// Class:      HSCPAnalysisFilter
// 
/**\class HSCPAnalysisFilter HSCPAnalysisFilter.cc SUSYBSMAnalysis/HSCPAnalysisFilter/src/HSCPAnalysisFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Rizzi
//         Created:  Tue Jun 26 11:37:21 CEST 2007
// $Id: HSCPAnalysisFilter.cc,v 1.3 2007/08/20 08:43:43 arizzi Exp $
//
//


// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SUSYBSMAnalysis/HSCP/interface/HSCPAnalysisFilter.h"

#include "DataFormats/TrackReco/interface/TrackDeDxEstimate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"

#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"
using namespace susybsm;


using namespace reco;
using namespace std;

HSCPAnalysisFilter::HSCPAnalysisFilter(const edm::ParameterSet& iConfig)
{

}


HSCPAnalysisFilter::~HSCPAnalysisFilter()
{
 

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HSCPAnalysisFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
using namespace edm;
   Handle<HSCParticleCollection> hscpH;
   iEvent.getByLabel("hscp",hscpH);
   const vector<HSCParticle> & candidates = *hscpH.product();

for(int i=0; i < candidates.size();i++)
{

 if(candidates[i].hasDt && candidates[i].hasMuonTrack() && candidates[i].massDt() > 100) return true;
 if(candidates[i].hasDt && candidates[i].massDtSta() > 100) return true;
 if(candidates[i].hasTk && candidates[i].hasDt && candidates[i].massTk() > 100 && candidates[i].dt.first->standAloneMuon()->pt() > 100 && candidates[i].tk.invBeta2 > 1.4  )return true;
 
}
 
  return false;
}

// ------------ method called once each job just before starting event loop  ------------
void 
HSCPAnalysisFilter::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HSCPAnalysisFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HSCPAnalysisFilter);
