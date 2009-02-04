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
// $Id: HSCPFilter.cc,v 1.4 2008/08/26 14:09:25 arizzi Exp $
//
//


// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"

#include "SUSYBSMAnalysis/HSCP/interface/HSCPFilter.h"

using namespace reco;
using namespace std;

HSCPFilter::HSCPFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
  m_dedxCut1=iConfig.getParameter<double>("DeDxMin1");
  m_tkPCut1=iConfig.getParameter<double>("PMin1");
  m_dedxCut2=iConfig.getParameter<double>("DeDxMin2");
  m_tkPCut2=iConfig.getParameter<double>("PMin2");
  m_dedxCut3=iConfig.getParameter<double>("DeDxMin3");
  m_tkPCut3=iConfig.getParameter<double>("PMin3");

  m_singleMuPtMin =iConfig.getParameter<double>("SingleMuPtMin");
  m_doubleMuPtMin =iConfig.getParameter<double>("DoubleMuPtMin");

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

   // dE/dx from the tracker
   Handle<DeDxDataValueMap> dedxH;
   iEvent.getByLabel("dedxTruncated40",dedxH);
   const ValueMap<DeDxData> dEdxTrack = *dedxH.product();
   edm::Handle<reco::TrackCollection> trackCollectionHandle;
   iEvent.getByLabel("TrackRefitter",trackCollectionHandle);
   for(unsigned int i=0; i<trackCollectionHandle->size(); i++) {
     reco::TrackRef track  = reco::TrackRef( trackCollectionHandle, i );
     const DeDxData& dedx = dEdxTrack[track];
     if( (track->normalizedChi2() < 5 && track->numberOfValidHits()>=8 ) //quality cuts
         &&
         ((track->p() > m_tkPCut1 && dedx.dEdx()> m_dedxCut1) ||
          (track->p() > m_tkPCut2 && dedx.dEdx()> m_dedxCut2) ||
          (track->p() > m_tkPCut3 && dedx.dEdx()> m_dedxCut2)   ) //slow particle  cuts
       ) return true;
   }
   
   // Pt of muon tracks
   Handle<TrackCollection> muonsH;
   iEvent.getByLabel("standAloneMuons",muonsH);
   const TrackCollection & muons = *muonsH.product();
   int found=0;
   for(size_t i=0; i<muons.size() ; i++) { 
      if(muons[i].pt() > m_singleMuPtMin ) return true;
      if(muons[i].pt() > m_doubleMuPtMin ) found++;
      if(found >=2) return true;
   }

  // failed all
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
