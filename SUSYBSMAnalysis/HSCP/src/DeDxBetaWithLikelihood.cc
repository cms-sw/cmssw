// -*- C++ -*-
//
// Package:    DeDxBetaWithLikelihood
// Class:      DeDxBetaWithLikelihood
// 
/**\class DeDxBetaWithLikelihood DeDxBetaWithLikelihood.cc SUSYBSMAnalysis/DeDxBetaWithLikelihood/src/DeDxBetaWithLikelihood.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Rizzi Andrea
//         Created:  Wed Oct 10 12:01:28 CEST 2007
// $Id: DeDxBetaWithLikelihood.cc,v 1.1 2007/10/10 10:14:20 arizzi Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "RecoTracker/DeDx/interface/DeDxEstimatorProducer.h"
#include "DataFormats/TrackReco/interface/TrackDeDxEstimate.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include "DataFormats/TrackReco/interface/Track.h"


#include <vector>
#include <TNtuple.h>
#include <TF1.h>
#include <iostream>
//
// class decleration
//

class DeDxBetaWithLikelihood : public edm::EDProducer {
   public:
      explicit DeDxBetaWithLikelihood(const edm::ParameterSet&);
      ~DeDxBetaWithLikelihood();

   private:
      float fit(const reco::TrackDeDxHits &);
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      edm::InputTag m_trackDeDxHitsTag;
   
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
DeDxBetaWithLikelihood::DeDxBetaWithLikelihood(const edm::ParameterSet& iConfig)
{

   m_trackDeDxHitsTag = iConfig.getParameter<edm::InputTag>("trackDeDxHits");
   produces<std::vector<float> >();

}


DeDxBetaWithLikelihood::~DeDxBetaWithLikelihood()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
DeDxBetaWithLikelihood::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace edm;
   using namespace reco;

   edm::Handle<reco::TrackDeDxHitsCollection> trackDeDxHitsCollectionHandle;
   iEvent.getByLabel(m_trackDeDxHitsTag,trackDeDxHitsCollectionHandle);
   const reco::TrackDeDxHitsCollection & hits = *trackDeDxHitsCollectionHandle.product();
  vector<float> * outputCollection = new vector<float>;

   reco::TrackDeDxHitsCollection::const_iterator it= hits.begin();
   for(int j=0;it!=hits.end();++it,j++)
   {
      float val=fit(*it);
      outputCollection->push_back(val);
   }

    std::auto_ptr<vector<float> > estimator(outputCollection);
    iEvent.put(estimator);

}

float DeDxBetaWithLikelihood::fit(const reco::TrackDeDxHits & dedxVec)
{
 using namespace std;
 TNtuple tmpNt("dedx","dedx","dedx");
 double mpv=-5.;
 double chi=-5.;

 // copy data into a tree:
 for (unsigned int i=0; i<dedxVec.second.size();i++) {
// cout << dedxVec.second[i].charge() << endl;
   tmpNt.Fill(dedxVec.second[i].charge());
 }
// stupid->Scan();
 // fit:
 TF1* f1 = new TF1("f1", "landaun");
 f1->SetParameters(1, 3.0 , 0.3);
 f1->SetParLimits(0, 1, 1); // fix the normalization parameter to 1
 int status = tmpNt.UnbinnedFit("f1","dedx","","Q");
 mpv = f1->GetParameter(1);
 if (status<=0) {
   cout << "(AnalyzeTracks::LandauFit) no convergence!   status = " << status << endl;
 tmpNt.Scan();
    return 0;
 }
 delete f1;
 return mpv;
}

// ------------ method called once each job just before starting event loop  ------------
void 
DeDxBetaWithLikelihood::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
DeDxBetaWithLikelihood::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeDxBetaWithLikelihood);
