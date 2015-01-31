// -*- C++ -*-
//
// Package:    PionTracksProducer
// Class:      PionTracksProducer
// 
/**\class PionTracksProducer PionTracksProducer.cc TrackPropagation/PionTracksProducer/src/PionTracksProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Lorenzo Viliani,32 3-B06,+41227676396,
//         Created:  Wed Aug  6 16:14:28 CEST 2014
// $Id$
//
//


// system include files
#include <memory>
#include <vector>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
//#include "DataFormats/VertexReco/interface/Vertex.h"
//#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"

//
// class declaration
//

class PionTracksProducer : public edm::EDFilter {
   public:
      explicit PionTracksProducer(const edm::ParameterSet&);
      ~PionTracksProducer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;
      
      virtual void beginRun(edm::Run&, edm::EventSetup const&);
      virtual void endRun(edm::Run&, edm::EventSetup const&);
      virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
      virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);

      // ----------member data ---------------------------
      edm::InputTag src_;
      bool filter_;
     // typedef reco::TrackRef PionTrack;
     // typedef std::vector<PionTrack> PionTracksCollection;
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
PionTracksProducer::PionTracksProducer(const edm::ParameterSet& iConfig)
{

  src_  = iConfig.getParameter<edm::InputTag>( "src" );
  filter_  = iConfig.getParameter<bool>( "filter" );
  produces<reco::TrackCollection>( "pionTrack" ).setBranchAlias("pionTracks");
}


PionTracksProducer::~PionTracksProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
bool
PionTracksProducer::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco; 
   using namespace std;

   Handle<reco::VertexCompositeCandidateCollection> vertex;
   iEvent.getByLabel( src_, vertex );
   auto_ptr<TrackCollection> pionTracks( new TrackCollection );   
   
   for(VertexCompositeCandidateCollection::const_iterator itVertex = vertex->begin();
       itVertex != vertex->end();
       ++itVertex ) {
       
       const Candidate* cand1 = itVertex->CompositeCandidate::daughter(0);
       const Candidate* cand2 = itVertex->CompositeCandidate::daughter(1);
       TrackRef trkRef1 = cand1->get<TrackRef>(); 
       TrackRef trkRef2 = cand2->get<TrackRef>();
       const Track* trk1 = trkRef1.get();
       const Track* trk2 = trkRef2.get();
       pionTracks->push_back( *trk1 );
       pionTracks->push_back( *trk2 );

   }
  
   unsigned int size = pionTracks->size();  

   iEvent.put( pionTracks, "pionTrack" );
  
   if (filter_ &&  size < 2)
     return false;
   
   return true;

}

// ------------ method called once each job just before starting event loop  ------------
void 
PionTracksProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
PionTracksProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void 
PionTracksProducer::beginRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void 
PionTracksProducer::endRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
PionTracksProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
PionTracksProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
PionTracksProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PionTracksProducer);
