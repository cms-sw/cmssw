#include "RecoVertex/GaussianSumVertexFit/test/TrackMix.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"

#include <iostream>

using namespace reco;
using namespace edm;
using namespace std;

TrackMix::TrackMix(const edm::ParameterSet& iConfig)
  : theConfig(iConfig)
{
  gsfTrackLabel_ = iConfig.getParameter<std::string>("gsfTrackLabel");
  ckfTrackLabel_ = iConfig.getParameter<std::string>("ckfTrackLabel");
}


TrackMix::~TrackMix() {
}

void TrackMix::beginJob(){
}


void TrackMix::endJob() {}

void
TrackMix::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{



  try {
    edm::LogInfo("RecoVertex/TrackMix") 
      << "Reconstructing event number: " << iEvent.id() << "\n";
    
    // get RECO tracks from the event
    // `tks` can be used as a ptr to a reco::TrackCollection
    edm::Handle<edm::View<reco::Track> > tks;
    iEvent.getByLabel(gsfTrackLabel_, tks);
    edm::Handle<edm::View<reco::Track> > tks2;
    iEvent.getByLabel(ckfTrackLabel_, tks2);

    cout << "got " << (*tks).size() << " gsf tracks " << endl;
    cout << "got " << (*tks2).size()<< " ckf tracks " << endl;

    // Transform Track to TransientTrack

    //get the builder:
    edm::ESHandle<TransientTrackBuilder> theB;
    iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theB);
    //do the conversion:
    vector<TransientTrack> t_tks = (*theB).build(tks);
    vector<TransientTrack> t_tks2 = (*theB).build(tks2);
    t_tks.insert(t_tks.end(), t_tks2.begin(), t_tks2.end());

    cout  << "Total: " << t_tks.size() << " reconstructed tracks" << "\n";
    
    // Call the KalmanVertexFitter if more than 1 track
    if (t_tks.size() > 1) {
      KalmanVertexFitter kvf(true);
      TransientVertex tv = kvf.vertex(t_tks);

      std::cout << "Position: " << Vertex::Point(tv.position()) << "\n";

      reco::Vertex v1 = tv;
      reco::Vertex::trackRef_iterator v1TrackIter;
      reco::Vertex::trackRef_iterator v1TrackBegin = v1.tracks_begin();
      reco::Vertex::trackRef_iterator v1TrackEnd   = v1.tracks_end();
      cout << v1.position()<<v1.tracksSize()<<endl;
            for (v1TrackIter = v1TrackBegin; v1TrackIter != v1TrackEnd; v1TrackIter++) {
	    cout << "pt" << (**v1TrackIter).pt() <<endl;
	    cout << " weight " << v1.trackWeight(*v1TrackIter)<<endl;
	    cout << " ref " << v1.refittedTrack(*v1TrackIter).pt()<<endl;
      }


    }
    
  }

  catch (std::exception & err) {
    cout  << "Exception during event number: " << iEvent.id() 
      << "\n" << err.what() << "\n";
  }

}

DEFINE_FWK_MODULE(TrackMix);
