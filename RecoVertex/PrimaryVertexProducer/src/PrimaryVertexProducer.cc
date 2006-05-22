#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexProducer.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertError.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"

using namespace reco;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PrimaryVertexProducer::PrimaryVertexProducer(const edm::ParameterSet& conf)
  : theConfig(conf)
{
  edm::LogInfo("RecoVertex/PrimaryVertexProducer") 
    << "Initializing PV producer " << "\n";

  produces<VertexCollection>("PrimaryVertex");
  
  // initialization of vertex finder algorithm
  // configurable parameters
  float ptMin = theConfig.getParameter<double>("TrackPtMin");
  theFinder.setPtCut(ptMin);
  float minTrackCompatibilityToMainVertex 
    = theConfig.getParameter<double>("MinTrackCompatibilityToMainVertex");
  theFinder.setTrackCompatibilityCut(minTrackCompatibilityToMainVertex);
  float minTrackCompatibilityToOtherVertex 
    = theConfig.getParameter<double>("MinTrackCompatibilityToOtherVertex");
  theFinder.setTrackCompatibilityToSV(minTrackCompatibilityToOtherVertex);
  int maxNbVertices = theConfig.getParameter<int>("MaxNbVertices");
  theFinder.setMaxNbOfVertices(maxNbVertices);

  // FIXME introduce track selection, track clustering 
  // and beam compatibility check

}


PrimaryVertexProducer::~PrimaryVertexProducer()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
PrimaryVertexProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  std::auto_ptr<reco::VertexCollection> result(new reco::VertexCollection);
  reco::VertexCollection vColl;

  try {
    edm::LogInfo("RecoVertex/PrimaryVertexProducer") 
      << "Reconstructing event number: " << iEvent.id() << "\n";
    
    // get RECO tracks from the event
    edm::Handle<reco::TrackCollection> trackCollection;
    iEvent.getByLabel("Tracks", trackCollection);
    //    iEvent.getByType(trackCollection);
    //    iEvent.getByLabel("Tracks", "recoTracks", trackCollection);
    //    std::vector< edm::Handle<reco::TrackCollection> > trackCollections;
    //    iEvent.getManyByType(trackCollections);
    //    trackCollection = trackCollections[1];

    reco::TrackCollection tks = *(trackCollection.product());
    
    // interface RECO tracks to vertex reconstruction
    vector<TransientTrack> t_tks;
    for (reco::TrackCollection::const_iterator it = tks.begin();
	 it != tks.end(); it++) {
      t_tks.push_back(*it);
    }
    edm::LogInfo("RecoVertex/PrimaryVertexProducer") 
      << "Found: " << t_tks.size() << " reconstructed tracks" << "\n";
    
    // here call vertex reconstruction

    vector<TransientVertex> t_vts = theFinder.vertices(t_tks);
    for (vector<TransientVertex>::const_iterator iv = t_vts.begin();
	 iv != t_vts.end(); iv++) {
      Vertex v(Vertex::Point((*iv).position()), 
	       RecoVertex::convertError((*iv).positionError()), 
	       (*iv).totalChiSquared(), 
	       (*iv).degreesOfFreedom() , 
      (*iv).originalTracks().size());
      vColl.push_back(v);
    }
    
    /*
    // test with vertex fitter
    if (t_tks.size() > 1) {
      KalmanVertexFitter kvf;
      TransientVertex tv = kvf.vertex(t_tks);
      Vertex v(Vertex::Point(tv.position()), 
	       RecoVertex::convertError(tv.positionError()), 
	       (tv).totalChiSquared(), 
	       (tv).degreesOfFreedom() , 
	       (tv).originalTracks().size());
      vColl.push_back(v);
    }
    */
  }

  catch (std::exception & err) {
    edm::LogInfo("RecoVertex/PrimaryVertexProducer") 
      << "Exception during event number: " << iEvent.id() 
      << "\n" << err.what() << "\n";
  }

  *result = vColl;
  iEvent.put(result, "PrimaryVertex");
  
}

//define this as a plug-in
DEFINE_FWK_MODULE(PrimaryVertexProducer)
