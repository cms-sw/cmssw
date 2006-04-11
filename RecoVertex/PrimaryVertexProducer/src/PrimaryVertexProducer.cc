#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexProducer.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoVertex/TrimmedKalmanVertexFinder/interface/KalmanTrimmedVertexFinder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertError.h"

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
PrimaryVertexProducer::PrimaryVertexProducer(const edm::ParameterSet& iConfig)
{

  produces<VertexCollection>("PrimaryVertex");
  
  // initialization of vertex finder algorithm
  theFinder = new KalmanTrimmedVertexFinder();

  // FIXME introduce nested configurable parameters

  // FIXME introduce track selection and beam compatibility check

}


PrimaryVertexProducer::~PrimaryVertexProducer()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  delete theFinder;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
PrimaryVertexProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  // get RECO tracks from the event
  reco::TrackCollection tks;

  // interface RECO tracks to vertex reconstruction
  vector<TransientTrack> t_tks;
  for (reco::TrackCollection::const_iterator it = tks.begin();
       it != tks.end(); it++) {
    t_tks.push_back(*it);
  }

  //  reco::Vertex::Point pos(-1, -1, -1);
  //  double e[6]; reco::Vertex::Error err(e);
  //  double chi2 = -1; double ndof = 1; double ntks = 0;
  
  std::auto_ptr<reco::VertexCollection> result(new reco::VertexCollection); // empty vertex collection,on the heap ??
  // here call vertex reconstruction
  vector<TransientVertex> t_vts = (*theFinder).vertices(t_tks);

  reco::VertexCollection vColl;
  for (vector<TransientVertex>::const_iterator iv = t_vts.begin();
       iv != t_vts.end(); iv++) {
    Vertex v(Vertex::Point((*iv).position()), 
	     RecoVertex::convertError((*iv).positionError()), 
	     (*iv).totalChiSquared(), 
	     (*iv).degreesOfFreedom() , 
	     (*iv).originalTracks().size());
    vColl.push_back(v);
  }

  //  reco::Vertex v(pos, err, chi2, ndof, ntks);
  //  tmpVColl.push_back(v);
  *result = vColl;
  iEvent.put(result, "PrimaryVertex");
  
}

//define this as a plug-in
DEFINE_FWK_MODULE(PrimaryVertexProducer)
