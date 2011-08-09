#include "RecoVertex/KalmanVertexFit/plugins/KVFTrackUpdate.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "RecoVertex/KalmanVertexFit/interface/SingleTrackVertexConstraint.h"

#include <iostream>

using namespace reco;
using namespace edm;
using namespace std;

KVFTrackUpdate::KVFTrackUpdate(const edm::ParameterSet& iConfig)
{
  trackLabel_ = iConfig.getParameter<edm::InputTag>("TrackLabel");
  beamSpotLabel = iConfig.getParameter<edm::InputTag>("beamSpotLabel");
}


KVFTrackUpdate::~KVFTrackUpdate() {
}

void KVFTrackUpdate::beginJob(){
}


void KVFTrackUpdate::endJob() {
}

//
// member functions
//

void
KVFTrackUpdate::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{



  try {
    edm::LogInfo("RecoVertex/KVFTrackUpdate") 
      << "Reconstructing event number: " << iEvent.id() << "\n";
    
    // get RECO tracks from the event
    // `tks` can be used as a ptr to a reco::TrackCollection
    edm::Handle<reco::TrackCollection> tks;
    iEvent.getByLabel(trackLabel_, tks);

    edm::LogInfo("RecoVertex/KVFTrackUpdate") 
      << "Found: " << (*tks).size() << " reconstructed tracks" << "\n";
    std::cout << "got " << (*tks).size() << " tracks " << std::endl;

    // Transform Track to TransientTrack

    //get the builder:
    edm::ESHandle<TransientTrackBuilder> theB;
    iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theB);
    //do the conversion:
    std::vector<TransientTrack> t_tks = (*theB).build(tks);

    edm::LogInfo("RecoVertex/KVFTrackUpdate") 
      << "Found: " << t_tks.size() << " reconstructed tracks" << "\n";
    
    GlobalPoint glbPos(0.,0.,0.);

    AlgebraicSymMatrix mat(3,0);
    mat[0][0] = (20.e-04)*(20.e-04);
    mat[1][1] = (20.e-04)*(20.e-04);
    mat[2][2] = (5.3)*(5.3);
    GlobalError glbErrPos(mat);

    reco::BeamSpot vertexBeamSpot;
    edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
    iEvent.getByLabel(beamSpotLabel,recoBeamSpotHandle);


    SingleTrackVertexConstraint stvc;
    for (unsigned int i = 0; i<t_tks.size();i++) {
      SingleTrackVertexConstraint::BTFtuple a = 
      	stvc.constrain(t_tks[i], glbPos, glbErrPos);
      std::cout << "Chi2: "<< a.get<2>()<<std::endl;
      if (recoBeamSpotHandle.isValid()){
	SingleTrackVertexConstraint::BTFtuple b =
	  stvc.constrain(t_tks[i], *recoBeamSpotHandle);
	std::cout << "Chi2: "<< b.get<2>()<<std::endl;
      }
    }
  }


  catch (std::exception & err) {
    edm::LogInfo("RecoVertex/KVFTrackUpdate") 
      << "Exception during event number: " << iEvent.id() 
      << "\n" << err.what() << "\n";
  }

}

DEFINE_FWK_MODULE(KVFTrackUpdate);
