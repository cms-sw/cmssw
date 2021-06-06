#include "RecoVertex/KalmanVertexFit/plugins/KVFTrackUpdate.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "RecoVertex/KalmanVertexFit/interface/SingleTrackVertexConstraint.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <iostream>

using namespace reco;
using namespace edm;
using namespace std;

KVFTrackUpdate::KVFTrackUpdate(const edm::ParameterSet& iConfig)
    : estoken_TTB(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))) {
  token_tracks = consumes<TrackCollection>(iConfig.getParameter<InputTag>("TrackLabel"));
  token_beamSpot = consumes<BeamSpot>(iConfig.getParameter<InputTag>("beamSpotLabel"));
}

KVFTrackUpdate::~KVFTrackUpdate() {}

void KVFTrackUpdate::beginJob() {}

void KVFTrackUpdate::endJob() {}

//
// member functions
//

void KVFTrackUpdate::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  try {
    edm::LogInfo("RecoVertex/KVFTrackUpdate") << "Reconstructing event number: " << iEvent.id() << "\n";

    // get RECO tracks from the event
    // `tks` can be used as a ptr to a reco::TrackCollection
    edm::Handle<reco::TrackCollection> tks;
    iEvent.getByToken(token_tracks, tks);

    edm::LogInfo("RecoVertex/KVFTrackUpdate") << "Found: " << (*tks).size() << " reconstructed tracks"
                                              << "\n";
    edm::LogPrint("RecoVertex/KVFTrackUpdate") << "got " << (*tks).size() << " tracks " << std::endl;

    // Transform Track to TransientTrack

    //get the builder:
    const auto& theB = &iSetup.getData(estoken_TTB);
    //do the conversion:
    std::vector<TransientTrack> t_tks = theB->build(tks);

    edm::LogInfo("RecoVertex/KVFTrackUpdate") << "Found: " << t_tks.size() << " reconstructed tracks"
                                              << "\n";

    GlobalPoint glbPos(0., 0., 0.);

    AlgebraicSymMatrix33 mat;
    mat[0][0] = (20.e-04) * (20.e-04);
    mat[1][1] = (20.e-04) * (20.e-04);
    mat[2][2] = (5.3) * (5.3);
    GlobalError glbErrPos(mat);

    reco::BeamSpot vertexBeamSpot;
    edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
    iEvent.getByToken(token_beamSpot, recoBeamSpotHandle);

    SingleTrackVertexConstraint stvc;
    for (unsigned int i = 0; i < t_tks.size(); i++) {
      SingleTrackVertexConstraint::BTFtuple a = stvc.constrain(t_tks[i], glbPos, glbErrPos);
      edm::LogPrint("RecoVertex/KVFTrackUpdate") << "Chi2: " << std::get<2>(a) << std::endl;
      if (recoBeamSpotHandle.isValid()) {
        SingleTrackVertexConstraint::BTFtuple b = stvc.constrain(t_tks[i], *recoBeamSpotHandle);
        edm::LogPrint("RecoVertex/KVFTrackUpdate") << "Chi2: " << std::get<2>(b) << std::endl;
      }
    }
  }

  catch (std::exception& err) {
    edm::LogInfo("RecoVertex/KVFTrackUpdate") << "Exception during event number: " << iEvent.id() << "\n"
                                              << err.what() << "\n";
  }
}

DEFINE_FWK_MODULE(KVFTrackUpdate);
