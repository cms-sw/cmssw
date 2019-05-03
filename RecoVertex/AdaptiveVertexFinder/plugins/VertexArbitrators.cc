#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoVertex/AdaptiveVertexFinder/plugins/TemplatedVertexArbitrator.h"

//#define VTXDEBUG

typedef TemplatedVertexArbitrator<reco::TrackCollection, reco::Vertex>
    TrackVertexArbitrator;
typedef TemplatedVertexArbitrator<edm::View<reco::Candidate>,
                                  reco::VertexCompositePtrCandidate>
    CandidateVertexArbitrator;

DEFINE_FWK_MODULE(TrackVertexArbitrator);
DEFINE_FWK_MODULE(CandidateVertexArbitrator);
