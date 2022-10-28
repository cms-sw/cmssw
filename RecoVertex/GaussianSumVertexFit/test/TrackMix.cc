/**\class TrackMix TrackMix.cc RecoVertex/TrackMix/src/TrackMix.cc

 Description: Simple test to see that reco::Vertex can store tracks from different
 Collections and of different types in the same vertex.
*/

// system include files
#include <memory>
#include <iostream>

// user include files
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

// ROOT includes
#include <TFile.h>

/**
   * This is a very simple test analyzer mean to test the KalmanVertexFitter
   */

class TrackMix : public edm::one::EDAnalyzer<> {
public:
  explicit TrackMix(const edm::ParameterSet&);
  ~TrackMix();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:
  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> estoken_ttk;
  edm::EDGetTokenT<edm::View<reco::Track> > token_gsf, token_ckf;
};

using namespace reco;
using namespace edm;
using namespace std;

TrackMix::TrackMix(const edm::ParameterSet& iConfig)
    : estoken_ttk(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))) {
  token_gsf = consumes<edm::View<reco::Track> >(iConfig.getParameter<string>("gsfTrackLabel"));
  token_ckf = consumes<edm::View<reco::Track> >(iConfig.getParameter<string>("ckfTrackLabel"));
}

TrackMix::~TrackMix() = default;

void TrackMix::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  try {
    edm::LogInfo("RecoVertex/TrackMix") << "Reconstructing event number: " << iEvent.id() << "\n";

    // get RECO tracks from the event
    // `tks` can be used as a ptr to a reco::TrackCollection
    edm::Handle<edm::View<reco::Track> > tks;
    iEvent.getByToken(token_gsf, tks);
    edm::Handle<edm::View<reco::Track> > tks2;
    iEvent.getByToken(token_ckf, tks2);

    edm::LogPrint("TrackMix") << "got " << (*tks).size() << " gsf tracks " << endl;
    edm::LogPrint("TrackMix") << "got " << (*tks2).size() << " ckf tracks " << endl;

    // Transform Track to TransientTrack

    //get the builder:
    edm::ESHandle<TransientTrackBuilder> theB = iSetup.getHandle(estoken_ttk);
    //do the conversion:
    vector<TransientTrack> t_tks = (*theB).build(tks);
    vector<TransientTrack> t_tks2 = (*theB).build(tks2);
    t_tks.insert(t_tks.end(), t_tks2.begin(), t_tks2.end());

    edm::LogPrint("TrackMix") << "Total: " << t_tks.size() << " reconstructed tracks"
                              << "\n";

    // Call the KalmanVertexFitter if more than 1 track
    if (t_tks.size() > 1) {
      KalmanVertexFitter kvf(true);
      TransientVertex tv = kvf.vertex(t_tks);

      edm::LogPrint("TrackMix") << "Position: " << Vertex::Point(tv.position()) << "\n";

      reco::Vertex v1 = tv;
      reco::Vertex::trackRef_iterator v1TrackIter;
      reco::Vertex::trackRef_iterator v1TrackBegin = v1.tracks_begin();
      reco::Vertex::trackRef_iterator v1TrackEnd = v1.tracks_end();
      edm::LogPrint("TrackMix") << v1.position() << v1.tracksSize() << endl;
      for (v1TrackIter = v1TrackBegin; v1TrackIter != v1TrackEnd; v1TrackIter++) {
        edm::LogPrint("TrackMix") << "pt" << (**v1TrackIter).pt() << endl;
        edm::LogPrint("TrackMix") << " weight " << v1.trackWeight(*v1TrackIter) << endl;
        edm::LogPrint("TrackMix") << " ref " << v1.refittedTrack(*v1TrackIter).pt() << endl;
      }
    }
  } catch (cms::Exception& err) {
    edm::LogError("TrackMix") << "Exception during event number: " << iEvent.id() << "\n" << err.what() << "\n";
  }
}

DEFINE_FWK_MODULE(TrackMix);
