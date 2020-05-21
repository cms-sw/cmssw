#include "RecoVertex/MultiVertexFit/interface/MultiVertexReconstructor.h"

using namespace std;

namespace {
  typedef MultiVertexFitter::TrackAndWeight TrackAndWeight;

  int verbose() { return 0; }

#ifndef __clang__
  inline void remove(vector<TransientVertex>& vtces, const vector<reco::TransientTrack>& trks) {
    cout << "[MultiVertexReconstructor] fixme remove not yet implemented" << endl;
    // remove trks from vtces
  }
#endif

  vector<vector<TrackAndWeight> > recover(const vector<TransientVertex>& vtces,
                                          const vector<reco::TransientTrack>& trks) {
    set<reco::TransientTrack> st;
    for (const auto& trk : trks) {
      st.insert(trk);
    }

    vector<vector<TrackAndWeight> > bundles;
    for (const auto& vtce : vtces) {
      vector<reco::TransientTrack> trks = vtce.originalTracks();
      vector<TrackAndWeight> tnws;
      for (const auto& trk : trks) {
        float w = vtce.trackWeight(trk);
        if (w > 1e-5) {
          TrackAndWeight tmp(trk, w);
          set<reco::TransientTrack>::iterator pos = st.find(trk);
          if (pos != st.end()) {
            st.erase(pos);
          }
          tnws.push_back(tmp);
        };
      };
      bundles.push_back(tnws);
    };

    if (bundles.empty())
      return bundles;

    // now add not-yet assigned tracks
    for (const auto& i : st) {
      // cout << "[MultiVertexReconstructor] recovering " << i->id() << endl;
      TrackAndWeight tmp(i, 0.);
      bundles[0].push_back(tmp);
    }
    return bundles;
  }
}  // namespace

MultiVertexReconstructor::MultiVertexReconstructor(const VertexReconstructor& o,
                                                   const AnnealingSchedule& s,
                                                   float revive)
    : theOldReconstructor(o.clone()), theFitter(MultiVertexFitter(s, DefaultLinearizationPointFinder(), revive)) {}

MultiVertexReconstructor::~MultiVertexReconstructor() { delete theOldReconstructor; }

MultiVertexReconstructor* MultiVertexReconstructor::clone() const { return new MultiVertexReconstructor(*this); }

MultiVertexReconstructor::MultiVertexReconstructor(const MultiVertexReconstructor& o)
    : theOldReconstructor(o.theOldReconstructor->clone()), theFitter(o.theFitter) {}

vector<TransientVertex> MultiVertexReconstructor::vertices(const vector<reco::TransientTrack>& trks,
                                                           const reco::BeamSpot& s) const {
  return vertices(trks);
}

vector<TransientVertex> MultiVertexReconstructor::vertices(const vector<reco::TransientTrack>& trks,
                                                           const vector<reco::TransientTrack>& primaries,
                                                           const reco::BeamSpot& s) const {
  return vertices(trks, primaries);
}

vector<TransientVertex> MultiVertexReconstructor::vertices(const vector<reco::TransientTrack>& trks) const {
  /*
  cout << "[MultiVertexReconstructor] input trks: ";
  for ( vector< reco::TransientTrack >::const_iterator i=trks.begin(); 
        i!=trks.end() ; ++i )
  {
    cout << i->id() << "  ";
  }
  cout << endl;*/
  vector<TransientVertex> tmp = theOldReconstructor->vertices(trks);
  if (verbose()) {
    cout << "[MultiVertexReconstructor] non-freezing seeder found " << tmp.size() << " vertices from " << trks.size()
         << " tracks." << endl;
  }
  vector<vector<TrackAndWeight> > rc = recover(tmp, trks);
  vector<CachingVertex<5> > cvts = theFitter.vertices(rc);
  vector<TransientVertex> ret;
  for (const auto& cvt : cvts) {
    ret.push_back(cvt);
  };

  if (verbose()) {
    cout << "[MultiVertexReconstructor] input " << tmp.size() << " vertices, output " << ret.size() << " vertices."
         << endl;
  };
  return ret;
}

vector<TransientVertex> MultiVertexReconstructor::vertices(const vector<reco::TransientTrack>& trks,
                                                           const vector<reco::TransientTrack>& primaries) const {
  /*
  cout << "[MultiVertexReconstructor] with " << primaries.size()
       << " primaries!" << endl;
       */

  map<reco::TransientTrack, bool> st;

  vector<reco::TransientTrack> total = trks;
  for (const auto& trk : trks) {
    st[trk] = true;
  }

  // cout << "[MultiVertexReconstructor] FIXME dont just add up tracks. superpose" << endl;
  for (const auto& primarie : primaries) {
    if (!(st[primarie])) {
      total.push_back(primarie);
    }
  }

  vector<TransientVertex> tmp = theOldReconstructor->vertices(total);

  if (verbose()) {
    cout << "[MultiVertexReconstructor] freezing seeder found " << tmp.size() << " vertices from " << total.size()
         << " tracks." << endl;
  }
  vector<vector<TrackAndWeight> > rc = recover(tmp, trks);
  vector<CachingVertex<5> > cvts = theFitter.vertices(rc, primaries);

  vector<TransientVertex> ret;
  for (const auto& cvt : cvts) {
    ret.push_back(cvt);
  };
  if (verbose()) {
    cout << "[MultiVertexReconstructor] input " << tmp.size() << " vertices, output " << ret.size() << " vertices."
         << endl;
  };
  return ret;
}

VertexReconstructor* MultiVertexReconstructor::reconstructor() const { return theOldReconstructor; }
