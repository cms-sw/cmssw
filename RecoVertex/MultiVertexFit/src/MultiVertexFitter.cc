#include "RecoVertex/MultiVertexFit/interface/MultiVertexFitter.h"
// #include "Vertex/VertexPrimitives/interface/TransientVertex.h"
#include <map>
#include <algorithm>
#include <iomanip>
// #include "Vertex/VertexRecoAnalysis/interface/RecTrackNamer.h"
// #include "Vertex/MultiVertexFit/interface/TransientVertexNamer.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexTools/interface/LinearizedTrackStateFactory.h"
#include "RecoVertex/VertexTools/interface/VertexTrackFactory.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackCompatibilityEstimator.h"

// #define MVFHarvestingDebug
#ifdef MVFHarvestingDebug
#include "Vertex/VertexSimpleVis/interface/PrimitivesHarvester.h"
#endif

using namespace std;
using namespace reco;

namespace {
  typedef MultiVertexFitter::TrackAndWeight TrackAndWeight;
  typedef MultiVertexFitter::TrackAndSeedToWeightMap TrackAndSeedToWeightMap;
  typedef MultiVertexFitter::SeedToWeightMap SeedToWeightMap;
  typedef CachingVertex<5>::RefCountedVertexTrack RefCountedVertexTrack;

  int verbose() {
    static const int ret = 0; /*SimpleConfigurable<int>
      (0, "MultiVertexFitter:Debug").value(); */
    return ret;
  }

  double minWeightFraction() {
    // minimum weight that a track has to have
    // in order to be taken into account for the
    // vertex fit.
    // Given as a fraction of the total weight.
    static const float ret = 1e-6; /* SimpleConfigurable<float>
      (1e-6, "MultiVertexFitter:MinimumWeightFraction").value(); */
    return ret;
  }

  bool discardLightWeights() {
    static const bool ret = true; /* SimpleConfigurable<bool>
      (true, "MultiVertexFitter:DiscardLightWeights").value();*/
    return ret;
  }

  /*
  struct CompareRaveTracks
  {
    bool operator() ( const TransientTrack & r1,
                      const TransientTrack & r2 ) const
    {
      return r1 < r2;
    };
  }*/

  CachingVertex<5> createSeedFromLinPt(const GlobalPoint &gp) {
    return CachingVertex<5>(gp, GlobalError(), vector<RefCountedVertexTrack>(), 0.0);
  }

  double validWeight(double weight) {
    if (weight > 1.0) {
      cout << "[MultiVertexFitter] weight=" << weight << "??" << endl;
      return 1.0;
    };

    if (weight < 0.0) {
      cout << "[MultiVertexFitter] weight=" << weight << "??" << endl;
      return 0.0;
    };
    return weight;
  }
}  // namespace

void MultiVertexFitter::clear() {
  theAssComp->resetAnnealing();
  theTracks.clear();
  thePrimaries.clear();
  theVertexStates.clear();
  theWeights.clear();
  theCache.clear();
}

// creates the seed, additionally it pushes all tracks
// onto theTracks

void MultiVertexFitter::createSeed(const vector<TransientTrack> &tracks) {
  if (tracks.size() > 1) {
    CachingVertex<5> vtx = createSeedFromLinPt(theSeeder->getLinearizationPoint(tracks));
    int snr = seedNr();
    theVertexStates.push_back(pair<int, CachingVertex<5> >(snr, vtx));
    for (const auto &track : tracks) {
      theWeights[track][snr] = 1.;
      theTracks.push_back(track);
    };
  };
}

void MultiVertexFitter::createPrimaries(const std::vector<reco::TransientTrack> &tracks) {
  // cout << "[MultiVertexFitter] creating primaries: ";
  for (const auto &track : tracks) {
    thePrimaries.insert(track);
    // cout << i->id() << "  ";
  }
  // cout << endl;
}

int MultiVertexFitter::seedNr() { return theVertexStateNr++; }

void MultiVertexFitter::resetSeedNr() { theVertexStateNr = 0; }

void MultiVertexFitter::createSeed(const vector<TrackAndWeight> &tracks) {
  // create initial seed for every bundle
  vector<RefCountedVertexTrack> newTracks;

  for (const auto &track : tracks) {
    double weight = validWeight(track.second);
    const GlobalPoint &pos = track.first.impactPointState().globalPosition();
    GlobalError err;  // FIXME
    VertexState realseed(pos, err);

    RefCountedLinearizedTrackState lTrData = theCache.linTrack(pos, track.first);

    VertexTrackFactory<5> vTrackFactory;
    RefCountedVertexTrack vTrData = vTrackFactory.vertexTrack(lTrData, realseed, weight);
    newTracks.push_back(vTrData);
  };

  if (newTracks.size() > 1) {
    CachingVertex<5> vtx = KalmanVertexFitter().vertex(newTracks);
    int snr = seedNr();
    theVertexStates.push_back(pair<int, CachingVertex<5> >(snr, vtx));

    // We initialise the weights with the original
    // user supplied weights.
    for (const auto &track : tracks) {
      if (thePrimaries.count(track.first)) {
        /*
        cout << "[MultiVertexFitter] " << track->first.id() << " is a primary."
             << " setting weight for state " << theVertexStates[0].first
             << " to " << track->second
             << endl;
             */
        theWeights[track.first][theVertexStates[0].first] = track.second;
        continue;
      };
      float weight = track.second;
      if (weight > 1.0) {
        cout << "[MultiVertexFitter] error weight " << weight << " > 1.0 given." << endl;
        cout << "[MultiVertexFitter] will revert to 1.0" << endl;
        weight = 1.0;
      };
      if (weight < 0.0) {
        cout << "[MultiVertexFitter] error weight " << weight << " < 0.0 given." << endl;
        cout << "[MultiVertexFitter] will revert to 0.0" << endl;
        weight = 0.0;
      };
      theWeights[track.first][snr] = weight;
      theTracks.push_back(track.first);
    };
  };

  // this thing will actually have to discard tracks
  // that have been submitted - attached to a different vertex - already.
  // sort ( theTracks.begin(), theTracks.end(), CompareRaveTracks() );
  sort(theTracks.begin(), theTracks.end());
  for (vector<TransientTrack>::iterator i = theTracks.begin(); i < theTracks.end(); ++i) {
    if (i != theTracks.begin()) {
      if ((*i) == (*(i - 1))) {
        theTracks.erase(i);
      };
    };
  };
}

vector<CachingVertex<5> > MultiVertexFitter::vertices(const vector<TransientVertex> &vtces,
                                                      const vector<TransientTrack> &primaries) {
  // FIXME if vtces.size < 1 return sth that includes the primaries
  if (vtces.empty()) {
    return vector<CachingVertex<5> >();
  };
  vector<vector<TrackAndWeight> > bundles;
  for (const auto &vtce : vtces) {
    vector<TransientTrack> trks = vtce.originalTracks();
    vector<TrackAndWeight> tnws;
    for (const auto &trk : trks) {
      float w = vtce.trackWeight(trk);
      if (w > 1e-5) {
        TrackAndWeight tmp(trk, w);
        tnws.push_back(tmp);
      };
    };
    bundles.push_back(tnws);
  };
  return vertices(bundles, primaries);
}

vector<CachingVertex<5> > MultiVertexFitter::vertices(const vector<CachingVertex<5> > &initials,
                                                      const vector<TransientTrack> &primaries) {
  clear();
  createPrimaries(primaries);
  // FIXME if initials size < 1 return sth that includes the primaries
  if (initials.empty())
    return initials;
  for (const auto &initial : initials) {
    int snr = seedNr();
    theVertexStates.push_back(pair<int, CachingVertex<5> >(snr, initial));
    TransientVertex rvtx = initial;
    const vector<TransientTrack> &trks = rvtx.originalTracks();
    for (const auto &trk : trks) {
      if (!(thePrimaries.count(trk))) {
        // cout << "[MultiVertexFitter] free track " << trk->id() << endl;
        theTracks.push_back(trk);
      } else {
        // cout << "[MultiVertexFitter " << trk->id() << " is not free." << endl;
      }
      cout << "[MultiVertexFitter] error! track weight currently set to one"
           << " FIXME!!!" << endl;
      theWeights[trk][snr] = 1.0;
    };
  };
#ifdef MVFHarvestingDebug
  for (vector<CachingVertex<5> >::const_iterator i = theVertexStates.begin(); i != theVertexStates.end(); ++i)
    PrimitivesHarvester::file()->save(*i);
#endif
  return fit();
}

vector<CachingVertex<5> > MultiVertexFitter::vertices(const vector<vector<TransientTrack> > &tracks,
                                                      const vector<TransientTrack> &primaries) {
  clear();
  createPrimaries(primaries);

  for (const auto &track : tracks) {
    createSeed(track);
  };
  if (verbose()) {
    printSeeds();
  };
#ifdef MVFHarvestingDebug
  for (vector<CachingVertex<5> >::const_iterator i = theVertexStates.begin(); i != theVertexStates.end(); ++i)
    PrimitivesHarvester::file()->save(*i);
#endif
  return fit();
}

vector<CachingVertex<5> > MultiVertexFitter::vertices(const vector<vector<TrackAndWeight> > &tracks,
                                                      const vector<TransientTrack> &primaries) {
  clear();
  createPrimaries(primaries);

  for (const auto &track : tracks) {
    createSeed(track);
  };
  if (verbose()) {
    printSeeds();
  };

  return fit();
}

MultiVertexFitter::MultiVertexFitter(const AnnealingSchedule &ann,
                                     const LinearizationPointFinder &seeder,
                                     float revive_below)
    : theVertexStateNr(0), theReviveBelow(revive_below), theAssComp(ann.clone()), theSeeder(seeder.clone()) {}

MultiVertexFitter::MultiVertexFitter(const MultiVertexFitter &o)
    : theVertexStateNr(o.theVertexStateNr),
      theReviveBelow(o.theReviveBelow),
      theAssComp(o.theAssComp->clone()),
      theSeeder(o.theSeeder->clone()) {}

MultiVertexFitter::~MultiVertexFitter() {
  delete theAssComp;
  delete theSeeder;
}

void MultiVertexFitter::updateWeights() {
  theWeights.clear();
  if (verbose() & 4) {
    cout << "[MultiVertexFitter] Start weight update." << endl;
  };

  KalmanVertexTrackCompatibilityEstimator<5> theComp;

  /** 
   *  add the primary only tracks to primary vertex only.
   */
  for (const auto &thePrimarie : thePrimaries) {
    int seednr = theVertexStates[0].first;
    CachingVertex<5> seed = theVertexStates[0].second;
    pair<bool, double> result = theComp.estimate(seed, theCache.linTrack(seed.position(), thePrimarie));
    double weight = 0.;
    if (result.first)
      weight = theAssComp->phi(result.second);
    theWeights[thePrimarie][seednr] = weight;  // FIXME maybe "hard" 1.0 or "soft" weight?
  }

  /**
   *  now add "free tracks" to all vertices
   */
  for (const auto &theTrack : theTracks) {
    double tot_weight = theAssComp->phi(theAssComp->cutoff() * theAssComp->cutoff());

    for (const auto &theVertexState : theVertexStates) {
      pair<bool, double> result =
          theComp.estimate(theVertexState.second, theCache.linTrack(theVertexState.second.position(), theTrack));
      double weight = 0.;
      if (result.first)
        weight = theAssComp->phi(result.second);
      tot_weight += weight;
      theWeights[theTrack][theVertexState.first] = weight;
      /* cout << "[MultiVertexFitter] w[" << TransientTrackNamer().name(*trk)
           << "," << seed->position() << "] = " << weight << endl;*/
    };

    // normalize to sum of all weights of one track equals 1.
    // (if we include the "cutoff", as well)
    if (tot_weight > 0.0) {
      for (const auto &theVertexState : theVertexStates) {
        double normedweight = theWeights[theTrack][theVertexState.first] / tot_weight;
        if (normedweight > 1.0) {
          cout << "[MultiVertexFitter] he? w["  // << TransientTrackNamer().name(*trk)
               << "," << theVertexState.second.position() << "] = " << normedweight << " totw=" << tot_weight << endl;
          normedweight = 1.0;
        };
        if (normedweight < 0.0) {
          cout << "[MultiVertexFitter] he? weight=" << normedweight << " totw=" << tot_weight << endl;
          normedweight = 0.0;
        };
        theWeights[theTrack][theVertexState.first] = normedweight;
      };
    } else {
      // total weight equals zero? restart, with uniform distribution!
      cout << "[MultiVertexFitter] track found with no assignment - ";
      cout << "will assign uniformly." << endl;
      float w = .5 / (float)theVertexStates.size();
      for (const auto &theVertexState : theVertexStates) {
        theWeights[theTrack][theVertexState.first] = w;
      };
    };
  };
  if (verbose() & 2)
    printWeights();
}

bool MultiVertexFitter::updateSeeds() {
  double max_disp = 0.;
  // need to fit with the right weights.
  // also trigger an updateWeights.
  // if the seeds dont move much we return true

  vector<pair<int, CachingVertex<5> > > newSeeds;

  for (const auto &theVertexState : theVertexStates) {
    // for each seed get the tracks with the right weights.
    // TransientVertex rv = seed->second;
    // const GlobalPoint & seedpos = seed->second.position();
    int snr = theVertexState.first;
    VertexState realseed(theVertexState.second.position(), theVertexState.second.error());

    double totweight = 0.;
    for (const auto &theTrack : theTracks) {
      totweight += theWeights[theTrack][snr];
    };

    int nr_good_trks = 0;  // how many tracks above weight limit
    // we count those tracks, because that way
    // we can discard lightweights if there are enough tracks
    // and not discard the lightweights if that would give us
    // fewer than two tracks ( we would loose a seed, then ).
    if (discardLightWeights()) {
      for (const auto &theTrack : theTracks) {
        if (theWeights[theTrack][snr] > totweight * minWeightFraction()) {
          nr_good_trks++;
        };
      };
    };

    vector<RefCountedVertexTrack> newTracks;
    for (const auto &theTrack : theTracks) {
      double weight = validWeight(theWeights[theTrack][snr]);
      // Now we add a track, if
      // a. we consider all tracks or
      // b. we discard the lightweights but the track's weight is high enough or
      // c. we discard the lightweights but there arent enough good tracks,
      //    so we add all lightweights again (too expensive to figure out
      //    which lightweights are the most important)
      if (!discardLightWeights() || weight > minWeightFraction() * totweight || nr_good_trks < 2) {
        // if the linearization point didnt move too much,
        // we take the old LinTrackState.
        // Otherwise we relinearize.

        RefCountedLinearizedTrackState lTrData = theCache.linTrack(theVertexState.second.position(), theTrack);

        VertexTrackFactory<5> vTrackFactory;
        RefCountedVertexTrack vTrData = vTrackFactory.vertexTrack(lTrData, realseed, weight);
        newTracks.push_back(vTrData);
      };
    };

    for (const auto &thePrimarie : thePrimaries) {
      double weight = validWeight(theWeights[thePrimarie][snr]);

      RefCountedLinearizedTrackState lTrData = theCache.linTrack(theVertexState.second.position(), thePrimarie);

      VertexTrackFactory<5> vTrackFactory;
      RefCountedVertexTrack vTrData = vTrackFactory.vertexTrack(lTrData, realseed, weight);
      newTracks.push_back(vTrData);
    };

    try {
      if (newTracks.size() < 2) {
        throw VertexException("less than two tracks in vector");
      };

      if (verbose()) {
        cout << "[MultiVertexFitter] now fitting with Kalman: ";
        for (const auto &newTrack : newTracks) {
          cout << (*newTrack).weight() << " ";
        };
        cout << endl;
      };

      if (newTracks.size() > 1) {
        KalmanVertexFitter fitter;
        // warning! first track determines lin pt!
        CachingVertex<5> newVertex = fitter.vertex(newTracks);
        int snr = seedNr();
        double disp = (newVertex.position() - theVertexState.second.position()).mag();
        if (disp > max_disp)
          max_disp = disp;
        newSeeds.push_back(pair<int, CachingVertex<5> >(snr, newVertex));
      };
    } catch (exception &e) {
      cout << "[MultiVertexFitter] exception: " << e.what() << endl;
    }
  };

  // now discard all old seeds and weights, compute new ones.
  theVertexStates.clear();
  theWeights.clear();
  theVertexStates = newSeeds;
#ifdef MVFHarvestingDebug
  for (vector<CachingVertex<5> >::const_iterator i = theVertexStates.begin(); i != theVertexStates.end(); ++i)
    PrimitivesHarvester::file()->save(*i);
#endif
  updateWeights();

  static const double disp_limit = 1e-4; /* SimpleConfigurable<double>
    (0.0001, "MultiVertexFitter:DisplacementLimit" ).value(); */

  if (verbose() & 2) {
    printSeeds();
    cout << "[MultiVertexFitter] max displacement in this iteration: " << max_disp << endl;
  };
  if (max_disp < disp_limit)
    return false;
  return true;
}

// iterating over the fits
vector<CachingVertex<5> > MultiVertexFitter::fit() {
  if (verbose() & 2)
    printWeights();
  int ctr = 1;
  static const int ctr_max = 50; /* SimpleConfigurable<int>(100,
      "MultiVertexFitter:MaxIterations").value(); */
  while (updateSeeds() || !(theAssComp->isAnnealed())) {
    if (++ctr >= ctr_max)
      break;
    theAssComp->anneal();
    // lostVertexClaimer(); // was a silly(?) idea to "revive" vertex candidates.
    resetSeedNr();
  };

  if (verbose()) {
    cout << "[MultiVertexFitter] number of iterations: " << ctr << endl;
    cout << "[MultiVertexFitter] remaining seeds: " << theVertexStates.size() << endl;
    printWeights();
  };

  vector<CachingVertex<5> > ret;
  for (const auto &theVertexState : theVertexStates) {
    ret.push_back(theVertexState.second);
  };

  return ret;
}

void MultiVertexFitter::printWeights(const reco::TransientTrack &t) const {
  // cout << "Trk " << t.id();
  for (const auto &theVertexState : theVertexStates) {
    double val = 0;
    auto a = theWeights.find(t);
    if (a != theWeights.end()) {
      auto b = a->second.find(theVertexState.first);
      if (b != a->second.end())
        val = b->second;
    }
    cout << "  -- Vertex[" << theVertexState.first << "] with " << setw(12) << setprecision(3) << val;
  };
  cout << endl;
}

void MultiVertexFitter::printWeights() const {
  cout << endl << "Weight table: " << endl << "=================" << endl;
  for (const auto &thePrimarie : thePrimaries) {
    printWeights(thePrimarie);
  };
  for (const auto &theTrack : theTracks) {
    printWeights(theTrack);
  };
}

void MultiVertexFitter::printSeeds() const {
  cout << endl << "Seed table: " << endl << "=====================" << endl;
  /*
  for ( vector < pair < int, CachingVertex<5> > >::const_iterator seed=theVertexStates.begin();
        seed!=theVertexStates.end(); ++seed )
  {
    cout << "  Vertex[" << TransientVertexNamer().name(seed->second) << "] at "
         << seed->second.position() << endl;
  };*/
}

void MultiVertexFitter::lostVertexClaimer() {
  if (!(theReviveBelow < 0.))
    return;
  // this experimental method is used to get almost lost vertices
  // back into the play by upweighting vertices with very low total weights

  bool has_revived = false;
  // find out about total weight
  for (const auto &theVertexState : theVertexStates) {
    double totweight = 0.;
    for (const auto &theTrack : theTracks) {
      totweight += theWeights[theTrack][theVertexState.first];
    };

    /*
    cout << "[MultiVertexFitter] vertex seed " << TransientVertexNamer().name(*i)
         << " total weight=" << totweight << endl;*/

    if (totweight < theReviveBelow && totweight > 0.0) {
      cout << "[MultiVertexFitter] now trying to revive vertex"
           << " revive_below=" << theReviveBelow << endl;
      has_revived = true;
      for (const auto &theTrack : theTracks) {
        theWeights[theTrack][theVertexState.first] /= totweight;
      };
    };
  };
  if (has_revived && verbose())
    printWeights();
}
