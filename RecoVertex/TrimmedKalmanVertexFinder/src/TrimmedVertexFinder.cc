#include "RecoVertex/TrimmedKalmanVertexFinder/interface/TrimmedVertexFinder.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "RecoVertex/VertexTools/interface/PerigeeLinearizedTrackState.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace reco;

TrimmedVertexFinder::TrimmedVertexFinder(
  const VertexFitter<5> * vf, const VertexUpdator<5> * vu, 
  const VertexTrackCompatibilityEstimator<5> * ve)
  : theFitter(vf->clone()), theUpdator(vu->clone()), 
    theEstimator(ve->clone()), theMinProb(0.05)
{}

    
TrimmedVertexFinder::TrimmedVertexFinder(
const TrimmedVertexFinder & other) 
  : theFitter(other.theFitter->clone()), 
  theUpdator(other.theUpdator->clone()), 
  theEstimator(other.theEstimator->clone()), 
  theMinProb(other.theMinProb)
{}


TrimmedVertexFinder::~TrimmedVertexFinder() {
  delete theFitter;
  delete theUpdator;
  delete theEstimator;
}

std::vector<TransientVertex> 
TrimmedVertexFinder::vertices(std::vector<TransientTrack> & tks ) const
{
  // FIXME write this!!!
  return vertices ( tks, reco::BeamSpot(), false );
}

std::vector<TransientVertex> 
TrimmedVertexFinder::vertices(std::vector<TransientTrack> & tks,
    const reco::BeamSpot & spot, bool use_spot )  const
{
  std::vector<TransientVertex> all;
  if (tks.size() < 2) return all;

  // prepare vertex tracks and initial vertex
  CachingVertex<5> vtx;
  if ( use_spot ) 
  {
     vtx = theFitter->vertex(tks, spot );
  } else {
     vtx = theFitter->vertex(tks);
  }
  if (!vtx.isValid()) {
    edm::LogWarning ( "TrimmedVertexFinder" ) << "initial vertex invalid."
	    << " vertex finding stops here.";
    return all;
  }

  std::vector<RefCountedVertexTrack> selected = vtx.tracks();

  // reject incompatible tracks starting from the worst
  std::vector<RefCountedVertexTrack> remain;
  bool found = false;
  while (!found && selected.size() >= 2) {

    // find track with worst compatibility
    std::vector<RefCountedVertexTrack>::iterator iWorst = theWorst(vtx, 
							      selected, 
							      theMinProb);
    
    if (iWorst != selected.end()) {
      // reject track
      remain.push_back(*iWorst);
      selected.erase(iWorst);
      
      if (selected.size() == 1) {
	// not enough tracks to build new vertices
	remain.push_back(selected.front());
      }
      else {
	// removing track from vertex
	// need to redo the vertex fit instead of removing the track;
	// successive removals lead to numerical problems
	// this is however quick since intermediate quantities are cached
	// in the RefCountedVertexTracks
  if ( use_spot ) // && all.size()==0 )
  {
	  vtx = theFitter->vertex(selected,spot);
  } else {
	  vtx = theFitter->vertex(selected);
  }
	if (!vtx.isValid()) {
    edm::LogWarning ( "TrimmedVertexFinder" ) << "current vertex invalid"
	       << "vertex finding stops here.";
	  return all;
	}

	// ref-counted tracks may have changed during the fit
	// if the linearization point has moved too much
	selected = vtx.tracks();
      }
      
    } else {
      // no incompatible track remains, make vertex
      found = true;
      int n_tracks_in_vertex = selected.size();

      // now return all tracks with weight < 0.5 to 'remain'.
      for ( std::vector< RefCountedVertexTrack >::const_iterator t=selected.begin();
          t!=selected.end() ; ++t )
      {
        if ( (**t).weight() < 0.5 )
        {
          /*
          cout << "[TrimmedVertexFinder] recycling track with weight "
               << (**t).weight() << endl;*/
          remain.push_back ( *t );
          n_tracks_in_vertex--; // one 'good' track less in the vertex
        };
      };

      if ( n_tracks_in_vertex > 1 ) {
        all.push_back(vtx);
      }
      else {
        edm::LogError ( "TrimmedVertexFinder" ) 
          << "found vertex has less than 2 tracks";
      }
    }
  }

  // modify list of incompatible tracks
  tks.clear();
  for (std::vector<RefCountedVertexTrack>::const_iterator i = remain.begin(); 
       i != remain.end(); i++) {
    const PerigeeLinearizedTrackState* plts = 
      dynamic_cast<const PerigeeLinearizedTrackState*>
      ((**i).linearizedTrack().get());
    if (plts == 0) {
      throw cms::Exception("TrimmedVertexFinder: can't take track from non-perigee track state");
    }

    tks.push_back(plts->track());
  }

  // return 0 or 1 vertex
  return all;
}


std::vector<TrimmedVertexFinder::RefCountedVertexTrack>::iterator 
TrimmedVertexFinder::theWorst(const CachingVertex<5> & vtx, 
  std::vector<RefCountedVertexTrack> & vtxTracks, float cut) const
{

  //  cout << "Cut is now " << cut << endl;

  // find track with worst compatibility
  std::vector<RefCountedVertexTrack>::iterator iWorst = vtxTracks.end();
  float worseChi2 = 0.;
  for (std::vector<RefCountedVertexTrack>::iterator itr = vtxTracks.begin();
       itr != vtxTracks.end(); itr++) {

    CachingVertex<5> newV = theUpdator->remove(vtx, *itr);
    if (!newV.isValid()) return itr;
    std::pair<bool, double> result = theEstimator->estimate(newV, *itr);
    if (!result.first) return itr;
    float chi2 = result.second;

    // compute number of degrees of freedom
    if ( chi2 > 0. ) {
      // we want to keep negative chi squares, and avoid calling
      // ChiSquaredProbability with a negative chi2.
      int ndf = 2;
      if (vtx.tracks().size() == 2) ndf = 1;

      float prob = ChiSquaredProbability(chi2, ndf);
      if (prob < cut && chi2 >= worseChi2) {
        // small inconsistency: sorting on chi2, not on chi2 probability
        // because large chi2 are not rounded off while small probabilities 
        // are rounded off to 0....
        // should not matter since at a given iteration 
        // all track chi2 increments have the same ndf
        iWorst = itr;
        worseChi2 = chi2;
      }
    }
  }

  return iWorst;
}
