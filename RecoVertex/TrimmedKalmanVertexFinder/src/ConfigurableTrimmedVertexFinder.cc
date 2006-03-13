#include "Utilities/Configuration/interface/Architecture.h"

#include "RecoVertex/TrimmedKalmanVertexFinder/interface/ConfigurableTrimmedVertexFinder.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"


ConfigurableTrimmedVertexFinder::ConfigurableTrimmedVertexFinder(
  const VertexFitter * vf, 
  const VertexUpdator * vu, 
  const VertexTrackCompatibilityEstimator * ve) 
  : theClusterFinder(vf, vu, ve), theVtxFitProbCut(0.01), 
    theTrackCompatibilityToPV(0.05), theTrackCompatibilityToSV(0.01), 
    theMaxNbOfVertices(0)
{
  // default pt cut is 1.5 GeV
  theFilter.setPtCut(1.5);
}


vector<RecVertex> ConfigurableTrimmedVertexFinder::vertices(
  const vector<RecTrack> & tracks) const
{
  vector<RecTrack> remaining;

  return vertices(tracks, remaining);

}


vector<RecVertex> ConfigurableTrimmedVertexFinder::vertices(
  const vector<RecTrack> & tracks, vector<RecTrack> & unused) 
  const
{
  resetEvent(tracks);
  analyseInputTracks(tracks);

  vector<RecTrack> filtered;
  for (vector<RecTrack>::const_iterator it = tracks.begin();
       it != tracks.end(); it++) {
    if (theFilter(*it)) { 
      filtered.push_back(*it);
    }
    else {
      unused.push_back(*it);
    }
  }

  vector<RecVertex> all = vertexCandidates(filtered, unused);

  analyseVertexCandidates(all);

  vector<RecVertex> sel = clean(all);

  analyseFoundVertices(sel);

  return sel;

}


vector<RecVertex> ConfigurableTrimmedVertexFinder::vertexCandidates(
  const vector<RecTrack> & tracks, vector<RecTrack> & unused) const 
{

  vector<RecVertex> cand;

  vector<RecTrack> remain = tracks;

  while (true) {

    float tkCompCut = (cand.size() == 0 ? 
		       theTrackCompatibilityToPV 
		       : theTrackCompatibilityToSV);

    //    cout << "PVR:compat cut " << tkCompCut << endl;
    theClusterFinder.setTrackCompatibilityCut(tkCompCut);
    //    cout << "PVCF:compat cut after setting " 
    //	 << theClusterFinder.trackCompatibilityCut() << endl;

    vector<RecVertex> newVertices = theClusterFinder.vertices(remain);
    if (newVertices.empty()) break;
    
    analyseClusterFinder(newVertices, remain);
    
    for (vector<RecVertex>::const_iterator iv = newVertices.begin();
         iv != newVertices.end(); iv++) {
      if ( iv->originalTracks().size() > 1 ) {
        cand.push_back(*iv);
      } 
      else {
        // candidate has too few tracks - get them back into the vector
        for ( vector< RecTrack >::const_iterator trk
		= iv->originalTracks().begin();
              trk != iv->originalTracks().end(); ++trk ) {
          unused.push_back ( *trk );
        }
      }
    }

    // when max number of vertices reached, stop
    if (theMaxNbOfVertices != 0) {
      if (cand.size() >= theMaxNbOfVertices) break;
    }
  }

  for (vector<RecTrack>::const_iterator it = remain.begin();
       it != remain.end(); it++) {
    unused.push_back(*it);
  }

  return cand;
}


vector<RecVertex> 
ConfigurableTrimmedVertexFinder::clean(const vector<RecVertex> & candidates) const
{
  vector<RecVertex> sel;
  for (vector<RecVertex>::const_iterator i = candidates.begin(); 
       i != candidates.end(); i++) {

    if (ChiSquaredProbability((*i).totalChiSquared(), (*i).degreesOfFreedom())
	> theVtxFitProbCut) { sel.push_back(*i); }
  }

  return sel;
}
