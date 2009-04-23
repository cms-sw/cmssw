#include "RecoVertex/TrimmedKalmanVertexFinder/interface/ConfigurableTrimmedVertexFinder.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

using namespace reco;

ConfigurableTrimmedVertexFinder::ConfigurableTrimmedVertexFinder(
  const VertexFitter<5> * vf, 
  const VertexUpdator<5> * vu, 
  const VertexTrackCompatibilityEstimator<5> * ve) 
  : theClusterFinder(vf, vu, ve), theVtxFitProbCut(0.01), 
    theTrackCompatibilityToPV(0.05), theTrackCompatibilityToSV(0.01), 
    theMaxNbOfVertices(0)
{
  // default pt cut is 1.5 GeV
  theFilter.setPtCut(1.5);
}


vector<TransientVertex> ConfigurableTrimmedVertexFinder::vertices(
  const vector<TransientTrack> & tracks) const
{
  vector<TransientTrack> remaining;

  return vertices(tracks, remaining);

}


vector<TransientVertex> ConfigurableTrimmedVertexFinder::vertices(
  const vector<TransientTrack> & tracks, vector<TransientTrack> & unused) 
  const
{
  resetEvent(tracks);
  analyseInputTracks(tracks);

  vector<TransientTrack> filtered;
  for (vector<TransientTrack>::const_iterator it = tracks.begin();
       it != tracks.end(); it++) {
    if (theFilter(*it)) { 
      filtered.push_back(*it);
    }
    else {
      unused.push_back(*it);
    }
  }

  vector<TransientVertex> all = vertexCandidates(filtered, unused);

  analyseVertexCandidates(all);

  vector<TransientVertex> sel = clean(all);

  analyseFoundVertices(sel);

  return sel;

}


vector<TransientVertex> ConfigurableTrimmedVertexFinder::vertexCandidates(
  const vector<TransientTrack> & tracks, vector<TransientTrack> & unused) const 
{

  vector<TransientVertex> cand;

  vector<TransientTrack> remain = tracks;

  while (true) {

    float tkCompCut = (cand.size() == 0 ? 
		       theTrackCompatibilityToPV 
		       : theTrackCompatibilityToSV);

    //    cout << "PVR:compat cut " << tkCompCut << endl;
    theClusterFinder.setTrackCompatibilityCut(tkCompCut);
    //    cout << "PVCF:compat cut after setting " 
    //	 << theClusterFinder.trackCompatibilityCut() << endl;

    vector<TransientVertex> newVertices = theClusterFinder.vertices(remain);
    if (newVertices.empty()) break;
    
    analyseClusterFinder(newVertices, remain);
    
    for (vector<TransientVertex>::const_iterator iv = newVertices.begin();
         iv != newVertices.end(); iv++) {
      if ( iv->originalTracks().size() > 1 ) {
        cand.push_back(*iv);
      } 
      else {
        // candidate has too few tracks - get them back into the vector
        for ( vector< TransientTrack >::const_iterator trk
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

  for (vector<TransientTrack>::const_iterator it = remain.begin();
       it != remain.end(); it++) {
    unused.push_back(*it);
  }

  return cand;
}


vector<TransientVertex> 
ConfigurableTrimmedVertexFinder::clean(const vector<TransientVertex> & candidates) const
{
  vector<TransientVertex> sel;
  for (vector<TransientVertex>::const_iterator i = candidates.begin(); 
       i != candidates.end(); i++) {

    if (ChiSquaredProbability((*i).totalChiSquared(), (*i).degreesOfFreedom())
	> theVtxFitProbCut) { sel.push_back(*i); }
  }

  return sel;
}
