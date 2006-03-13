#ifndef _ConfigurableTrimmedVertexFinder_H_
#define _ConfigurableTrimmedVertexFinder_H_

#include "RecoVertex/VertexPrimitives/interface/VertexReconstructor.h"
#include "RecoVertex/TrimmedKalmanVertexFinder/interface/TrimmedVertexFinder.h"
#include "RecoVertex/TrimmedKalmanVertexFinder/interface/TrimmedTrackFilter.h"
#include <vector>

/** Algorithm to find a series of distinct vertices among the given set 
 *  of tracks. The technique is: <BR>
 *    1) use `TrimmedTrackFilter` to select tracks 
 *    above a certain pT; <BR>
 *    2) use `TrimmedVertexFinder` to split the set of tracks into 
 *    a cluster of compatible tracks and a set of remaining tracks; <BR>
 *    3) repeat 2) with the remaining set, and repeat as long as 
 *    (a) a cluster of compatible tracks can be found or (b) 
 *    the maximum number of clusters asked for is reached; <BR>
 *    4) reject vertices with a low fit probability. <BR>
 *
 *  This algorithm has 5 parameters that can be set at runtime 
 *  via the corresponding set() methods: <BR>
 *   - "ptCut" (default: 1.5 GeV/c) 
 *  which defines the minimum pT of the tracks used to make vertices. 
 *  This value overrides the corresponding configurable parameter 
 *  of the TrimmedTrackFilter used internally. <BR>
 *   - "trackCompatibilityCut" (default: 0.05) 
 *  which defines the probability below which a track is considered 
 *  incompatible with the 1st vertex candidate formed. 
 *  This value overrides the corresponding configurable parameter 
 *  of the TrimmedVertexFinder used internally. <BR>
 *   - "trackCompatibilityToSV" (default: 0.01) 
 *  which defines the probability below which a track is considered 
 *  incompatible with the next vertex candidates formed. 
 *  This value overrides the corresponding configurable parameter 
 *  of the TrimmedVertexFinder used internally. <BR>
 *   - "vertexFitProbabilityCut" (default: 0.01) 
 *  which defines the probability below which a vertex is rejected. <BR>
 *   - "maxNbOfVertices" (default: 0) 
 *  which defines the maximum number of vertices searched for. 
 *  0 means all vertices which can be found. 
 */

class ConfigurableTrimmedVertexFinder : public VertexReconstructor {

public:

  ConfigurableTrimmedVertexFinder(const VertexFitter * vf, 
				  const VertexUpdator * vu, 
				  const VertexTrackCompatibilityEstimator * ve);

  virtual ~ConfigurableTrimmedVertexFinder() {}

  virtual vector<RecVertex> vertices(const vector<TransientTrack> & tracks)
    const;

  vector<RecVertex> vertices( const vector<TransientTrack> & tracks,
			      vector<TransientTrack>& unused) const;

  /** Access to parameters
   */
  float ptCut() const { return theFilter.ptCut(); }
  const TrimmedTrackFilter & trackFilter() const { 
    return theFilter; 
  }
  float trackCompatibilityCut() const { return theTrackCompatibilityToPV; }
  float trackCompatibilityToSV() const { return theTrackCompatibilityToSV; }
  float vertexFitProbabilityCut() const { return theVtxFitProbCut; }
  int maxNbOfVertices() const { return theMaxNbOfVertices; }

  /** Set parameters
   */
  void setPtCut(float cut) { theFilter.setPtCut(cut); }
  void setTrackCompatibilityCut(float cut) {
    theTrackCompatibilityToPV = cut;
  }
  void setTrackCompatibilityToSV(float cut) {
    theTrackCompatibilityToSV = cut;
  }
  void setVertexFitProbabilityCut(float cut) { theVtxFitProbCut = cut; }
  void setMaxNbOfVertices(int max) { theMaxNbOfVertices = max; }

  /** Clone method
   */
  virtual ConfigurableTrimmedVertexFinder * clone() const {
    return new ConfigurableTrimmedVertexFinder(*this);
  }

protected:

  virtual void resetEvent(const vector<TransientTrack> & tracks) const {}

  virtual void analyseInputTracks(const vector<TransientTrack> & tracks) 
    const {}

  virtual void analyseClusterFinder(const vector<RecVertex> & vts, 
				    const vector<TransientTrack> & remain) 
    const {}

  virtual void analyseVertexCandidates(const vector<RecVertex> & vts) 
    const {}

  virtual void analyseFoundVertices(const vector<RecVertex> & vts) 
    const {}


private:

  // separate tracks passing the filter
  //  void separateTracks(vector<TransientTrack>& filtered, 
  //		      vector<TransientTrack>& unused) const;

  // find vertex candidates
  vector<RecVertex> vertexCandidates(const vector<TransientTrack> & tracks, 
				     vector<TransientTrack>& unused) const;

  // remove bad candidates
  vector<RecVertex> clean(const vector<RecVertex> & candidates) const;
  
  // 
  mutable TrimmedVertexFinder theClusterFinder;
  float theVtxFitProbCut;
  float theTrackCompatibilityToPV;
  float theTrackCompatibilityToSV;
  int theMaxNbOfVertices;
  TrimmedTrackFilter theFilter;

};

#endif
