#ifndef _RecoVertex_TrimmedVertexFinder_H_
#define _RecoVertex_TrimmedVertexFinder_H_

#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"
#include <vector>

class VertexFitter;
class VertexUpdator;
class VertexTrackCompatibilityEstimator;

/** Algorithm to find 0 or 1 cluster of tracks that are compatible 
 *  with a single vertex, among `remain`, the initial set of tracks. 
 *  A track is declared incompatible with the vertex if the chi-squared 
 *  probability between the track and the vertex is smaller than `min`. 
 *  The algorithm applied is:  <BR>
 *    1) fit a single vertex with all tracks;  <BR>
 *    2) remove incompatible tracks 1 by 1 starting from 
 *  the least compatible one.  <BR>
 *  On output, `remain` contains the incompatible tracks. 
 *  This algorithm has 1 parameter that can be set at runtime 
 *  via the corresponding set() method: 
 *   - "trackCompatibilityCut" (default: 0.05)
 *  which defines the probability below which a track is considered 
 *  incompatible with a vertex. 
 */

class TrimmedVertexFinder {

public:

  TrimmedVertexFinder(const VertexFitter * vf, 
		      const VertexUpdator * vu, 
		      const VertexTrackCompatibilityEstimator * ve);

  /** Copy constructor, needed to handle copy of pointer data members correctly
   */
  TrimmedVertexFinder(const TrimmedVertexFinder & other);
  ~TrimmedVertexFinder();

  /** Make 0 or 1 vertex 
   *  On output, `remain` contains the incompatible tracks
   */
  vector<RecVertex> vertices(vector<TransientTrack> & remain) const;

  /** Access to parameter
   */
  float trackCompatibilityCut() const { return theMinProb; }

  /** Set parameter
   */
  void setTrackCompatibilityCut(float cut) { theMinProb = cut; }

  /** clone method
   */
  TrimmedVertexFinder * clone() const {
    return new TrimmedVertexFinder(*this);
  }


private:

  // finds least compatible track
  // returns vtxTracks.end() if all tracks are compatible
  //
  vector<RefCountedVertexTrack>::iterator theWorst(
    const CachingVertex & vtx, 
    vector<RefCountedVertexTrack> & vtxTracks, 
    float cut) const;

  float theMinProb;
  VertexFitter * theFitter;
  VertexUpdator * theUpdator;
  VertexTrackCompatibilityEstimator * theEstimator;

};

#endif
