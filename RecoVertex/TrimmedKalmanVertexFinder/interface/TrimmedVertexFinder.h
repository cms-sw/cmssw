#ifndef _RecoVertex_TrimmedVertexFinder_H_
#define _RecoVertex_TrimmedVertexFinder_H_

#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"
#include "RecoVertex/VertexPrimitives/interface/VertexFitter.h"
#include "RecoVertex/VertexPrimitives/interface/VertexUpdator.h"
#include "RecoVertex/VertexPrimitives/interface/VertexTrackCompatibilityEstimator.h"
#include <vector>

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
  typedef ReferenceCountingPointer<VertexTrack<5> > RefCountedVertexTrack;
  typedef ReferenceCountingPointer<LinearizedTrackState<5> > RefCountedLinearizedTrackState;

  TrimmedVertexFinder(const VertexFitter<5>* vf,
                      const VertexUpdator<5>* vu,
                      const VertexTrackCompatibilityEstimator<5>* ve);

  /** Copy constructor, needed to handle copy of pointer data members correctly
   */
  TrimmedVertexFinder(const TrimmedVertexFinder& other);
  ~TrimmedVertexFinder();

  /** Make 0 or 1 vertex 
   *  On output, `remain` contains the incompatible tracks
   */
  std::vector<TransientVertex> vertices(std::vector<reco::TransientTrack>& remain) const;

  /** Same as above,
   * only with the extra information of the beamspot
   * constraint.
   */
  std::vector<TransientVertex> vertices(std::vector<reco::TransientTrack>& remain,
                                        const reco::BeamSpot& s,
                                        bool use_beamspot = true) const;

  /** Access to parameter
   */
  float trackCompatibilityCut() const { return theMinProb; }

  /** Set parameter
   */
  void setTrackCompatibilityCut(float cut) { theMinProb = cut; }

  /** clone method
   */
  TrimmedVertexFinder* clone() const { return new TrimmedVertexFinder(*this); }

private:
  // finds least compatible track
  // returns vtxTracks.end() if all tracks are compatible
  //
  std::vector<RefCountedVertexTrack>::iterator theWorst(const CachingVertex<5>& vtx,
                                                        std::vector<RefCountedVertexTrack>& vtxTracks,
                                                        float cut) const;

  VertexFitter<5>* theFitter;
  VertexUpdator<5>* theUpdator;
  VertexTrackCompatibilityEstimator<5>* theEstimator;
  float theMinProb;
};

#endif
