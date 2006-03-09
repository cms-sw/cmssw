#ifndef SequentialKinematicVertexFitter_H
#define SequentialKinematicVertexFitter_H

#include "RecoVertex/VertexTools/interface/SequentialVertexFitter.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicVertex.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"
#include "RecoVertex/VertexPrimitives/interface/VertexSmoother.h"
#include "RecoVertex/VertexPrimitives/interface/VertexUpdator.h"

/**
 * Analog of SequentialVertexFitter, using the
 * same algorithm and notation, but working
 * with KinematicFit objects:
 * For internal KinematicFit library needs
 * WARNING: no inherited methods should be
 * called from this class: it may lead to
 * usage of unapropriate type of
 * LinearizedTrackState. 
 *
 * Kirill Prokofiev, Mai 2003
 */

class SequentialKinematicVertexFitter
{
public: 
 
/**
 * Constructor with user selected updator and smoother
 */ 
 SequentialKinematicVertexFitter(const VertexUpdator& theUpdator, const VertexSmoother& theSmoother);

 ~SequentialKinematicVertexFitter();

/**
 * Reconstruct a KinematicVertex out of set
 * of given VertexTracks. Here VertexTracks are
 * created out of KinematicLinearizedTrackStates 
 */ 
 CachingVertex vertex(const vector<RefCountedVertexTrack> & tracks) const;
 
private:

/**
 * Reading simple configurable parameters
 */
  void readParameters();
 
/**
 * Construct new a container of VertexTrack with a new linearization point
 * and vertex seed, from an existing set of VertexTrack, from which only the
 * recTracks will be used.
 * \param tracks The original container of VertexTracks, from which the TransientTracks
 * will be extracted.
 * \param seed The seed to use for the VertexTracks. This position will
 * also be used as the new linearization point.
 * \return The container of VertexTracks which are to be used in the next fit.
 */
  vector<RefCountedVertexTrack> reLinearizeTracks(const vector<RefCountedVertexTrack> & tracks,
					const VertexState seed) const;
				
  CachingVertex fit(const vector<RefCountedVertexTrack> & tracks,
		const VertexState state, bool withPrior) const;		

//configurable data
  float theMaxShift;
  int theMaxStep;
  VertexUpdator * updator;
  VertexSmoother * smoother;				
  VertexTrackFactory vTrackFactory;
};
#endif
