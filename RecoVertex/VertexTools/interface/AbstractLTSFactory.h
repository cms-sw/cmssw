#ifndef VertexTools_AbstractLTSFactory_H
#define VertexTools_AbstractLTSFactory_H

#include "RecoVertex/VertexPrimitives/interface/RefCountedLinearizedTrackState.h"
#include "RecoVertex/VertexTools/interface/PerigeeLinearizedTrackState.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"


/**
 *  Abstract class that defines an LinearzedTrackStateFactory
 */

class AbstractLTSFactory {

public:

  virtual RefCountedLinearizedTrackState
    linearizedTrackState(const GlobalPoint & linP, const reco::TransientTrack & track) const = 0;

  virtual RefCountedLinearizedTrackState
    linearizedTrackState(const GlobalPoint & linP, const reco::TransientTrack & track,
    	const TrajectoryStateOnSurface& tsos) const = 0;

  virtual ~AbstractLTSFactory() {};

  virtual const AbstractLTSFactory * clone() const = 0;

};

#endif
