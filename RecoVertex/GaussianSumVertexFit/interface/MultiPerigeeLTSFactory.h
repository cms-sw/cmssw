#ifndef MultiPerigeeLTSFactory_H
#define MultiPerigeeLTSFactory_H

#include "RecoVertex/VertexPrimitives/interface/RefCountedLinearizedTrackState.h"
#include "RecoVertex/GaussianSumVertexFit/interface/PerigeeMultiLTS.h"
#include "TrackingTools/TransientTrack/interface/GsfTransientTrack.h"

// #include "CommonReco/PatternTools/interface/TransverseImpactPointExtrapolator.h"
//#include "CommonReco/PatternTools/interface/RecTrack.h"
//#include "Vertex/VertexPrimitives/interface/LinearizedTrackStateFactory.h"

/** 
 *  Concrete class to encapsulate the creation of RefCountedLinearizedTrackState, 
 *  which is a reference-counting pointer. 
 *  This class handles the possibility that the input track is described by 
 *  not one state, but a Gaussian Mixture of state (see BasicMultiTrajectoryState).
 *  Should always be used in order to create a new RefCountedLinearizedTrackState, 
 *  so that the reference-counting mechanism works well. 
 */ 

class MultiPerigeeLTSFactory {

public:
//   MultiPerigeeLTSFactory();
//   virtual ~MultiPerigeeLTSFactory();

  RefCountedLinearizedTrackState
    linearizedTrackState(const GlobalPoint & linP, const reco::TransientTrack & track) const;

  RefCountedLinearizedTrackState
    linearizedTrackState(const GlobalPoint & linP, const reco::TransientTrack & track,
    	const TrajectoryStateOnSurface& tsos) const;
};

#endif
