/** Updates a GsfTrack with a virtual hit representing a vertex constraint */

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h" 

#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateTransform.h"
#include "TrackingTools/GsfTracking/interface/GsfMultiStateUpdator.h"
#include "TrackingTools/GsfTracking/interface/GsfChi2MeasurementEstimator.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TRecHit2DPosConstraint.h"

class TrackerGeometry;
class MagneticField;
class GsfPropagatorAdapter;
class TransverseImpactPointExtrapolator;

class GsfConstraintAtVertex {
public:
  explicit GsfConstraintAtVertex (const edm::EventSetup&);
  ~GsfConstraintAtVertex();
  
  /// (multi)TSOS after including the beamspot
  TrajectoryStateOnSurface constrainAtBeamSpot (const reco::GsfTrack&,
						const reco::BeamSpot&) const;
  /// (multi)TSOS after include a vertex
  TrajectoryStateOnSurface constrainAtVertex (const reco::GsfTrack&,
					      const reco::Vertex&) const;
  /// (multi)TSOS after including a point with covariance matrix
  TrajectoryStateOnSurface constrainAtPoint (const reco::GsfTrack&,
					     const GlobalPoint& globalPosition,
					     const GlobalError& globalError) const;

private:

  MultiTrajectoryStateTransform multiStateTransformer_;
  GsfMultiStateUpdator gsfUpdator_;
  const TrackerGeometry* geometry_;
  const MagneticField* magField_;
  GsfPropagatorAdapter* gsfPropagator_;
  TransverseImpactPointExtrapolator* tipExtrapolator_;

};

