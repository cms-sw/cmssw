#ifndef MultiTrajectoryStateTransform_H
#define MultiTrajectoryStateTransform_H

/** Extracts innermost and  outermost states from a GsfTrack
 *  in form of a TrajectoryStateOnSurface */

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"

class TrajectoryStateOnSurface;
class TrackingGeometry;
class Surface;
class MagneticField;

class MultiTrajectoryStateTransform {
private:
  typedef reco::GsfTrackExtra::LocalParameterVector ParameterVector;
  typedef reco::GsfTrackExtra::LocalCovarianceMatrix CovarianceMatrix;
  enum { dimension = reco::GsfTrackExtra::dimension };

public:

  /** Construct a TrajectoryStateOnSurface from the reco::GsfTrack 
   *  innermost or outermost state, requires access to tracking geometry */
  TrajectoryStateOnSurface innerStateOnSurface( const reco::GsfTrack& tk, 
						const TrackingGeometry& geom,
						const MagneticField* field) const;
  TrajectoryStateOnSurface outerStateOnSurface( const reco::GsfTrack& tk, 
						const TrackingGeometry& geom,
						const MagneticField* field) const;

private:
  TrajectoryStateOnSurface stateOnSurface (const std::vector<double>& weights,
					   const std::vector<ParameterVector>& parameters,
					   const std::vector<CovarianceMatrix>& covariances,
					   const double& pzSign,
					   const Surface& surface,
					   const MagneticField* field) const;
};

#endif
