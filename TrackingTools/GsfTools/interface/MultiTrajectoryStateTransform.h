#ifndef MultiTrajectoryStateTransform_H
#define MultiTrajectoryStateTransform_H

/** Extracts innermost and  outermost states from a GsfTrack
 *  in form of a TrajectoryStateOnSurface */

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

class TrajectoryStateOnSurface;
class TrackingGeometry;
class Surface;
class MagneticField;
class TransverseImpactPointExtrapolator;

class MultiTrajectoryStateTransform {
private:
  typedef reco::GsfTrackExtra::LocalParameterVector ParameterVector;
  typedef reco::GsfTrackExtra::LocalCovarianceMatrix CovarianceMatrix;
  enum { dimension = reco::GsfTrackExtra::dimension };

public:
  /** Default constructor (deprecated) -> ES components have to be passed explicitely */
  MultiTrajectoryStateTransform() : geometry_(nullptr), field_(nullptr), extrapolator_(nullptr) {}
  /** Constructor from geometry and magnetic field */
  MultiTrajectoryStateTransform(const TrackingGeometry* geom, const MagneticField* field)
      : geometry_(geom), field_(field), extrapolator_(nullptr) {}
  /** Destructor */
  ~MultiTrajectoryStateTransform();

  /** TrajectoryStateOnSurface from the innermost state of a reco::GsfTrack */
  TrajectoryStateOnSurface innerStateOnSurface(const reco::GsfTrack& tk) const;
  /** TrajectoryStateOnSurface from the outermost state of a reco::GsfTrack */
  TrajectoryStateOnSurface outerStateOnSurface(const reco::GsfTrack& tk) const;

  /** Momentum vector from mode corresponding to the innermost state. 
   *  Returns true for success. */
  bool innerMomentumFromMode(const reco::GsfTrack& tk, GlobalVector& momentum) const;
  /** Momentum vector from mode corresponding to the outermost state */
  bool outerMomentumFromMode(const reco::GsfTrack& tk, GlobalVector& momentum) const;

  /** Extrapolation to a point using the TransverseImpactPointExtrapolator */
  TrajectoryStateOnSurface extrapolatedState(const TrajectoryStateOnSurface tsos, const GlobalPoint& point) const;

  /** TrajectoryStateOnSurface from the innermost state of a reco::GsfTrack 
   *  passing geometry and magnetic field (deprecated: use field from constructor) */
  static TrajectoryStateOnSurface innerStateOnSurface(const reco::GsfTrack& tk,
                                                      const TrackingGeometry& geom,
                                                      const MagneticField* field);
  /** TrajectoryStateOnSurface from the outermost state of a reco::GsfTrack 
   *  passing geometry and magnetic field (deprecated: use field from constructor) */
  static TrajectoryStateOnSurface outerStateOnSurface(const reco::GsfTrack& tk,
                                                      const TrackingGeometry& geom,
                                                      const MagneticField* field);

private:
  /** TSOS from a mixture in local parameters */
  static TrajectoryStateOnSurface stateOnSurface(const std::vector<double>& weights,
                                                 const std::vector<ParameterVector>& parameters,
                                                 const std::vector<CovarianceMatrix>& covariances,
                                                 const double& pzSign,
                                                 const Surface& surface,
                                                 const MagneticField* field);
  /** On-demand creation of a TransverseImpactPointExtrapolator */
  bool checkExtrapolator() const;
  /** Verification of the presence of geometry and field */
  bool checkGeometry() const;

private:
  const TrackingGeometry* geometry_;
  const MagneticField* field_;
  mutable TransverseImpactPointExtrapolator* extrapolator_;
};

#endif
