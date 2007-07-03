#ifndef TrajectoryStateClosestToBeamLine_H
#define TrajectoryStateClosestToBeamLine_H

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

/**
 * Trajectory state defined at the point of closest approach (PCA) of the
 * track to the beamline. It gives also access to the point on the beamline which is
 * the closest to the track.
 */

class TrajectoryStateClosestToBeamLine
{
public:

  typedef FreeTrajectoryState		FTS;

  TrajectoryStateClosestToBeamLine() {}

  TrajectoryStateClosestToBeamLine
    (const FTS& stateAtPCA, const GlobalPoint & pointOnBeamLine,
     const reco::BeamSpot & beamSpot);

  ~TrajectoryStateClosestToBeamLine(){}

  /**
   * State of the track at the PCA to the beamline
   */

  FTS trackStateAtPCA() const {return theFTS;}

  /**
   * Point on the beamline which is the closest to the track
   */
  GlobalPoint beamLinePCA() const {return thePointOnBeamLine;}

  /**
   * Transverse impact parameter of the track to the beamline.
   * It is the transverse distance of the two PCAs.
   */
  Measurement1D transverseImpactParameter() const;

  /**
   * The beamline
   */
  reco::BeamSpot beamSpot() {return theBeamSpot;}

private:

  FTS theFTS;
  GlobalPoint thePointOnBeamLine;
  reco::BeamSpot theBeamSpot;

};
#endif
