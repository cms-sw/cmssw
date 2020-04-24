#ifndef _Tracker_TwoTrackMinimumDistanceHelixLine_H_
#define _Tracker_TwoTrackMinimumDistanceHelixLine_H_

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include <utility>
/** \class TwoTrackMinimumDistanceHelixLine
 *  This is a helper class for TwoTrackMinimumDistance, for the
 *  case where one of the tracks is charged and the other not.
 *  No user should need direct access to this class.
 *  It implements a Newton method
 *  for finding the minimum distance between two tracks.
 */

class GlobalTrajectoryParameters;

class TwoTrackMinimumDistanceHelixLine {

public:

  TwoTrackMinimumDistanceHelixLine(): theH(nullptr), theL(nullptr), themaxiter(12),
	pointsUpdated(false){}
  ~TwoTrackMinimumDistanceHelixLine() {}


  /**
   * Calculates the PCA between a charged particle (helix) and a neutral 
   * particle (line). The order of the trajectories (helix-line or line-helix)
   * is irrelevent, and will be conserved.
   */

  bool calculate( const GlobalTrajectoryParameters &,
      const GlobalTrajectoryParameters &,
      const float qual=.0001 ); // retval=true? error occured.

  /**
   * Returns the PCA's on the two trajectories. The first point lies on the
   * first trajectory, the second point on the second trajectory.
   */

  std::pair <GlobalPoint, GlobalPoint> points() const;
  std::pair <double, double> pathLength() const;

  double firstAngle() const;
  double secondAngle() const;

private:
  GlobalTrajectoryParameters *theH, *theL, *firstGTP, *secondGTP;
  GlobalVector posDiff;
  GlobalVector theLp;
  double X, Y, Z, px, py, pz, px2, py2, pz2, baseFct, baseDer;
  double theh, thePhiH0, thesinPhiH0, thecosPhiH0, thetanlambdaH;
  double thePhiH;
  double Hn, Ln;
  double aa,bb,cc,dd,ee,ff;

  int themaxiter;
  bool updateCoeffs();
  void finalPoints() const;
  bool oneIteration(double & thePhiH, double & fct, double & derivative ) const;
  mutable GlobalPoint helixPoint, linePoint;
  mutable double tL, linePath, helixPath;
  mutable bool pointsUpdated;

};
#endif
