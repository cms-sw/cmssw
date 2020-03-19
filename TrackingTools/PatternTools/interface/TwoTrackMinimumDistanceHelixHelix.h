#ifndef _Tracker_TwoTrackMinimumDistanceHelixHelix_H_
#define _Tracker_TwoTrackMinimumDistanceHelixHelix_H_

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
// #include <string>
// #include <sstream>
#include <utility>

/** \class TwoTrackMinimumDistanceHelixHelix
 *  This is a helper class for TwoTrackMinimumDistance.
 *  No user should need direct access to this class.
 *  It actually implements a Newton-Kantorowitsch method
 *  for finding the minimum distance between two tracks.
 */

class GlobalTrajectoryParameters;

class TwoTrackMinimumDistanceHelixHelix {
public:
  TwoTrackMinimumDistanceHelixHelix();
  ~TwoTrackMinimumDistanceHelixHelix();

  bool calculate(const GlobalTrajectoryParameters &,
                 const GlobalTrajectoryParameters &,
                 const float qual = .001);  // retval=true? error occured.

  std::pair<GlobalPoint, GlobalPoint> points() const { return std::pair<GlobalPoint, GlobalPoint>(pointG, pointH); }

  std::pair<double, double> pathLength() const { return std::pair<double, double>(pathG, pathH); }

  double firstAngle() const { return thepG; }
  double secondAngle() const { return thepH; }

private:
  bool updateCoeffs(const GlobalPoint &, const GlobalPoint &);
  bool oneIteration(double &, double &);

  // bool parallelTracks () const;
  void finalPoints();

private:
  GlobalTrajectoryParameters const *theH, *theG;
  // the 'GH-track data' (constants)
  double thea, theb, thec1, thec2, thed1, thed2, thee1, thee2, theg, theh;
  // double thelambdaG, thelambdaH;
  double thetanlambdaG, thetanlambdaH;
  double thesinpG0, thecospG0;
  double thesinpH0, thecospH0;
  double thepG0, thepH0;

  // the variable stuff
  // = the point we are currently looking at.
  double thepG, thepH;
  double thesinpG, thesinpH;
  double thecospG, thecospH;
  GlobalPoint pointG, pointH;
  double pathG, pathH;
  bool pointsUpdated;

  double themaxjump, thesingjacI;
  int themaxiter;
};
#endif
