#ifndef SteppingHelixPropagator_SteppingHelixStateInfo_h
#define SteppingHelixPropagator_SteppingHelixStateInfo_h



/** \class SteppingHelixStateInfo
 *  Holder of SteppingHelixState information
 *
 *  $Date: 2007/02/06 20:49:17 $
 *  $Revision: 1.6 $
 *  \author Vyacheslav Krutelyov (slava77)
 */

//
// Original Author:  Vyacheslav Krutelyov
//         Created:  Wed Jan  3 16:01:24 CST 2007
// $Id: SteppingHelixStateInfo.h,v 1.6 2007/02/06 20:49:17 slava77 Exp $
//
//

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"


#include "CLHEP/Matrix/SymMatrix.h"
#include "CLHEP/Vector/ThreeVector.h"



class MagneticField;
class MagVolume;
class Surface;
class SteppingHelixPropagator;
class FreeTrajectoryState;

class SteppingHelixStateInfo {
  friend class SteppingHelixPropagator;
 public:
  typedef Hep3Vector Vector;
  typedef Hep3Vector  Point;
  
  enum Result {
    OK=0,
    FAULT,
    APPROX,
    RANGEOUT,
    INACC,
    NOT_IMPLEMENTED,
    UNDEFINED
  };

  SteppingHelixStateInfo(): isComplete(0), isValid_(0), status_(UNDEFINED) {}
  SteppingHelixStateInfo(const FreeTrajectoryState& fts);

  TrajectoryStateOnSurface getStateOnSurface(const Surface& surf) const;

  ///convert internal structure into the fts
  void getFreeState(FreeTrajectoryState& fts) const;

  GlobalPoint position() const {return GlobalPoint(r3.x(), r3.y(), r3.z());}
  GlobalVector momentum() const {return GlobalVector(p3.x(), p3.y(), p3.z());}
  int charge() const {return q;}
  double path() const {return isValid_ ? path_ : 0;}

  bool isValid() const {return isValid_;}

  Result status() const {return status_;}

 protected:
  struct VolumeBounds {
    VolumeBounds(): zMin(0), zMax(1e4), rMin(0), rMax(1e4), th1(0), th2(0) {}
    VolumeBounds(double r0, double r1, double z0, double z1):
      zMin(z0), zMax(z1), rMin(r0), rMax(r1), th1(0), th2(0) {}
    VolumeBounds(double r0, double r1, double z0, double z1, double t1, double t2):
      zMin(z0), zMax(z1), rMin(r0), rMax(r1), th1(t1), th2(t2) {}
    double zMin;
    double zMax;
    double rMin;
    double rMax;
    double th1;
    double th2;
  };
  

  int q;
  Vector p3;
  Point r3;
  HepSymMatrix cov;
  HepSymMatrix matDCov;
  double path_;
  double radPath;
  double dir;
  Vector bf;
  Vector bfGradLoc;
  const MagVolume* magVol;
  const MagneticField* field;  

  VolumeBounds rzLims;
  double dEdx;
  double dEdXPrime;
  double radX0;

  bool isComplete;
  bool isValid_;
  Result status_;
};
#endif
