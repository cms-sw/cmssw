#ifndef SteppingHelixPropagator_SteppingHelixStateInfo_h
#define SteppingHelixPropagator_SteppingHelixStateInfo_h



/** \class SteppingHelixStateInfo
 *  Holder of SteppingHelixState information
 *
 *  $Date: 2006/12/28 03:28:05 $
 *  $Revision: 1.9 $
 *  \author Vyacheslav Krutelyov (slava77)
 */

//
// Original Author:  Vyacheslav Krutelyov
//         Created:  Wed Jan  3 16:01:24 CST 2007
// $Id: SteppingHelixStateInfo.h,v 1.9 2006/12/28 03:28:05 slava77 Exp $
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
  
  struct Basis {
    Vector lX;
    Vector lY;
    Vector lZ;
  };
  

  SteppingHelixStateInfo(): isComplete(0) {}
  SteppingHelixStateInfo(const FreeTrajectoryState& fts);

  TrajectoryStateOnSurface getStateOnSurface(const Surface& surf) const;

  bool isValid() const {return isValidInfo;}

 protected:
  void loadFreeState(FreeTrajectoryState& fts) const;

  int q;
  Vector p3;
  Point r3;
  HepSymMatrix cov;
  HepSymMatrix matDCov;
  double path;
  double radPath;
  Basis rep;
  double dir;
  Vector bf;
  Vector bfGradLoc;
  const MagVolume* magVol;
  const MagneticField* field;  

  bool isComplete;
  bool isValidInfo;
};
#endif
