#ifndef SteppingHelixPropagator_SteppingHelixPropagator_h
#define SteppingHelixPropagator_SteppingHelixPropagator_h



/** \class SteppingHelixPropagator
 *  Propagator implementation using steps along a helix.
 *  Minimal geometry navigation.
 *  Material effects (multiple scattering and energy loss) are based on tuning
 *  to MC and (eventually) data. 
 *
 *  $Date: 2006/10/05 23:48:55 $
 *  $Revision: 1.7 $
 *  \author Vyacheslav Krutelyov (slava77)
 */

//
// Original Author:  Vyacheslav Krutelyov
//         Created:  Fri Mar  3 16:01:24 CST 2006
// $Id: SteppingHelixPropagator.h,v 1.7 2006/10/05 23:48:55 slava77 Exp $
//
//



#include "Geometry/Vector/interface/GlobalVector.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include "CLHEP/Matrix/SymMatrix.h"
#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/DiagMatrix.h"
#include "CLHEP/Vector/ThreeVector.h"



class MagneticField;
class MagVolume;

class SteppingHelixPropagator : public Propagator {
 public:
  typedef Hep3Vector Vector;
  typedef Hep3Vector  Point;

  struct Basis {
    Vector lX;
    Vector lY;
    Vector lZ;
  };

  enum Result {
    OK=0,
    FAULT,
    APPROX,
    RANGEOUT,
    INACC,
    NOT_IMPLEMENTED,
    UNDEFINED
  };

  enum Pars {
    RADIUS_P=0,
    Z_P = 0,
    PATHL_P = 0
  };

  enum DestType {
    RADIUS_DT=0,
    Z_DT,
    PLANE_DT,
    CONE_DT,
    CYLINDER_DT,
    PATHL_DT,
    POINT_PCA_DT,
    LINE_PCA_DT,
    UNDEFINED_DT
  };

  enum Fancy {
    HEL_AS_F=0, //simple analytical helix, eloss at end of step
    HEL_ALL_F,  //analytical helix with linear eloss
    POL_1_F, //1st order approximation, straight line
    POL_2_F,//2nd order
    POL_M_F //highest available
  };

  struct StateInfo {
    int q;
    Vector p3;
    Point r3;
    HepSymMatrix covLoc;
    double path;
    double radPath;
    Basis rep;
    double dir;
    Vector bf;
    Vector bfGradLoc;
    const MagVolume* magVol;
  };

  /// Constructors
  SteppingHelixPropagator();
  SteppingHelixPropagator(const MagneticField* field, PropagationDirection dir = alongMomentum);

  virtual SteppingHelixPropagator* clone() const {return new SteppingHelixPropagator(*this);}

  /// Destructor
  ~SteppingHelixPropagator() {}
  
  virtual const MagneticField* magneticField() const { return field_;}
  
  /// Propagate to Plane given a starting point
  virtual TrajectoryStateOnSurface 
    propagate(const FreeTrajectoryState& ftsStart, const Plane& pDest) const;
  /// Propagate to Cylinder given a starting point (a Cylinder is assumed to be positioned at 0,0,0)
  virtual TrajectoryStateOnSurface 
    propagate(const FreeTrajectoryState& ftsStart, const Cylinder& cDest) const;
  /// Propagate to PCA to point given a starting point 
  virtual FreeTrajectoryState 
    propagate(const FreeTrajectoryState& ftsStart, const GlobalPoint& pDest) const;
  /// Propagate to PCA to a line (given by 2 points) given a starting point 
  virtual FreeTrajectoryState 
    propagate(const FreeTrajectoryState& ftsStart, 
	      const GlobalPoint& pDest1, const GlobalPoint& pDest2) const;

  /// Propagate to Plane given a starting point: return final 
  /// TrajectoryState and path length from start to this point
  virtual std::pair<TrajectoryStateOnSurface, double> 
    propagateWithPath(const FreeTrajectoryState& ftsStart, const Plane& pDest) const;
  /// Propagate to Cylinder given a starting point: return final TrajectoryState 
  ///and path length from start to this point
  virtual std::pair<TrajectoryStateOnSurface, double> 
    propagateWithPath(const FreeTrajectoryState& ftsStart, const Cylinder& cDest) const;
  /// Propagate to PCA to point given a starting point 
  virtual std::pair<FreeTrajectoryState, double> 
    propagateWithPath(const FreeTrajectoryState& ftsStart, const GlobalPoint& pDest) const;
  /// Propagate to PCA to a line (given by 2 points) given a starting point 
  virtual std::pair<FreeTrajectoryState, double> 
    propagateWithPath(const FreeTrajectoryState& ftsStart, 
	      const GlobalPoint& pDest1, const GlobalPoint& pDest2) const;
  

  /// Switch debug printouts (to cout) .. very verbose
  void setDebug(bool debug){ debug_ = debug;}

  /// Switch for material effects mode: no material effects if true
  void setMaterialMode(bool noMaterial) { noMaterialMode_ = noMaterial;}

  /// Force no error propagation
  void setNoErrorPropagation(bool noErrorPropagation) { noErrorPropagation_ = noErrorPropagation;}

  /// Apply radLength correction (1+0.036*ln(radX0+1)) to covariance matrix
  /// +1 is added for IR-safety
  /// Should be done with care: it's easy to make the end-point result dependent
  /// on the intermediate stop points
  void applyRadX0Correction(bool applyRadX0Correction) { applyRadX0Correction_ = applyRadX0Correction;}

  ///Switch to using MagneticField Volumes .. as in VolumeBasedMagneticField
    void setUseMagVolumes(bool val){ useMagVolumes_ = val;}

 protected:
  /// (Internals) Init starting point
  void setIState(const FreeTrajectoryState& ftsStart) const;
  void setIState(const SteppingHelixPropagator::Vector& p3, 
		 const SteppingHelixPropagator::Point& r3, 
		 int charge, const HepSymMatrix& cov, 
		 PropagationDirection dir = alongMomentum) const;
  /// (Internals) get final state
  void getFState(FreeTrajectoryState& ftsDest) const;
  void getFState(SteppingHelixPropagator::Vector& p3, SteppingHelixPropagator::Point& r3,  
		 HepSymMatrix& cov) const;

  /// propagate: chose stop point by type argument
  /// propagate to fixed radius [ r = sqrt(x**2+y**2) ] with precision epsilon
  /// propagate to plane by [x0,y0,z0, n_x, n_y, n_z] parameters
  Result propagate(SteppingHelixPropagator::DestType type, 
		   const double pars[6], double epsilon = 1e-3) const;

  /// (Internals) compute transient values for initial point (resets step counter).
  ///  Called by setIState
  void loadState(SteppingHelixPropagator::StateInfo& svCurrent, 
		 const SteppingHelixPropagator::Vector& p3, 
		 const SteppingHelixPropagator::Point& r3, int charge,
		 const HepSymMatrix& cov, PropagationDirection dir) const;

  /// (Internals) compute transients for current point (increments step counter).
  ///  Called by makeAtomStep
    void getNextState(const SteppingHelixPropagator::StateInfo& svPrevious,
		      SteppingHelixPropagator::StateInfo& svNext,		      
		      double dP, 
		      SteppingHelixPropagator::Vector tau, double dX, double dY, double dZ, 
		      double dS, double dX0,
		      const HepMatrix& dCov) const;
  
  /// Set/compute basis vectors for local coordinates at current step (called by incrementState)
  void setRep(SteppingHelixPropagator::StateInfo& sv) const;

  /// main stepping function: compute next state vector after a step of length dS
  bool makeAtomStep(SteppingHelixPropagator::StateInfo& svCurrent, 
		    SteppingHelixPropagator::StateInfo& svNext, 
		    double dS, PropagationDirection dir, 
		    SteppingHelixPropagator::Fancy fancy) const;

  /// estimate average (in fact smth. close to MPV and median) energy loss per unit path length
  double getDeDx(const SteppingHelixPropagator::StateInfo& sv, 
		 double& dEdXPrime, double& radX0) const;

  /// (Internals) circular index for array of transients
  int cIndex_(int ind) const;

  /// (Internals) determine distance and direction from the current position to the plane
  Result refToDest(DestType dest, const SteppingHelixPropagator::StateInfo& sv,
		   const double pars[6], 
		   double& dist, double& tanDist, PropagationDirection& refDirection) const;

  /// (Internals) determine distance and direction from the current position to the 
  /// boundary of current mag volume
  Result refToMagVolume(const SteppingHelixPropagator::StateInfo& sv,
			PropagationDirection dir,
			double& dist, double& tanDist) const;

  /// Compute covariance matrix rotation given change in basis vectors
  void initCovRotation(const SteppingHelixPropagator::Vector* repI[3], 
		       const SteppingHelixPropagator::Vector* repF[3],
		       HepMatrix& covRot) const;
		       
  /// |B|-field gradient in local coordinates
  void getLocBGrad(SteppingHelixPropagator::StateInfo& sv,
		   double delta) const;

 private:
  typedef std::pair<TrajectoryStateOnSurface, double> TsosPP;
  typedef std::pair<FreeTrajectoryState, double> FtsPP;
  static const int MAX_STEPS = 10000;
  static const int MAX_POINTS = 50;
  mutable int nPoints_;
  mutable StateInfo svBuf_[MAX_POINTS+1];

  mutable HepMatrix covRot_;
  mutable HepMatrix dCTransform_;

  const MagneticField* field_;
  const HepDiagMatrix unit66_;
  bool debug_;
  bool noMaterialMode_;
  bool noErrorPropagation_;
  bool applyRadX0Correction_;
  bool useMagVolumes_;
};

#endif
