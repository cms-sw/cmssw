#ifndef SteppingHelixPropagator_SteppingHelixPropagator_h
#define SteppingHelixPropagator_SteppingHelixPropagator_h



/** \class SteppingHelixPropagator
 *  Propagator implementation using steps along a helix.
 *  Minimal geometry navigation.
 *  Material effects (multiple scattering and energy loss) are based on tuning
 *  to MC and (eventually) data. 
 *
 *  $Date: 2012/05/08 08:41:00 $
 *  $Revision: 1.31 $
 *  \author Vyacheslav Krutelyov (slava77)
 */

//
// Original Author:  Vyacheslav Krutelyov
//         Created:  Fri Mar  3 16:01:24 CST 2006
// $Id: SteppingHelixPropagator.h,v 1.31 2012/05/08 08:41:00 innocent Exp $
//
//



#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "CLHEP/Matrix/SymMatrix.h"
#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Vector/ThreeVector.h"


#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixStateInfo.h"

class MagneticField;
class VolumeBasedMagneticField;
class MagVolume;

class SteppingHelixPropagator GCC11_FINAL : public Propagator {
 public:
  typedef CLHEP::Hep3Vector Vector;
  typedef CLHEP::Hep3Vector  Point;

  typedef SteppingHelixStateInfo StateInfo;
  typedef SteppingHelixStateInfo::Result Result;

  struct Basis {
    Vector lX;
    Vector lY;
    Vector lZ;
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

  //! Constructors
  SteppingHelixPropagator();
  SteppingHelixPropagator(const MagneticField* field, PropagationDirection dir = alongMomentum);

  virtual SteppingHelixPropagator* clone() const {return new SteppingHelixPropagator(*this);}

  //! Destructor
  ~SteppingHelixPropagator() {}
  
  virtual const MagneticField* magneticField() const { return field_;}
  
  //! Propagate to Plane given a starting point
  virtual TrajectoryStateOnSurface 
    propagate(const FreeTrajectoryState& ftsStart, const Plane& pDest) const;
  //! Propagate to Cylinder given a starting point (a Cylinder is assumed to be positioned at 0,0,0)
  virtual TrajectoryStateOnSurface 
    propagate(const FreeTrajectoryState& ftsStart, const Cylinder& cDest) const;
  //! Propagate to PCA to point given a starting point 
  virtual FreeTrajectoryState 
    propagate(const FreeTrajectoryState& ftsStart, const GlobalPoint& pDest) const;
  //! Propagate to PCA to a line (given by 2 points) given a starting point 
  virtual FreeTrajectoryState 
    propagate(const FreeTrajectoryState& ftsStart, 
	      const GlobalPoint& pDest1, const GlobalPoint& pDest2) const;
  //! Propagate to PCA to a line determined by BeamSpot position and slope given a starting point 
  virtual FreeTrajectoryState 
    propagate(const FreeTrajectoryState& ftsStart, 
	      const reco::BeamSpot& beamSpot) const;

  //! Propagate to Plane given a starting point: return final 
  //! TrajectoryState and path length from start to this point
  virtual std::pair<TrajectoryStateOnSurface, double> 
    propagateWithPath(const FreeTrajectoryState& ftsStart, const Plane& pDest) const;
  //! Propagate to Cylinder given a starting point: return final TrajectoryState 
  //!and path length from start to this point
  virtual std::pair<TrajectoryStateOnSurface, double> 
    propagateWithPath(const FreeTrajectoryState& ftsStart, const Cylinder& cDest) const;
  //! Propagate to PCA to point given a starting point 
  virtual std::pair<FreeTrajectoryState, double> 
    propagateWithPath(const FreeTrajectoryState& ftsStart, const GlobalPoint& pDest) const;
  //! Propagate to PCA to a line (given by 2 points) given a starting point 
  virtual std::pair<FreeTrajectoryState, double> 
    propagateWithPath(const FreeTrajectoryState& ftsStart, 
		      const GlobalPoint& pDest1, const GlobalPoint& pDest2) const;
  //! Propagate to PCA to a line (given by beamSpot position and slope) given a starting point 
  virtual std::pair<FreeTrajectoryState, double> 
    propagateWithPath(const FreeTrajectoryState& ftsStart, 
		      const reco::BeamSpot& beamSpot) const;
    
    
  //! Propagate to Plane given a starting point
  const SteppingHelixStateInfo& 
    propagate(const SteppingHelixStateInfo& ftsStart, const Surface& sDest) const;
  const SteppingHelixStateInfo& 
    propagate(const SteppingHelixStateInfo& ftsStart, const Plane& pDest) const;
  //! Propagate to Cylinder given a starting point (a Cylinder is assumed to be positioned at 0,0,0)
  const SteppingHelixStateInfo& 
    propagate(const SteppingHelixStateInfo& ftsStart, const Cylinder& cDest) const;
  //! Propagate to PCA to point given a starting point 
  const SteppingHelixStateInfo& 
    propagate(const SteppingHelixStateInfo& ftsStart, const GlobalPoint& pDest) const;
  //! Propagate to PCA to a line (given by 2 points) given a starting point 
  const SteppingHelixStateInfo& 
    propagate(const SteppingHelixStateInfo& ftsStart, 
	      const GlobalPoint& pDest1, const GlobalPoint& pDest2) const;

  
  //! Switch debug printouts (to cout) .. very verbose
  void setDebug(bool debug){ debug_ = debug;}

  //! Switch for material effects mode: no material effects if true
  void setMaterialMode(bool noMaterial) { noMaterialMode_ = noMaterial;}

  //! Force no error propagation
  void setNoErrorPropagation(bool noErrorPropagation) { noErrorPropagation_ = noErrorPropagation;}

  //! Apply radLength correction (1+0.036*ln(radX0+1)) to covariance matrix
  //! +1 is added for IR-safety
  //! Should be done with care: it's easy to make the end-point result dependent
  //! on the intermediate stop points
  void applyRadX0Correction(bool applyRadX0Correction) { applyRadX0Correction_ = applyRadX0Correction;}

  //!Switch to using MagneticField Volumes .. as in VolumeBasedMagneticField
  void setUseMagVolumes(bool val){ useMagVolumes_ = val;}

  //!Switch to using Material Volumes .. internally defined for now
  void setUseMatVolumes(bool val){ useMatVolumes_ = val;}

  //! flag to return tangent plane for non-plane input
  void setReturnTangentPlane(bool val){ returnTangentPlane_ = val;}

  //! flag to send LogWarning on failures
  void setSendLogWarning(bool val){ sendLogWarning_ = val;}

  //! (temporary?) flag to identify if the volume is yoke based on MagVolume internals
  //! works if useMatVolumes_ is also true
  void setUseIsYokeFlag(bool val){ useIsYokeFlag_ = val;}

  //! will use bigger steps ~tuned for good-enough L2/Standalone reco
  //! use this in hope of a speed-up
  void setUseTuningForL2Speed(bool val){ useTuningForL2Speed_ = val;}

  //! set VolumBasedField pointer
  //! allows to have geometry description in uniformField scenario
  //! only important/relevant in the barrel region
  void setVBFPointer(const VolumeBasedMagneticField* val) { vbField_ = val;}

  //! force getting field value from MagneticField, not the geometric one
  void setUseInTeslaFromMagField(bool val) { useInTeslaFromMagField_ = val;}

  //! set shifts in Z for endcap pieces (includes EE, HE, ME, YE)
  void setEndcapShiftsInZPosNeg(double valPos, double valNeg){
    ecShiftPos_ = valPos; ecShiftNeg_ = valNeg;
  }

 protected:
  typedef SteppingHelixStateInfo::VolumeBounds MatBounds;
  //! (Internals) Init starting point
  void setIState(const SteppingHelixStateInfo& sStart) const;

  //! propagate: chose stop point by type argument
  //! propagate to fixed radius [ r = sqrt(x**2+y**2) ] with precision epsilon
  //! propagate to plane by [x0,y0,z0, n_x, n_y, n_z] parameters
  Result propagate(SteppingHelixPropagator::DestType type, 
		   const double pars[6], double epsilon = 1e-3) const;

  //! (Internals) compute transient values for initial point (resets step counter).
  //!  Called by setIState
  void loadState(SteppingHelixPropagator::StateInfo& svCurrent, 
		 const SteppingHelixPropagator::Vector& p3, 
		 const SteppingHelixPropagator::Point& r3, int charge,
		 PropagationDirection dir,
		 const AlgebraicSymMatrix55& covCurv) const;

  //! (Internals) compute transients for current point (increments step counter).
  //!  Called by makeAtomStep
  void getNextState(const SteppingHelixPropagator::StateInfo& svPrevious,
		    SteppingHelixPropagator::StateInfo& svNext,		      
		    double dP, 
		    const SteppingHelixPropagator::Vector& tau, const SteppingHelixPropagator::Vector& drVec, 
		    double dS, double dX0,
		    const AlgebraicMatrix55& dCovCurv) const;
  
  //! Set/compute basis vectors for local coordinates at current step (called by incrementState)
  void setRep(SteppingHelixPropagator::Basis& rep,
	      const SteppingHelixPropagator::Vector& tau) const;

  //! main stepping function: compute next state vector after a step of length dS
  bool makeAtomStep(SteppingHelixPropagator::StateInfo& svCurrent, 
		    SteppingHelixPropagator::StateInfo& svNext, 
		    double dS, PropagationDirection dir, 
		    SteppingHelixPropagator::Fancy fancy) const;

  //! estimate average (in fact smth. close to MPV and median) energy loss per unit path length
  double getDeDx(const SteppingHelixPropagator::StateInfo& sv, 
		 double& dEdXPrime, double& radX0, MatBounds& rzLims) const;

  //! (Internals) circular index for array of transients
  int cIndex_(int ind) const;

  //! (Internals) determine distance and direction from the current position to the plane
  Result refToDest(DestType dest, const SteppingHelixPropagator::StateInfo& sv,
		   const double pars[6], 
		   double& dist, double& tanDist, PropagationDirection& refDirection,
		   double fastSkipDist = 1e12) const;

  //! (Internals) determine distance and direction from the current position to the 
  //! boundary of current mag volume
  Result refToMagVolume(const SteppingHelixPropagator::StateInfo& sv,
			PropagationDirection dir,
			double& dist, double& tanDist,
			double fastSkipDist = 1e12, bool expectNewMagVolume = false, double maxStep = 1e12) const;

  Result refToMatVolume(const SteppingHelixPropagator::StateInfo& sv,
			PropagationDirection dir,
			double& dist, double& tanDist,
			double fastSkipDist = 1e12) const;

  //! check if it's a yoke/iron based on this MagVol internals  
  bool isYokeVolume(const MagVolume* vol) const;


 private:
  typedef std::pair<TrajectoryStateOnSurface, double> TsosPP;
  typedef std::pair<FreeTrajectoryState, double> FtsPP;
  static const int MAX_STEPS = 10000;
  static const int MAX_POINTS = 7;
  mutable int nPoints_;
  mutable StateInfo svBuf_[MAX_POINTS+1];

  StateInfo invalidState_;

  mutable AlgebraicMatrix55 covCurvRot_;
  mutable AlgebraicMatrix55 dCCurvTransform_;

  const MagneticField* field_;
  const VolumeBasedMagneticField* vbField_;
  const AlgebraicSymMatrix55 unit55_;
  bool debug_;
  bool noMaterialMode_;
  bool noErrorPropagation_;
  bool applyRadX0Correction_;
  bool useMagVolumes_;
  bool useIsYokeFlag_;
  bool useMatVolumes_;
  bool useInTeslaFromMagField_;
  bool returnTangentPlane_;
  bool sendLogWarning_;
  bool useTuningForL2Speed_;

  double defaultStep_;

  double ecShiftPos_;
  double ecShiftNeg_;
};

#endif
