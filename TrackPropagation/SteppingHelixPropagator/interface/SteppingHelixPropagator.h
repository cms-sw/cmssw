#ifndef TrackPropagation_SteppingHelixPropagator_SteppingHelixPropagator_h
#define TrackPropagation_SteppingHelixPropagator_SteppingHelixPropagator_h 1

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
    NOT_IMPLEMENTED
  } ;

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
    PATHL_DT
  };

  enum Fancy {
    HEL_AS_F=0, //simple analytical helix, eloss at end of step
    HEL_ALL_F,  //analytical helix with linear eloss
    POL_1_F, //1st order approximation, straight line
    POL_2_F,//2nd order
    POL_M_F //highest available
  };

  SteppingHelixPropagator();
  SteppingHelixPropagator(const MagneticField* field, PropagationDirection dir = alongMomentum);

  SteppingHelixPropagator* clone() const {return new SteppingHelixPropagator(*this);}

  ~SteppingHelixPropagator() {}

  const MagneticField* magneticField() const { return field_;}

  virtual TrajectoryStateOnSurface propagate(const FreeTrajectoryState&, const Plane&) const;
  virtual TrajectoryStateOnSurface propagate(const FreeTrajectoryState&, const Cylinder&) const;
  virtual std::pair<TrajectoryStateOnSurface, double> propagateWithPath(const FreeTrajectoryState&, const Plane&) const;
  virtual std::pair<TrajectoryStateOnSurface, double> propagateWithPath(const FreeTrajectoryState&, const Cylinder&) const;
  


  void setDebug(bool debug){ debug_ = debug;}

  void setMaterialMode(bool noMaterial) { noMaterialMode_ = noMaterial;}

 protected:
  void setIState(const SteppingHelixPropagator::Vector& p3, const SteppingHelixPropagator::Point& r3, 
		 int charge, const HepSymMatrix& cov, PropagationDirection dir = alongMomentum) const;
  void getFState(SteppingHelixPropagator::Vector& p3, SteppingHelixPropagator::Point& r3,  HepSymMatrix& cov) const;

  Result propagateToR(double rDest, double epsilon = 1e-2) const;
  Result propagateToZ(double zDest, double epsilon = 1e-2) const;
  Result propagateByPathLength(double sDest, double epsilon = 1e-2) const;
  Result propagateToPlane(const double pars[6], double epsilon = 1e-2) const;
  Result propagate(SteppingHelixPropagator::DestType type, const double pars[6]) const;

  void loadState(int ind, const SteppingHelixPropagator::Vector& p3, const SteppingHelixPropagator::Point& r3, int charge,
		 const HepSymMatrix& cov, PropagationDirection dir) const;

  void incrementState(int ind, 
		    double dP, SteppingHelixPropagator::Vector tau,
		    double dX, double dY, double dZ, double dS,
		    const HepMatrix& dCov) const;

  void setReps(int ind) const;

  bool makeAtomStep(int iIn, double dS, PropagationDirection dir, SteppingHelixPropagator::Fancy fancy) const;

  double getDeDx(int iIn, double& dEdXPrime) const;

  int cIndex_(int ind) const;

  void refToPlane(int ind, const double pars[6], double& dist, bool& isIncoming) const;

  void initCovRotation(const SteppingHelixPropagator::Vector* repI[3], const SteppingHelixPropagator::Vector* repF[3],
		       HepMatrix& covRot) const;
		       
  void getLocBGrad(int ind, double delta) const;

 private:
  typedef std::pair<TrajectoryStateOnSurface, double> TsosPP;
  static const int MAX_STEPS = 10000;
  static const int MAX_POINTS = 50;
  mutable int nPoints_;
  mutable int q_[MAX_POINTS+1];
  mutable Vector p3_[MAX_POINTS+1];
  mutable Point r3_[MAX_POINTS+1];
  mutable HepSymMatrix cov_[MAX_POINTS+1];
  mutable HepSymMatrix covLoc_[MAX_POINTS+1];
  mutable double path_[MAX_POINTS+1];
  mutable Basis reps_[MAX_POINTS+1]; //not normalized though (but orthogonal)
  mutable double dir_[MAX_POINTS+1];
  mutable Vector bf_[MAX_POINTS+1];
  mutable Vector bfGradLoc_[MAX_POINTS+1];

  mutable HepMatrix covRot_;
  mutable HepMatrix dCTransform_;

  const MagneticField* field_;
  const HepDiagMatrix unit66_;
  bool debug_;
  bool noMaterialMode_;
};

#endif
