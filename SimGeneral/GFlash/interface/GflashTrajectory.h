#ifndef GflashTrajectory_H
#define GflashTrajectory_H 1

#include "SimGeneral/GFlash/interface/GflashTrajectoryPoint.h"

class GflashTrajectory {
public:
  GflashTrajectory();
  ~GflashTrajectory();

  void initializeTrajectory(const HepGeom::Vector3D<double> &, const HepGeom::Point3D<double> &, double q, double Field);

  void setCotTheta(double cotTheta);
  void setCurvature(double curvature);
  void setZ0(double z0);
  void setD0(double d0);
  void setPhi0(double phi0);

  double getCotTheta() const { return _cotTheta; }
  double getCurvature() const { return _curvature; }
  double getZ0() const { return _z0; };
  double getD0() const { return _d0; };
  double getPhi0() const { return _phi0; };

  // Get sines and cosines of Phi0 and Theta
  double getSinPhi0() const;
  double getCosPhi0() const;
  double getSinTheta() const;
  double getCosTheta() const;

  // Get Position as a function of (three-dimensional) path length
  HepGeom::Point3D<double> getPosition(double s = 0.0) const;

  // Get Direction as a function of (three-dimensional) path length
  HepGeom::Vector3D<double> getDirection(double s = 0.0) const;

  void getGflashTrajectoryPoint(GflashTrajectoryPoint &point, double s) const;
  double getPathLengthAtRhoEquals(double rho) const;
  double getPathLengthAtZ(double z) const;

  double getZAtR(double r) const;
  double getL2DAtR(double r) const;

  // needed whenever _sinPhi0, _cosPh0, _sinTheta, or _cosTheta is used.
  void _refreshCache() const;

  // neede whenever _ss or _cc are used.
  void _cacheSinesAndCosines(double s) const;

private:
  // This is the GflashTrajectory:
  double _cotTheta;
  double _curvature;
  double _z0;
  double _d0;
  double _phi0;

  // This is the cache
  mutable bool _isStale;
  mutable double _sinPhi0;
  mutable double _cosPhi0;
  mutable double _sinTheta;
  mutable double _cosTheta;
  mutable double _s;

  mutable double _aa;
  mutable double _ss;
  mutable double _cc;
};

#endif
