// Most methods of this class are excerpted from CDF Helix and Trajectory class

#include "SimGeneral/GFlash/interface/GflashTrajectory.h"

GflashTrajectory::GflashTrajectory()
    : _cotTheta(0.0),
      _curvature(0.0),
      _z0(0.0),
      _d0(0.0),
      _phi0(0.0),
      _isStale(true),
      _sinPhi0(2),
      _cosPhi0(2),
      _sinTheta(2),
      _cosTheta(2),
      _s(-999.999),
      _aa(2),
      _ss(2),
      _cc(2) {
  // detault constructor
}

// GflashTrajectory::GflashTrajectory(const HepGeom::Vector3D<double>  &
// MomentumGev, const HepGeom::Point3D<double>   & PositionCm, 	     double q,
// double
// BFieldTesla)
void GflashTrajectory::initializeTrajectory(const HepGeom::Vector3D<double> &MomentumGev,
                                            const HepGeom::Point3D<double> &PositionCm,
                                            double q,
                                            double BFieldTesla) {
  double CotTheta = 0.0;
  double W = 0;
  double Z0 = 0;
  double D0 = 0;
  double Phi0 = 0;

  if (BFieldTesla != 0.0 && q != 0.0) {
    double CurvatureConstant = 0.0029979;
    double Helicity = -1.0 * fabs(BFieldTesla) * fabs(q) / (BFieldTesla * q);
    double Radius = fabs(MomentumGev.perp() / (CurvatureConstant * BFieldTesla * q));

    if (Radius == 0.0)
      W = HUGE_VAL;
    else
      W = Helicity / Radius;
    double phi1 = MomentumGev.phi();
    double x = PositionCm.x(), y = PositionCm.y(), z = PositionCm.z();
    double sinPhi1 = sin(phi1), cosPhi1 = cos(phi1);
    double gamma = atan((x * cosPhi1 + y * sinPhi1) / (x * sinPhi1 - y * cosPhi1 - 1 / W));
    Phi0 = phi1 + gamma;
    if (Phi0 > M_PI)
      Phi0 = Phi0 - 2.0 * M_PI;
    if (Phi0 < -M_PI)
      Phi0 = Phi0 + 2.0 * M_PI;
    D0 = ((1 / W + y * cosPhi1 - x * sinPhi1) / cos(gamma) - 1 / W);
    CotTheta = MomentumGev.z() / MomentumGev.perp();
    Z0 = z + gamma * CotTheta / W;
  } else {
    CLHEP::Hep3Vector direction = MomentumGev.unit();
    CLHEP::Hep3Vector projectedDirection = CLHEP::Hep3Vector(direction.x(), direction.y(), 0.0).unit();
    double s = projectedDirection.dot(PositionCm);
    double sprime = s / sin(direction.theta());
    Z0 = (PositionCm - sprime * direction).z();
    Phi0 = MomentumGev.phi();
    CotTheta = MomentumGev.z() / MomentumGev.perp();
    W = 0.0;
    D0 = (PositionCm.y() * cos(Phi0) - PositionCm.x() * sin(Phi0));
  }

  _cotTheta = CotTheta;
  _curvature = W / 2;
  _z0 = Z0;
  _d0 = D0;
  _phi0 = Phi0;

  _isStale = true;
  _s = -999.999;
  _aa = -999.999;
  _ss = -999.999;
  _cc = -999.999;
  _sinPhi0 = 1.0;
  _cosPhi0 = 1.0;
  _sinTheta = 1.0;
  _cosTheta = 1.0;
}

GflashTrajectory::~GflashTrajectory() {}

void GflashTrajectory::setCotTheta(double cotTheta) {
  _cotTheta = cotTheta;
  _isStale = true;
}

void GflashTrajectory::setCurvature(double curvature) {
  _curvature = curvature;
  _isStale = true;
}

void GflashTrajectory::setZ0(double z0) {
  _z0 = z0;
  _isStale = true;
}

void GflashTrajectory::setD0(double d0) {
  _d0 = d0;
  _isStale = true;
}

void GflashTrajectory::setPhi0(double phi0) {
  _phi0 = phi0;
  _isStale = true;
}

double GflashTrajectory::getSinPhi0() const {
  _refreshCache();
  return _sinPhi0;
}
double GflashTrajectory::getCosPhi0() const {
  _refreshCache();
  return _cosPhi0;
}
double GflashTrajectory::getSinTheta() const {
  _refreshCache();
  return _sinTheta;
}
double GflashTrajectory::getCosTheta() const {
  _refreshCache();
  return _cosTheta;
}

HepGeom::Point3D<double> GflashTrajectory::getPosition(double s) const {
  _cacheSinesAndCosines(s);
  if (s == 0.0 || _curvature == 0.0) {
    return HepGeom::Point3D<double>(
        -_d0 * _sinPhi0 + s * _cosPhi0 * _sinTheta, _d0 * _cosPhi0 + s * _sinPhi0 * _sinTheta, _z0 + s * _cosTheta);
  } else {
    return HepGeom::Point3D<double>(
        (_cosPhi0 * _ss - _sinPhi0 * (2.0 * _curvature * _d0 + 1.0 - _cc)) / (2.0 * _curvature),
        (_sinPhi0 * _ss + _cosPhi0 * (2.0 * _curvature * _d0 + 1.0 - _cc)) / (2.0 * _curvature),
        _s * _cosTheta + _z0);
  }
}

HepGeom::Vector3D<double> GflashTrajectory::getDirection(double s) const {
  _cacheSinesAndCosines(s);
  if (s == 0.0) {
    return HepGeom::Vector3D<double>(_cosPhi0 * _sinTheta, _sinPhi0 * _sinTheta, _cosTheta);
  }
  double xtan = _sinTheta * (_cosPhi0 * _cc - _sinPhi0 * _ss);
  double ytan = _sinTheta * (_cosPhi0 * _ss + _sinPhi0 * _cc);
  double ztan = _cosTheta;
  return HepGeom::Vector3D<double>(xtan, ytan, ztan);
}

void GflashTrajectory::getGflashTrajectoryPoint(GflashTrajectoryPoint &point, double s) const {
  _cacheSinesAndCosines(s);

  double cP0sT = _cosPhi0 * _sinTheta, sP0sT = _sinPhi0 * _sinTheta;
  if (s && _curvature) {
    point.getPosition().set((_cosPhi0 * _ss - _sinPhi0 * (2.0 * _curvature * _d0 + 1.0 - _cc)) / (2.0 * _curvature),
                            (_sinPhi0 * _ss + _cosPhi0 * (2.0 * _curvature * _d0 + 1.0 - _cc)) / (2.0 * _curvature),
                            s * _cosTheta + _z0);

    point.getMomentum().set(cP0sT * _cc - sP0sT * _ss, cP0sT * _ss + sP0sT * _cc, _cosTheta);
    point.setPathLength(s);
  } else {
    point.getPosition().set(-_d0 * _sinPhi0 + s * cP0sT, _d0 * _cosPhi0 + s * sP0sT, _z0 + s * _cosTheta);

    point.getMomentum().set(cP0sT, sP0sT, _cosTheta);

    point.setPathLength(s);
  }
}

double GflashTrajectory::getPathLengthAtRhoEquals(double rho) const {
  return (getSinTheta() ? (getL2DAtR(rho) / getSinTheta()) : 0.0);
}

double GflashTrajectory::getPathLengthAtZ(double z) const {
  return (getCosTheta() ? (z - getZ0()) / getCosTheta() : 0.0);
}

double GflashTrajectory::getZAtR(double rho) const { return _z0 + getCotTheta() * getL2DAtR(rho); }

double GflashTrajectory::getL2DAtR(double rho) const {
  double L2D;

  double c = getCurvature();
  double d = getD0();

  if (c != 0.0) {
    double rad = (rho * rho - d * d) / (1.0 + 2.0 * c * d);
    double rprime;
    if (rad < 0.0) {
      rprime = 0.0;
    } else {
      rprime = sqrt(rad);
    }
    if (c * rprime > 1.0 || c * rprime < -1.0) {
      L2D = c * rprime > 0. ? M_PI / c : -M_PI / c;
    } else
      L2D = asin(c * rprime) / c;
  } else {
    double rad = rho * rho - d * d;
    double rprime;
    if (rad < 0.0)
      rprime = 0.0;
    else
      rprime = sqrt(rad);

    L2D = rprime;
  }
  return L2D;
}

// Update _sinTheta,_cosTheta,_sinPhi0, and _cosPhi0
void GflashTrajectory::_refreshCache() const {
  if (_isStale) {
    _isStale = false;
    double theta;
    if (_cotTheta == 0.0) {
      theta = M_PI / 2.0;
    } else {
      theta = atan(1.0 / _cotTheta);
      if (theta < 0.0)
        theta += M_PI;
    }
    if (theta == 0.0) {
      _sinTheta = 0.0;
      _cosTheta = 1.0;
    } else {
      _cosTheta = cos(theta);
      _sinTheta = sqrt(1 - _cosTheta * _cosTheta);
    }
    if (_phi0 == 0.0) {
      _sinPhi0 = 0.0;
      _cosPhi0 = 1.0;
    } else {
      _cosPhi0 = cos(_phi0);
      _sinPhi0 = sin(_phi0);
      //      _sinPhi0 = sqrt(1.0-_cosPhi0*_cosPhi0);
      //      if (_phi0>M_PI) _sinPhi0 = -_sinPhi0;
    }
  }
}
// Update _s, _aa, _ss, and _cc if the arclength has changed.
void GflashTrajectory::_cacheSinesAndCosines(double s) const {
  _refreshCache();
  if (_s != s) {
    _s = s;
    _aa = 2.0 * _s * _curvature * _sinTheta;
    if (_aa == 0.0) {
      _ss = 0.0;
      _cc = 1.0;
    } else {
      _ss = sin(_aa);
      _cc = cos(_aa);
    }
  }
}
