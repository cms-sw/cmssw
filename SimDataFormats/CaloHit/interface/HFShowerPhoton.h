#ifndef SimDataFormats_HFShowerPhoton_H
#define SimDataFormats_HFShowerPhoton_H
///////////////////////////////////////////////////////////////////////////////
// File: HFShowerPhoton.h
// Photons which will generate single photo electron as in HFShowerLibrary
///////////////////////////////////////////////////////////////////////////////

#include "DataFormats/Math/interface/Point3D.h"
#include <iostream>
#include <cmath>
#include <vector>

class HFShowerPhoton {
public:
  /// point in the space
  typedef math::XYZPointF Point;

  HFShowerPhoton(float x = 0, float y = 0, float z = 0, float lambda = 0, float t = 0);
  HFShowerPhoton(const Point& p, float time, float lambda);
  HFShowerPhoton(const HFShowerPhoton&) = default;
  HFShowerPhoton(HFShowerPhoton&&) = default;

  HFShowerPhoton& operator=(const HFShowerPhoton&) = default;
  HFShowerPhoton& operator=(HFShowerPhoton&&) = default;

  const Point& position() const { return position_; }
  float x() const { return position_.X(); }
  float y() const { return position_.Y(); }
  float z() const { return position_.Z(); }
  float lambda() const { return lambda_; }
  float t() const { return time_; }

private:
  Point position_;
  float lambda_;
  float time_;
};

typedef std::vector<HFShowerPhoton> HFShowerPhotonCollection;

std::ostream& operator<<(std::ostream&, const HFShowerPhoton&);

#endif
