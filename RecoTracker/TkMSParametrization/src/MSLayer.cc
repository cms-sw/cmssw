#include "RecoTracker/TkMSParametrization/interface/MSLayer.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "MSLayersKeeper.h"
#include "FWCore/Utilities/interface/Likely.h"

#include <iostream>

using namespace GeomDetEnumerators;
using namespace std;
template <class T>
T sqr(T t) {
  return t * t;
}

//----------------------------------------------------------------------
ostream& operator<<(ostream& s, const MSLayer& l) {
  s << " face: " << l.face() << " pos:" << l.position() << ", "
    << " range:" << l.range() << ", " << l.theX0Data;
  return s;
}
//----------------------------------------------------------------------
ostream& operator<<(ostream& s, const MSLayer::DataX0& d) {
  if (d.hasX0)
    s << "x0=" << d.x0 << " sumX0D=" << d.sumX0D;
  else if (d.allLayers)
    s << "x0 by MSLayersKeeper";
  else
    s << "empty DataX0";
  return s;
}
//----------------------------------------------------------------------
//MP
MSLayer::MSLayer(const DetLayer* layer, const DataX0& dataX0)
    : theFace(layer->location()), theSeqNum(layer->seqNum()), theX0Data(dataX0) {
  const BarrelDetLayer* bl;
  const ForwardDetLayer* fl;
  theHalfThickness = layer->surface().bounds().thickness() / 2;

  switch (theFace) {
    case barrel:
      bl = static_cast<const BarrelDetLayer*>(layer);
      thePosition = bl->specificSurface().radius();
      theRange = Range(-bl->surface().bounds().length() / 2, bl->surface().bounds().length() / 2);
      break;
    case endcap:
      fl = static_cast<const ForwardDetLayer*>(layer);
      thePosition = fl->position().z();
      theRange = Range(fl->specificSurface().innerRadius(), fl->specificSurface().outerRadius());
      break;
    default:
      // should throw or simimal
      cout << " ** MSLayer ** unknown part - will not work!" << endl;
      break;
  }
}
//----------------------------------------------------------------------
MSLayer::MSLayer(Location part, float position, Range range, float halfThickness, const DataX0& dataX0)
    : theFace(part),
      thePosition(position),
      theRange(range),
      theHalfThickness(halfThickness),
      theSeqNum(-1),
      theX0Data(dataX0) {}

//----------------------------------------------------------------------
bool MSLayer::operator==(const MSLayer& o) const {
  return theFace == o.theFace && std::abs(thePosition - o.thePosition) < 1.e-3f;
}
//----------------------------------------------------------------------
bool MSLayer::operator<(const MSLayer& o) const {
  if (theFace == barrel && o.theFace == barrel)
    return thePosition < o.thePosition;
  else if (theFace == barrel && o.theFace == endcap)
    return thePosition < o.range().max();
  else if (theFace == endcap && o.theFace == endcap)
    return std::abs(thePosition) < std::abs(o.thePosition);
  else
    return range().max() < o.thePosition;
}

//----------------------------------------------------------------------
pair<PixelRecoPointRZ, bool> MSLayer::crossing(const PixelRecoLineRZ& line) const {
  const float eps = 1.e-5;
  bool inLayer = true;
  float value = (theFace == barrel) ? line.zAtR(thePosition) : line.rAtZ(thePosition);
  if (value > theRange.max()) {
    value = theRange.max() - eps;
    inLayer = false;
  } else if (value < theRange.min()) {
    value = theRange.min() + eps;
    inLayer = false;
  }
  float z = thePosition;
  if (theFace == barrel)
    std::swap(z, value);  // if barrel value is z
  return make_pair(PixelRecoPointRZ(value, z), inLayer);
}
pair<PixelRecoPointRZ, bool> MSLayer::crossing(const SimpleLineRZ& line) const {
  const float eps = 1.e-5;
  bool inLayer = true;
  float value = (theFace == barrel) ? line.zAtR(thePosition) : line.rAtZ(thePosition);
  if (value > theRange.max()) {
    value = theRange.max() - eps;
    inLayer = false;
  } else if (value < theRange.min()) {
    value = theRange.min() + eps;
    inLayer = false;
  }
  float z = thePosition;
  if (theFace == barrel)
    std::swap(z, value);  // if barrel value is z
  return make_pair(PixelRecoPointRZ(value, z), inLayer);
}

//----------------------------------------------------------------------
float MSLayer::distance2(const PixelRecoPointRZ& point) const {
  float u = (theFace == barrel) ? point.r() : point.z();
  float v = (theFace == barrel) ? point.z() : point.r();

  float du = std::abs(u - thePosition);
  if (theRange.inside(v))
    return (du < theHalfThickness) ? 0.f : du * du;

  float dv = (v > theRange.max()) ? v - theRange.max() : theRange.min() - v;
  return sqr(du) + sqr(dv);
}

//----------------------------------------------------------------------
float MSLayer::x0(float cotTheta) const {
  if LIKELY (theX0Data.hasX0) {
    float OverSinTheta = std::sqrt(1.f + cotTheta * cotTheta);
    return (theFace == barrel) ? theX0Data.x0 * OverSinTheta : theX0Data.x0 * OverSinTheta / std::abs(cotTheta);
  } else if (theX0Data.allLayers) {
    const MSLayer* dataLayer = theX0Data.allLayers->layers(cotTheta).findLayer(*this);
    if (dataLayer)
      return dataLayer->x0(cotTheta);
  }
  return 0.;
}

//----------------------------------------------------------------------
float MSLayer::sumX0D(float cotTheta) const {
  if LIKELY (theX0Data.hasX0) {
    switch (theFace) {
      case barrel:
        return theX0Data.sumX0D *
               std::sqrt(std::sqrt((1.f + cotTheta * cotTheta) / (1.f + theX0Data.cotTheta * theX0Data.cotTheta)));
      case endcap:
        return (theX0Data.hasFSlope)
                   ? theX0Data.sumX0D + theX0Data.slopeSumX0D * (1.f / cotTheta - 1.f / theX0Data.cotTheta)
                   : theX0Data.sumX0D;
      case invalidLoc:
        break;  // make gcc happy
    }
  } else if (theX0Data.allLayers) {
    const MSLayer* dataLayer = theX0Data.allLayers->layers(cotTheta).findLayer(*this);
    if (dataLayer)
      return dataLayer->sumX0D(cotTheta);
  }
  return 0.;
}
