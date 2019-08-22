#include "RecoTracker/TkHitPairs/src/InnerDeltaPhi.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoPointRZ.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionBase.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include "DataFormats/GeometryVector/interface/Basic2DVector.h"
#include "DataFormats/Math/interface/ExtVec.h"

#if !defined(__INTEL_COMPILER)
#define USE_VECTORS_HERE
#endif

using namespace std;

#ifdef VI_DEBUG
namespace {
  struct Stat {
    float xmin = 1.1;
    float xmax = -1.1;
    int nt = 0;
    int nn = 0;
    int nl = 0;

    ~Stat() { std::cout << "ASIN " << xmin << ',' << xmax << ',' << nt << ',' << nn << ',' << nl << std::endl; }
  };

  Stat stat;

}  // namespace

namespace {
  template <class T>
  inline T cropped_asin(T x) {
    stat.nt++;
    if (x < 0.f)
      stat.nn++;
    if (x > 0.5f)
      stat.nl++;
    stat.xmin = std::min(x, stat.xmin);
    stat.xmax = std::max(x, stat.xmax);

    return std::abs(x) <= 1 ? std::asin(x) : (x > 0 ? T(M_PI / 2) : -T(M_PI / 2));
  }
}  // namespace
#else  // for icc
namespace {
  template <class T>
  inline T cropped_asin(T x) {
    return std::abs(x) <= 1 ? std::asin(x) : (x > 0 ? T(M_PI / 2) : -T(M_PI / 2));
  }
}  // namespace
#endif

namespace {

  template <class T>
  inline T sqr(T t) {
    return t * t;
  }

  // reasonable (5.e-4) only for |x|<0.7 then degrades..
  template <typename T>
  inline T f_asin07f(T x) {
    auto ret = 1.f + (x * x) * (0.157549798488616943359375f + (x * x) * 0.125192224979400634765625f);

    return x * ret;
  }

}  // namespace

#include "DataFormats/Math/interface/approx_atan2.h"

namespace {
  inline float f_atan2f(float y, float x) { return unsafe_atan2f<7>(y, x); }
  template <typename V>
  inline float f_phi(V v) {
    return f_atan2f(v.y(), v.x());
  }
}  // namespace

InnerDeltaPhi::InnerDeltaPhi(const DetLayer& outlayer,
                             const DetLayer& layer,
                             const TrackingRegion& region,
                             const edm::EventSetup& iSetup,
                             bool precise,
                             float extraTolerance)
    : innerIsBarrel(layer.isBarrel()),
      outerIsBarrel(outlayer.isBarrel()),
      thePrecise(precise),
      ol(outlayer.seqNum()),
      theROrigin(region.originRBound()),
      theRLayer(0),
      theThickness(0),
      theExtraTolerance(extraTolerance),
      theA(0),
      theB(0),
      theVtxZ(region.origin().z()),
      thePtMin(region.ptMin()),
      theVtx(region.origin().x(), region.origin().y()),
      sigma(&layer, iSetup)

{
  float zMinOrigin = theVtxZ - region.originZBound();
  float zMaxOrigin = theVtxZ + region.originZBound();
  theRCurvature = PixelRecoUtilities::bendingRadius(thePtMin, iSetup);

  if (innerIsBarrel)
    initBarrelLayer(layer);
  else
    initForwardLayer(layer, zMinOrigin, zMaxOrigin);

  if (outerIsBarrel)
    initBarrelMS(outlayer);
  else
    initForwardMS(outlayer);
}

void InnerDeltaPhi::initBarrelMS(const DetLayer& outLayer) {
  const BarrelDetLayer& bl = static_cast<const BarrelDetLayer&>(outLayer);
  float rLayer = bl.specificSurface().radius();
  auto zmax = 0.5f * outLayer.surface().bounds().length();
  PixelRecoPointRZ zero(0., 0.);
  PixelRecoPointRZ point1(rLayer, 0.);
  PixelRecoPointRZ point2(rLayer, zmax);
  auto scatt1 = 3.f * sigma(thePtMin, zero, point1, ol);
  auto scatt2 = 3.f * sigma(thePtMin, zero, point2, ol);
  theDeltaScatt = (scatt2 - scatt1) / zmax;
  theScatt0 = scatt1;
}

void InnerDeltaPhi::initForwardMS(const DetLayer& outLayer) {
  const ForwardDetLayer& fl = static_cast<const ForwardDetLayer&>(outLayer);
  auto minR = fl.specificSurface().innerRadius();
  auto maxR = fl.specificSurface().outerRadius();
  auto layerZ = outLayer.position().z();
  // compute min and max multiple scattering correction
  PixelRecoPointRZ zero(0., theVtxZ);
  PixelRecoPointRZ point1(minR, layerZ);
  PixelRecoPointRZ point2(maxR, layerZ);
  auto scatt1 = 3.f * sigma(thePtMin, zero, point1, ol);
  auto scatt2 = 3.f * sigma(thePtMin, zero, point2, ol);
  theDeltaScatt = (scatt2 - scatt1) / (maxR - minR);
  theScatt0 = scatt1 - theDeltaScatt * minR;
}

void InnerDeltaPhi::initBarrelLayer(const DetLayer& layer) {
  const BarrelDetLayer& bl = static_cast<const BarrelDetLayer&>(layer);
  float rLayer = bl.specificSurface().radius();

  // the maximal delta phi will be for the innermost hits
  theThickness = layer.surface().bounds().thickness();
  theRLayer = rLayer - 0.5f * theThickness;
}

void InnerDeltaPhi::initForwardLayer(const DetLayer& layer, float zMinOrigin, float zMaxOrigin) {
  const ForwardDetLayer& fl = static_cast<const ForwardDetLayer&>(layer);
  theRLayer = fl.specificSurface().innerRadius();
  float layerZ = layer.position().z();
  theThickness = layer.surface().bounds().thickness();
  float layerZmin = layerZ > 0 ? layerZ - 0.5f * theThickness : layerZ + 0.5f * theThickness;
  theB = layerZ > 0 ? zMaxOrigin : zMinOrigin;
  theA = layerZmin - theB;
}

PixelRecoRange<float> InnerDeltaPhi::phiRange(const Point2D& hitXY, float hitZ, float errRPhi) const {
  float rLayer = theRLayer;
  Point2D crossing;

  Point2D dHit = hitXY - theVtx;
  auto dHitmag = dHit.mag();
  float dLayer = 0.;
  float dL = 0.;

  // track is crossing layer with angle such as:
  // this factor should be taken in computation of eror projection
  float cosCross = 0;

  //
  // compute crossing of stright track with inner layer
  //
  if (!innerIsBarrel) {
    auto t = theA / (hitZ - theB);
    auto dt = std::abs(theThickness / (hitZ - theB));
    crossing = theVtx + t * dHit;
    rLayer = crossing.mag();
    dLayer = t * dHitmag;
    dL = dt * dHitmag;
    cosCross = std::abs(dHit.unit().dot(crossing.unit()));
  } else {
    //
    // compute crossing of track with layer
    // dHit - from VTX to outer hit
    // rLayer - layer radius
    // dLayer - distance from VTX to inner layer in direction of dHit
    // vect(rLayer) = vect(rVTX) + vect(dHit).unit * dLayer
    //     rLayer^2 = (vect(rVTX) + vect(dHit).unit * dLayer)^2 and we have square eqation for dLayer
    //
    // barrel case
    //
    auto vtxmag2 = theVtx.mag2();
    if (vtxmag2 < 1.e-10f) {
      dLayer = rLayer;
    } else {
      // there are cancellation here....
      double var_c = vtxmag2 - sqr(rLayer);
      double var_b = theVtx.dot(dHit.unit());
      double var_delta = sqr(var_b) - var_c;
      if (var_delta <= 0.)
        var_delta = 0;
      dLayer = -var_b + std::sqrt(var_delta);  //only the value along vector is OK.
    }
    crossing = theVtx + dHit.unit() * dLayer;
    cosCross = std::abs(dHit.unit().dot(crossing.unit()));
    dL = theThickness / cosCross;
  }

#ifdef USE_VECTORS_HERE
  cms_float32x4_t num{dHitmag, dLayer, theROrigin * (dHitmag - dLayer), 1.f};
  cms_float32x4_t den{2 * theRCurvature, 2 * theRCurvature, dHitmag * dLayer, 1.f};
  auto phis = f_asin07f(num / den);
  phis = phis * dLayer / (rLayer * cosCross);
  auto deltaPhi = std::abs(phis[0] - phis[1]);
  auto deltaPhiOrig = phis[2];
#else
#warning no vector!
  auto alphaHit = cropped_asin(dHitmag / (2 * theRCurvature));
  auto OdeltaPhi = std::abs(alphaHit - cropped_asin(dLayer / (2 * theRCurvature)));
  OdeltaPhi *= dLayer / (rLayer * cosCross);
  // compute additional delta phi due to origin radius
  auto OdeltaPhiOrig = cropped_asin(theROrigin * (dHitmag - dLayer) / (dHitmag * dLayer));
  OdeltaPhiOrig *= dLayer / (rLayer * cosCross);
  // std::cout << "dphi " << OdeltaPhi<<'/'<<OdeltaPhiOrig << ' ' << deltaPhi<<'/'<<deltaPhiOrig << std::endl;

  auto deltaPhi = OdeltaPhi;
  auto deltaPhiOrig = OdeltaPhiOrig;
#endif

  // additinal angle due to not perpendicular stright line crossing  (for displaced beam)
  //  double dPhiCrossing = (cosCross > 0.9999) ? 0 : dL *  sqrt(1-sqr(cosCross))/ rLayer;
  Point2D crossing2 = theVtx + dHit.unit() * (dLayer + dL);
  auto phicross2 = f_phi(crossing2);
  auto phicross1 = f_phi(crossing);
  auto dphicross = phicross2 - phicross1;
  if (dphicross < -float(M_PI))
    dphicross += float(2 * M_PI);
  if (dphicross > float(M_PI))
    dphicross -= float(2 * M_PI);
  if (dphicross > float(M_PI / 2))
    dphicross = 0.;  // something wrong?
  phicross2 = phicross1 + dphicross;

  // inner hit error taken as constant
  auto deltaPhiHit = theExtraTolerance / rLayer;

  // outer hit error
  //   double deltaPhiHitOuter = errRPhi/rLayer;
  auto deltaPhiHitOuter = errRPhi / hitXY.mag();

  auto margin = deltaPhi + deltaPhiOrig + deltaPhiHit + deltaPhiHitOuter;

  if (thePrecise) {
    // add multiple scattering correction

    /*
    PixelRecoPointRZ zero(0., theVtxZ);
    PixelRecoPointRZ point(hitXY.mag(), hitZ);
    auto scatt = 3.f*sigma(thePtMin,zero, point, ol); 
    */

    auto w = outerIsBarrel ? std::abs(hitZ) : hitXY.mag();
    auto nscatt = theScatt0 + theDeltaScatt * w;

    // std::cout << "scatt " << (outerIsBarrel ? "B" : "F") << (innerIsBarrel ? "B " : "F ")
    //          << scatt << ' ' << nscatt << ' ' << nscatt/scatt << std::endl;

    margin += nscatt / rLayer;
  }

  return PixelRecoRange<float>(std::min(phicross1, phicross2) - margin, std::max(phicross1, phicross2) + margin);
}
