#ifndef RecoTracker_PixelSeeding_ThirdHitRZPrediction_h
#define RecoTracker_PixelSeeding_ThirdHitRZPrediction_h

/** predicts a range in r-z for the position of third hit.
 *  the predicted reange is defined by the template argument, which
 *  is a straight line extrapolation/interpolation if PixelRecoLineRZ is used.
 */

#include <algorithm>

#include "RecoTracker/PixelSeeding/interface/ThirdHitRZPredictionBase.h"

template <class Propagator>
class ThirdHitRZPrediction : public ThirdHitRZPredictionBase {
public:
  ThirdHitRZPrediction() : ThirdHitRZPredictionBase(), thePropagator(nullptr) {}
  ThirdHitRZPrediction(const Propagator *propagator, float tolerance, const DetLayer *layer = nullptr)
      : ThirdHitRZPredictionBase(tolerance, layer), thePropagator(propagator) {}

  inline Range operator()(const DetLayer *layer = nullptr);
  inline Range operator()(float rORz) const { return (*this)(rORz, *thePropagator); }
  inline Range operator()(float rORz, const Propagator &propagator) const;

  void initPropagator(const Propagator *propagator) { thePropagator = propagator; }

private:
  float transform(const Propagator &propagator, float rOrZ) const {
    return theBarrel ? propagator.zAtR(rOrZ) : propagator.rAtZ(rOrZ);
  }

  const Propagator *thePropagator;
};

template <class Propagator>
typename ThirdHitRZPrediction<Propagator>::Range ThirdHitRZPrediction<Propagator>::operator()(const DetLayer *layer) {
  if (layer)
    initLayer(layer);
  if (!theBarrel && !theForward)
    return Range(0., 0.);
  float v1 = transform(*thePropagator, theDetRange.min());
  float v2 = transform(*thePropagator, theDetRange.max());
  if (v1 > v2)
    std::swap(v1, v2);
  return Range(v1 - theTolerance.left(), v2 + theTolerance.right());
}

template <class Propagator>
typename ThirdHitRZPrediction<Propagator>::Range ThirdHitRZPrediction<Propagator>::operator()(
    float rOrZ, const Propagator &propagator) const {
  float v = transform(propagator, rOrZ);
  return Range(v - theTolerance.left(), v + theTolerance.right());
}

#endif
