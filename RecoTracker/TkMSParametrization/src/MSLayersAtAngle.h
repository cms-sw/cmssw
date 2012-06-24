#ifndef MSLayersAtAngle_H
#define MSLayersAtAngle_H

/**
 *
 */

#include <vector>
#include <cmath>

#include "RecoTracker/TkMSParametrization/interface/MSLayer.h"
#include "FWCore/Utilities/interface/GCC11Compatibility.h"

class PixelRecoLineRZ;

class dso_hidden MSLayersAtAngle {

public:
  MSLayersAtAngle() { }
  MSLayersAtAngle(const std::vector<MSLayer> & layers);
  void update(const MSLayer & layer);
  const MSLayer * findLayer(const MSLayer & layer) const;

  float sumX0D(const PixelRecoPointRZ & pointI,
               const PixelRecoPointRZ & pointO,
               float tip = 0.) const;
  float sumX0D(const PixelRecoPointRZ & pointI,
               const PixelRecoPointRZ & pointM,
               const PixelRecoPointRZ & pointO,
               float tip = 0.) const;

  int size() const { return theLayers.size(); }
  void print() const;

private:
  std::vector<MSLayer> theLayers;

private:
  typedef std::vector<MSLayer>::const_iterator LayerItr;
  LayerItr findLayer(const PixelRecoPointRZ & point,
                     LayerItr i1, LayerItr i2) const;
  float sum2RmRn(LayerItr i1, LayerItr i2,
                 float rTarget, const PixelRecoLineRZ & line) const;
};

#endif
