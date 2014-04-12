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
               const PixelRecoPointRZ & pointO) const;

  float sumX0D( int il, int ol,
		const PixelRecoPointRZ & pointI,
               const PixelRecoPointRZ & pointO) const;

  float sumX0D(const PixelRecoPointRZ & pointI,
               const PixelRecoPointRZ & pointM,
               const PixelRecoPointRZ & pointO) const;

  // as used in seeding
  // z at beamline, point on two layers
  float sumX0D(float zV, int il, int ol, 
	       const PixelRecoPointRZ & pointI,
	       const PixelRecoPointRZ & pointO) const;

  int size() const { return theLayers.size(); }
  void print() const;

private:
  std::vector<MSLayer> theLayers;
  std::vector<int> indeces;

private:
  void init();

  typedef std::vector<MSLayer>::const_iterator LayerItr;
  LayerItr findLayer(const PixelRecoPointRZ & point,
                     LayerItr i1, LayerItr i2) const;
  float sum2RmRn(LayerItr i1, LayerItr i2,
                 float rTarget, const SimpleLineRZ & line) const;
};

#endif
