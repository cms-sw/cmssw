
#include <vector>
#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisation.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoPointRZ.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoLineRZ.h"

template <class T>
inline T sqr(T t) {
  return t * t;
}

#include "MSLayersKeeper.h"
#include "MSLayersAtAngle.h"

using namespace std;

const float MultipleScatteringParametrisation::x0ToSigma = 0.0136f;

//----------------------------------------------------------------------
MultipleScatteringParametrisation::MultipleScatteringParametrisation(const DetLayer *layer,
                                                                     const MSLayersKeeper *layerKeeper)
    : theLayerKeeper(layerKeeper) {
  if (layer) {
    theLayer = theLayerKeeper->layer(layer);
  }
}

//----------------------------------------------------------------------
float MultipleScatteringParametrisation::operator()(float pT, float cotTheta, float) const {
  float sumX0D = theLayer.sumX0D(cotTheta);
  return x0ToSigma * sumX0D / pT;
}

//----------------------------------------------------------------------
float MultipleScatteringParametrisation::operator()(float pT,
                                                    float cotTheta,
                                                    const PixelRecoPointRZ &pointI,
                                                    float tip) const {
  PixelRecoLineRZ lineIO(pointI, cotTheta, tip);
  PixelRecoPointRZ pointO = theLayer.crossing(lineIO).first;

  const MSLayersAtAngle &layersAtEta = theLayerKeeper->layers(cotTheta);

  float sumX0D = layersAtEta.sumX0D(pointI, pointO);
  return x0ToSigma * sumX0D / pT;
}

float MultipleScatteringParametrisation::operator()(float pT,
                                                    float cotTheta,
                                                    const PixelRecoPointRZ &pointI,
                                                    int il) const {
  PixelRecoLineRZ lineIO(pointI, cotTheta);
  PixelRecoPointRZ pointO = theLayer.crossing(lineIO).first;

  const MSLayersAtAngle &layersAtEta = theLayerKeeper->layers(cotTheta);

  float sumX0D = layersAtEta.sumX0D(il, theLayer.seqNum(), pointI, pointO);
  return x0ToSigma * sumX0D / pT;
}

//----------------------------------------------------------------------
float MultipleScatteringParametrisation::operator()(float pT,
                                                    const PixelRecoPointRZ &pointI,
                                                    const PixelRecoPointRZ &pointO,
                                                    Consecutive consecutive,
                                                    float tip) const {
  PixelRecoLineRZ lineIO(pointI, pointO, tip);
  PixelRecoPointRZ pointM = theLayer.crossing(lineIO).first;
  float cotTheta = lineIO.cotLine();

  if (consecutive == useConsecutive) {
    float dist = fabs((pointO.r() - pointM.r()) * (pointM.r() - pointI.r()) / (pointO.r() - pointI.r()));
    return x0ToSigma * sqrt(theLayer.x0(cotTheta)) * dist / pT;
  } else {
    const MSLayersAtAngle &layersAtEta = theLayerKeeper->layers(cotTheta);
    float sumX0D = layersAtEta.sumX0D(pointI, pointM, pointO);
    return x0ToSigma * sumX0D / pT;
  }
}

float MultipleScatteringParametrisation::operator()(float pT,
                                                    const PixelRecoPointRZ &pointV,
                                                    const PixelRecoPointRZ &pointO,
                                                    int ol) const {
  PixelRecoLineRZ lineIO(pointV, pointO);
  PixelRecoPointRZ pointI = theLayer.crossing(lineIO).first;
  float cotTheta = lineIO.cotLine();

  const MSLayersAtAngle &layersAtEta = theLayerKeeper->layers(cotTheta);
  float sumX0D = layersAtEta.sumX0D(pointV.z(), theLayer.seqNum(), ol, pointI, pointO);
  return x0ToSigma * sumX0D / pT;
}
