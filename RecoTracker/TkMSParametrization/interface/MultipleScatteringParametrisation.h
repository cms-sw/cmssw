#ifndef MultipleScatteringParametrisation_H
#define MultipleScatteringParametrisation_H 

/** \class MultipleScatteringParametrisation
 * Parametrisation of multiple scattering sigma in tracker.
 */
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoPointRZ.h"
#include "RecoTracker/TkMSParametrization/interface/MSLayer.h"
#include "FWCore/Framework/interface/EventSetup.h"

class MSLayersKeeper;
class PixelRecoPointRZ;
class DetLayer;



class MultipleScatteringParametrisation {

public:

  enum X0Source { useDetLayer, useX0AtEta, useX0DataAveraged };
  enum Consecutive { notAssumeConsecutive, useConsecutive };

  MultipleScatteringParametrisation( const DetLayer* layer, 
				     const edm::EventSetup &iSetup,
                                     X0Source x0source = useX0AtEta);


  /// MS sigma  at the layer for which parametrisation is initialised;
  /// particle assumed to come from nominal vertex, "fast" methods called
  float operator()(float pt, float cotTheta) const;

  /// MS sigma  at the layer for which parametrisation is initialised;
  /// particle assumed to come from constraint point (inner to layer).
  /// layer by layer contribution is calculated
  float operator()(float pt,
                   float cotTheta,
                   const PixelRecoPointRZ & point) const;

  /// MS sigma  at the layer for which parametrisation is initialised;
  /// particle assumed to be measured at point1 and point2,
  /// it is assumed that layer is between point1 and point2.
  /// layer by layer contribution is calculated
  float operator()(float pt,
                   const PixelRecoPointRZ & point1,
                   const PixelRecoPointRZ & point2,
                   Consecutive consecutive = notAssumeConsecutive) const;

private:

  MSLayer theLayer;
  MSLayersKeeper * theLayerKeeper;
  static const float x0ToSigma;

};
#endif
