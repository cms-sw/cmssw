#ifndef MultipleScatteringParametrisation_H
#define MultipleScatteringParametrisation_H 

/** \class MultipleScatteringParametrisation
 * Parametrisation of multiple scattering sigma in tracker.
 */
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoPointRZ.h"
#include "RecoTracker/TkMSParametrization/interface/MSLayer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/GCC11Compatibility.h"


class MSLayersKeeper;
class PixelRecoPointRZ;
class DetLayer;



class MultipleScatteringParametrisation {

public:

  static void initKeepers(const edm::EventSetup &iSetup);

  enum X0Source { useDetLayer=0, useX0AtEta=1, useX0DataAveraged=2 };
  enum Consecutive { notAssumeConsecutive, useConsecutive };

  MultipleScatteringParametrisation( const DetLayer* layer, 
				     const edm::EventSetup &iSetup,
                                     X0Source x0source = useX0AtEta);


  /// MS sigma  at the layer for which parametrisation is initialised;
  /// particle assumed to come from nominal vertex, "fast" methods called
  float operator()(float pt, float cotTheta, float transverseIP = 0.) const;

  /// MS sigma  at the layer for which parametrisation is initialised;
  /// particle assumed to come from constraint point (inner to layer).
  /// layer by layer contribution is calculated
  float operator()(float pt,
                   float cotTheta,
                   const PixelRecoPointRZ & point,
                   float transverseIP=0.) const;
  float operator()(float pt,
                   float cotTheta,
                   const PixelRecoPointRZ & point, int ol) const;


  /// MS sigma  at the layer for which parametrisation is initialised;
  /// particle assumed to be measured at point1 and point2,
  /// it is assumed that layer is between point1 and point2.
  /// layer by layer contribution is calculated
  float operator()(float pt,
                   const PixelRecoPointRZ & point1,
                   const PixelRecoPointRZ & point2,
                   Consecutive consecutive = notAssumeConsecutive,
                   float transverseIP = 0.) const;

  // as above, pointV is at vertex and pointO is on layer ol
  float operator()(
		   float pT,
		   const PixelRecoPointRZ & pointV,
		   const PixelRecoPointRZ & pointO,
		   int ol) const;


private:

  MSLayer theLayer;
  MSLayersKeeper const * theLayerKeeper;
  static const float x0ToSigma;

};
#endif
