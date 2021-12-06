#ifndef RecoTracker_TkMSParametrization_MSLayersKeepers_h
#define RecoTracker_TkMSParametrization_MSLayersKeepers_h

#include "FWCore/Utilities/interface/propagate_const.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisation.h"

class DetLayer;

class MultipleScatteringParametrisationMaker {
public:
  enum class X0Source { useDetLayer = 0, useX0AtEta = 1, useX0DataAveraged = 2 };

  MultipleScatteringParametrisationMaker(GeometricSearchTracker const& tracker, MagneticField const& bfield);
  ~MultipleScatteringParametrisationMaker();

  MultipleScatteringParametrisation parametrisation(const DetLayer* layer,
                                                    X0Source x0Source = X0Source::useX0AtEta) const;

private:
  struct Keepers;
  edm::propagate_const<std::unique_ptr<Keepers>> impl_;
};

#endif
