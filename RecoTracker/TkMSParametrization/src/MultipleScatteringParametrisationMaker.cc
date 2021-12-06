#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisationMaker.h"

#include "MSLayersKeeper.h"
#include "MSLayersKeeperX0AtEta.h"
#include "MSLayersKeeperX0Averaged.h"
#include "MSLayersKeeperX0DetLayer.h"
#include "MSLayersAtAngle.h"

struct MultipleScatteringParametrisationMaker::Keepers {
  Keepers(GeometricSearchTracker const& tracker, MagneticField const& bfield)
      : x0AtEta(tracker, bfield), x0Averaged(tracker, bfield), keepers{&x0DetLayer, &x0AtEta, &x0Averaged} {}

  MSLayersKeeperX0DetLayer x0DetLayer;
  MSLayersKeeperX0AtEta x0AtEta;
  MSLayersKeeperX0Averaged x0Averaged;
  MSLayersKeeper* keepers[3];  // {&x0DetLayer,&x0AtEta,&x0Averaged};
};

MultipleScatteringParametrisationMaker::MultipleScatteringParametrisationMaker(GeometricSearchTracker const& tracker,
                                                                               MagneticField const& bfield)
    : impl_(std::make_unique<Keepers>(tracker, bfield)) {}

MultipleScatteringParametrisationMaker::~MultipleScatteringParametrisationMaker() = default;

MultipleScatteringParametrisation MultipleScatteringParametrisationMaker::parametrisation(const DetLayer* layer,
                                                                                          X0Source x0Source) const {
  return MultipleScatteringParametrisation(layer, impl_->keepers[static_cast<int>(x0Source)]);
}
