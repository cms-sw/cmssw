#ifndef RKPropagatorDefault_H
#define RKPropagatorDefault_H

#include "TrackPropagation/RungeKutta/interface/RKPropagatorInS.h"
#include "MagneticField/VolumeGeometry/interface/MagneticFieldProvider.h"

namespace defaultRKPropagator {

  using RKPropagator = RKPropagatorInS;

  // not clear why we need all this
  class TrivialFieldProvider final : public MagneticFieldProvider<float> {
  public:
    TrivialFieldProvider(const MagneticField* field) : theField(field) {}

    LocalVectorType valueInTesla(const LocalPointType& lp) const override {
      // NOTE: the following transformation only works for the central volume
      // where global and local coordinates are numerically equal !
      GlobalPoint gp(lp.basicVector());
      return LocalVectorType(theField->inTesla(gp).basicVector());
    }

  private:
    const MagneticField* theField;
  };

  class RKMagVolume final : public MagVolume {
  public:
    RKMagVolume(const PositionType& pos, const RotationType& rot, const MagneticFieldProvider<float>* mfp)
        : MagVolume(pos, rot, mfp) {}

    bool inside(const GlobalPoint& gp, double tolerance = 0.) const override { return true; }

    /// Access to volume faces - dummy implementation
    const std::vector<VolumeSide>& faces() const override { return theFaces; }

  private:
    std::vector<VolumeSide> theFaces;
  };

  struct Product {
    explicit Product(const MagneticField* field, PropagationDirection dir = alongMomentum, double tolerance = 5.e-5)
        : mpf(field),
          volume(MagVolume::PositionType(0, 0, 0), MagVolume::RotationType(), &mpf),
          propagator(volume, dir, tolerance) {}
    TrivialFieldProvider mpf;
    RKMagVolume volume;
    RKPropagator propagator;
  };

}  // namespace defaultRKPropagator

#endif  // RKPropagatorDefault
