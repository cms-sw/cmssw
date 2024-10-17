#ifndef SimCalorimetry_HGCalSimAlgos_HGCalECONDEmulator_h
#define SimCalorimetry_HGCalSimAlgos_HGCalECONDEmulator_h

#include <cstddef>

#include "SimCalorimetry/HGCalSimAlgos/interface/HGCalECONDEmulatorParameters.h"
#include "SimCalorimetry/HGCalSimAlgos/interface/SlinkTypes.h"

namespace hgcal::econd {
  /// Pure virtual base class for a ECON-D event emulator implementation
  class Emulator {
  public:
    explicit Emulator(const EmulatorParameters& params) : params_(params) {}
    virtual ~Emulator() = default;

    /// Fetch the next ECON-D event
    virtual ECONDInput next() = 0;

  protected:
    const EmulatorParameters params_;
  };

  /// A "trivial" ECON-D emulator emulating non-empty ECON-D events
  class TrivialEmulator : public Emulator {
  public:
    using Emulator::Emulator;

    ECONDInput next() override;

  private:
    uint32_t event_id_{1}, bx_id_{2}, orbit_id_{3};
  };
}  // namespace hgcal::econd

#endif
