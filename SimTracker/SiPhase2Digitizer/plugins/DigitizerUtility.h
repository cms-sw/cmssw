#ifndef __SimTracker_SiPhase2Digitizer_DigitizerUtility_h
#define __SimTracker_SiPhase2Digitizer_DigitizerUtility_h

#include <map>
#include <memory>
#include <vector>
#include <iostream>

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimTracker/Common/interface/SimHitInfoForLinks.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

namespace DigitizerUtility {
  class Amplitude {
  public:
    Amplitude() : _amp(0.0) {}
    Amplitude(float amp, const PSimHit* hitp, float frac = 0, size_t hitIndex = 0, uint32_t tofBin = 0) : _amp(amp) {
      if (frac > 0) {
        if (hitp != nullptr)
          _simInfoList.push_back({frac, std::make_unique<SimHitInfoForLinks>(hitp, hitIndex, tofBin)});
        else
          _simInfoList.push_back({frac, nullptr});
      }
    }

    // can be used as a float by convers.
    operator float() const { return _amp; }
    float ampl() const { return _amp; }
    const std::vector<std::pair<float, std::unique_ptr<SimHitInfoForLinks> > >& simInfoList() const {
      return _simInfoList;
    }

    void operator+=(const Amplitude& other) {
      _amp += other._amp;
      // in case of digi from the noise, the MC information need not be there
      for (auto const& ic : other.simInfoList()) {
        if (ic.first > -0.5)
          _simInfoList.push_back({ic.first, std::make_unique<SimHitInfoForLinks>(*ic.second)});
      }
    }
    void operator+=(const float& amp) { _amp += amp; }
    void set(const float amplitude) {  // Used to reset the amplitude
      _amp = amplitude;
    }
    // void setind (const float indamplitude) { // Used to reset the amplitude
    // _frac = idamplitude;
    // }

  private:
    float _amp;
    std::vector<std::pair<float, std::unique_ptr<SimHitInfoForLinks> > > _simInfoList;
  };

  //*********************************************************
  // Define a class for 3D ionization points and energy
  //*********************************************************
  class EnergyDepositUnit {
  public:
    EnergyDepositUnit() : _energy(0), _position(0, 0, 0) {}
    EnergyDepositUnit(float energy, float x, float y, float z) : _energy(energy), _position(x, y, z) {}
    EnergyDepositUnit(float energy, Local3DPoint position) : _energy(energy), _position(position) {}
    float x() const { return _position.x(); }
    float y() const { return _position.y(); }
    float z() const { return _position.z(); }
    float energy() const { return _energy; }

  private:
    float _energy;
    Local3DPoint _position;
  };

  //**********************************************************
  // define class to store signals on the collection surface
  //**********************************************************
  class SignalPoint {
  public:
    SignalPoint() : _pos(0, 0), _time(0), _amplitude(0), _sigma_x(1.), _sigma_y(1.), _hitp(nullptr) {}

    SignalPoint(float x, float y, float sigma_x, float sigma_y, float t, float a = 1.0)
        : _pos(x, y), _time(t), _amplitude(a), _sigma_x(sigma_x), _sigma_y(sigma_y), _hitp(nullptr) {}

    SignalPoint(float x, float y, float sigma_x, float sigma_y, float t, const PSimHit& hit, float a = 1.0)
        : _pos(x, y), _time(t), _amplitude(a), _sigma_x(sigma_x), _sigma_y(sigma_y), _hitp(&hit) {}

    const LocalPoint& position() const { return _pos; }
    float x() const { return _pos.x(); }
    float y() const { return _pos.y(); }
    float sigma_x() const { return _sigma_x; }
    float sigma_y() const { return _sigma_y; }
    float time() const { return _time; }
    float amplitude() const { return _amplitude; }
    const PSimHit& hit() { return *_hitp; }
    SignalPoint& set_amplitude(float amp) {
      _amplitude = amp;
      return *this;
    }

  private:
    LocalPoint _pos;
    float _time;
    float _amplitude;
    float _sigma_x;  // gaussian sigma in the x direction (cm)
    float _sigma_y;  //    "       "          y direction (cm) */
    const PSimHit* _hitp;
  };
  struct DigiSimInfo {
    int sig_tot;
    bool ot_bit;
    std::vector<std::pair<float, SimHitInfoForLinks*> > simInfoList;
  };
}  // namespace DigitizerUtility
#endif
