#ifndef SimTracker_Common_DigitizerUtility_h
#define SimTracker_Common_DigitizerUtility_h

#include <map>
#include <memory>
#include <vector>
#include <iostream>

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimTracker/Common/interface/SimHitInfoForLinks.h"

namespace digitizerUtility {

  class SimHitInfo {
  public:
    SimHitInfo(const PSimHit* hitp, float corrTime, size_t hitIndex, uint32_t tofBin)
        : eventId_(hitp->eventId()), trackId_(hitp->trackId()), hitIndex_(hitIndex), tofBin_(tofBin), time_(corrTime) {}

    uint32_t hitIndex() const { return hitIndex_; };
    uint32_t tofBin() const { return tofBin_; };
    EncodedEventId eventId() const { return eventId_; };
    uint32_t trackId() const { return trackId_; };
    float time() const { return time_; };

  private:
    EncodedEventId eventId_;
    uint32_t trackId_;
    uint32_t hitIndex_;
    uint32_t tofBin_;
    float time_;
  };

  //===================================================================================================
  class Amplitude {
  public:
    Amplitude() : _amp(0.0) {}
    Amplitude(float amp, float frac) : _amp(amp), _frac(1, frac) {
      //in case of digi from noisypixels
      //the MC information are removed
      if (_frac[0] < -0.5) {
        _frac.pop_back();
      }
    }

    Amplitude(float amp, const PSimHit* hitp, size_t hitIndex, size_t hitInd4CR, unsigned int tofBin, float frac)
        : _amp(amp), _frac(1, frac) {
      //in case of digi from noisypixels
      //the MC information are removed
      if (_frac[0] < -0.5) {
        _frac.pop_back();
      } else {
        _hitInfos.emplace_back(hitp, hitIndex, tofBin, hitInd4CR, amp);
      }
    }

    // can be used as a float by convers.
    operator float() const { return _amp; }
    float ampl() const { return _amp; }
    const std::vector<float>& individualampl() const { return _frac; }
    const std::vector<SimHitInfoForLinks>& hitInfos() const { return _hitInfos; }

    void operator+=(const Amplitude& other) {
      _amp += other._amp;
      //in case of contribution of noise to the digi
      //the MC information are removed
      if (other._frac[0] > -0.5) {
        if (!other._hitInfos.empty()) {
          _hitInfos.insert(_hitInfos.end(), other._hitInfos.begin(), other._hitInfos.end());
        }
        _frac.insert(_frac.end(), other._frac.begin(), other._frac.end());
      }
    }
    void operator+=(const float& amp) { _amp += amp; }

    void set(const float amplitude) {  // Used to reset the amplitude
      _amp = amplitude;
    }
    /*     void setind (const float indamplitude) {  // Used to reset the amplitude */
    /*       _frac = idamplitude; */
    /*     } */
  private:
    float _amp;
    std::vector<float> _frac;
    std::vector<SimHitInfoForLinks> _hitInfos;
  };  // end class Amplitude

  //===================================================================================================
  class Ph2Amplitude {
  public:
    Ph2Amplitude() : _amp(0.0) {}
    Ph2Amplitude(
        float amp, const PSimHit* hitp, float frac = 0, float tcor = 0, size_t hitIndex = 0, uint32_t tofBin = 0)
        : _amp(amp) {
      if (frac > 0) {
        if (hitp != nullptr)
          _simInfoList.push_back({frac, std::make_unique<SimHitInfo>(hitp, tcor, hitIndex, tofBin)});
        else
          _simInfoList.push_back({frac, nullptr});
      }
    }

    // can be used as a float by convers.
    operator float() const { return _amp; }
    float ampl() const { return _amp; }
    const std::vector<std::pair<float, std::unique_ptr<SimHitInfo> > >& simInfoList() const { return _simInfoList; }

    void operator+=(const Ph2Amplitude& other) {
      _amp += other._amp;
      // in case of digi from the noise, the MC information need not be there
      for (auto const& ic : other.simInfoList()) {
        if (ic.first > -0.5)
          _simInfoList.push_back({ic.first, std::make_unique<SimHitInfo>(*ic.second)});
      }
    }
    void operator+=(const float& amp) { _amp += amp; }
    void set(const float amplitude) {  // Used to reset the amplitude
      _amp = amplitude;
    }

  private:
    float _amp;
    std::vector<std::pair<float, std::unique_ptr<SimHitInfo> > > _simInfoList;
  };  // end class Ph2Amplitude

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

    // Allow migration between pixel cells
    void migrate_position(const Local3DPoint& pos) { _position = pos; }

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
    std::vector<std::pair<float, SimHitInfo*> > simInfoList;
  };
}  // namespace digitizerUtility
#endif
