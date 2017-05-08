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
    Amplitude() : _amp(0.0)
    {}
    Amplitude(float amp, float frac) :
      _amp(amp), _frac(1, frac), _hitInfo() 
    {
      // in case of digi from noisypixels
      // the MC information are removed
      if (_frac[0] < -0.5)
	_frac.pop_back();
    }

    Amplitude( float amp, const PSimHit* hitp, size_t hitIndex, unsigned int tofBin, float frac) :
    _amp(amp), _frac(1, frac), _hitInfo(new SimHitInfoForLinks(hitp, hitIndex, tofBin)) {
      // in case of digi from noisypixels
      // the MC information are removed
      if (_frac[0] < -0.5) {
	_frac.pop_back();
	_hitInfo->trackIds_.pop_back();
      }
    }

    // can be used as a float by convers.
    operator float() const {return _amp;}
    float ampl() const {return _amp;}
    std::vector<float> individualampl() const {return _frac;}
    const std::vector<unsigned int>& trackIds() const {return _hitInfo->trackIds_;}
    const std::shared_ptr<SimHitInfoForLinks>& hitInfo() const {return _hitInfo;}

    void operator+= (const Amplitude& other) {
      _amp += other._amp;

      // in case of contribution of noise to the digi
      // the MC information are removed
      if (other._frac.size() > 0 && other._frac[0] >- 0.5) {
        if (other._hitInfo) {
          std::vector<unsigned int>& otherTrackIds = other._hitInfo->trackIds_;
          if (_hitInfo) {
            std::vector<unsigned int>& trackIds = _hitInfo->trackIds_;
	    trackIds.insert(trackIds.end(), otherTrackIds.begin(), otherTrackIds.end());
          } 
	  else 
            _hitInfo.reset(new SimHitInfoForLinks(*other._hitInfo));
        }
	_frac.insert(_frac.end(), other._frac.begin(), other._frac.end());
      }
    }
    const EncodedEventId& eventId() const {
      return _hitInfo->eventId_;
    }
    const unsigned int hitIndex() const {
      return _hitInfo->hitIndex_;
    }
    const unsigned int tofBin() const {
      return _hitInfo->tofBin_;
    }
    void operator+= (const float& amp) {
      _amp += amp;
    }
    void set (const float amplitude) { // Used to reset the amplitude
      _amp = amplitude;
    }
    // void setind (const float indamplitude) { // Used to reset the amplitude
    // _frac = idamplitude;
    // }

  private:
    float _amp;
    std::vector<float> _frac;
    std::shared_ptr<SimHitInfoForLinks> _hitInfo;
  };

  //*********************************************************
  // Define a class for 3D ionization points and energy
  //*********************************************************
  class EnergyDepositUnit {
  public:
    EnergyDepositUnit(): _energy(0),_position(0,0,0) {}
    EnergyDepositUnit(float energy,float x, float y, float z):
      _energy(energy),_position(x,y,z) {}
    EnergyDepositUnit(float energy, Local3DPoint position):
      _energy(energy),_position(position) {}
    float x() const{return _position.x();}
    float y() const{return _position.y();}
    float z() const{return _position.z();}
    float energy() const { return _energy;}
  private:
    float _energy;
    Local3DPoint _position;
  };

  //**********************************************************
  // define class to store signals on the collection surface
  //**********************************************************
  class SignalPoint {
  public:
    SignalPoint(): _pos(0,0), _time(0), _amplitude(0), 
      _sigma_x(1.), _sigma_y(1.), _hitp(0) {}
    
    SignalPoint(float x, float y, float sigma_x, float sigma_y,
		float t, float a=1.0):
      _pos(x,y), _time(t), _amplitude(a), _sigma_x(sigma_x), 
      _sigma_y(sigma_y), _hitp(0) {}
    
    SignalPoint(float x, float y, float sigma_x, float sigma_y,
		float t, const PSimHit& hit, float a=1.0):
      _pos(x,y), _time(t), _amplitude(a), _sigma_x(sigma_x), 
      _sigma_y(sigma_y),_hitp(&hit) {}
    
    const LocalPoint& position() const {return _pos;}
    float x()         const {return _pos.x();}
    float y()         const {return _pos.y();}
    float sigma_x()   const {return _sigma_x;}
    float sigma_y()   const {return _sigma_y;}
    float time()      const {return _time;}
    float amplitude() const {return _amplitude;}
    const PSimHit& hit() {return *_hitp;}
    SignalPoint& set_amplitude(float amp) {_amplitude = amp; return *this;}
  private:
    LocalPoint     _pos;
    float          _time;
    float          _amplitude;
    float          _sigma_x;   // gaussian sigma in the x direction (cm)
    float          _sigma_y;   //    "       "          y direction (cm) */
    const PSimHit* _hitp;
  };
  struct DigiSimInfo {
    int sig_tot;
    std::map<unsigned int, float> track_map;
    unsigned int hit_counter;
    unsigned int tof_bin; 
    EncodedEventId event_id;
  };
}
#endif
