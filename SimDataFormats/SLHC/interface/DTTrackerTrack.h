//-------------------------------------------------
//
//   \class TrackerTrack
/**
 *   Description:  store Tracker Tracks 
*/
//   120529 
//   Pierluigi Zotto (from Sara Vanini model for stubs) - Padua University
//
//--------------------------------------------------
#ifndef TrackerTrack_H
#define TrackerTrack_H

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <map>
#include <algorithm>
#include <vector>
#include <set>

#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerDetId.h"

using namespace cmsUpgrades;

//           ---------------------
//           -- Class Interface --
//           ---------------------


class TrackerTrack {
 public:
  // trivial default constructor needed by dictionary building (Ignazio)
  TrackerTrack()
    {
      _rho = NAN;      
      _phi = NAN;
      _theta = NAN;
      _pt =NAN ;
      _ptbin =NAN ;
      _valid = false; 
      _ptFlag = false;    
    }
  // default constructor 
  TrackerTrack(   float const rho, float const phi, float const theta, 
		  float const pt, bool PTflag ) 
    {
      _rho = rho;     
      _phi = phi;
      _theta = theta;
      _pt = pt; 
      _ptbin = pt; 
      _valid = true;
      _ptFlag = PTflag;
      
    }

  // copy constructor (Ignazio)
  TrackerTrack(const TrackerTrack& TrSt) {
    _rho = TrSt.rho();
    _phi = TrSt.phi();
    _theta = TrSt.theta();
    _pt = TrSt.pt(); 
    _valid = true;
    _ptFlag = TrSt.PTFlag();   
  }
  
  // destructor
  ~TrackerTrack(){;} // explicitly trivial (was implicit; Ignazio) 
  
  //return functions 
  inline float  rho()		  const	{ return _rho; }
  inline float 	phi() 		  const	{ return _phi; }
  inline float 	theta() 	  const	{ return _theta; }
  inline float  pt()              const { return _pt; } 
  inline float  ptbin()           const { return _ptbin; } 
  inline bool   valid()           const { return _valid; } 
  inline bool   PTFlag()          const { return _ptFlag; }                   

 private:
  float _phi, _theta, _rho;
  float _pt;
  float _ptbin;
  bool _valid;
  bool _ptFlag;
};


/*struct lt_Track 
{
  bool operator()(const TrackerTrack* a, const TrackerTrack* b) const {
    return ( (a->id()).rawId() < (b->id()).rawId() );
  }
};


typedef std::set<TrackerTrack*, lt_stub> TrackTrack;*/

#endif

