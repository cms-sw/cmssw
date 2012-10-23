//-------------------------------------------------
//
//   \class TrackerTracklet
/**
 *   Description:  store Tracker Tracklets 
*/
//   110531 
//   Pierluigi Zotto (from Sara Vanini model for stubs) - Padua University
//
//--------------------------------------------------
#ifndef TrackerTracklet_H
#define TrackerTracklet_H

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


class TrackerTracklet {
 public:
  // trivial default constructor needed by dictionary building (Ignazio)
  TrackerTracklet()
    {
      _sl = -1; 
      _rho = NAN;      
      _phi = NAN;
      _theta = NAN;
      _pt =NAN ;
      _valid = false; 
      _ptFlag = false;    
    }
  // default constructor 
  TrackerTracklet(int const sl,
		  float const rho, float const phi, float const theta, 
		  float const pt, bool PTflag ) 
    {
      _sl = sl;
      _rho = rho;     
      _phi = phi;
      _theta = theta;
      _pt = pt; 
      _valid = true;
      _ptFlag = PTflag;
      
    }

  // copy constructor (Ignazio)
  TrackerTracklet(const TrackerTracklet& TrSt) {
    _sl = TrSt.sl();
    _rho = TrSt.rho();
    _phi = TrSt.phi();
    _theta = TrSt.theta();
    _pt = TrSt.pt(); 
    _valid = true;
    _ptFlag = TrSt.PTFlag();   
  }
  
  // destructor
  ~TrackerTracklet(){;} // explicitly trivial (was implicit; Ignazio) 
  
  //return functions 
  inline int    sl()              const	{ return _sl; }
  inline float  rho()		  const	{ return _rho; }
  inline float 	phi() 		  const	{ return _phi; }
  inline float 	theta() 	  const	{ return _theta; }
  inline float  pt()              const { return _pt; } 
  inline bool   valid()           const { return _valid; } 
  inline bool   PTFlag()           const { return _ptFlag; }                   

 private:
  int _sl;
  float _phi, _theta, _rho;
  float _pt;
  bool _valid;
  bool _ptFlag;
};


/*struct lt_tracklet 
{
  bool operator()(const TrackerTracklet* a, const TrackerTracklet* b) const {
    return ( (a->id()).rawId() < (b->id()).rawId() );
  }
};


typedef std::set<TrackerTracklet*, lt_stub> TrackletTracklet;*/

#endif

