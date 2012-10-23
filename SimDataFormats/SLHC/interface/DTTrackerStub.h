//-------------------------------------------------
//
//   \class TrackerStub
/**
 *   Description:  store Tracker Stubs 
*/
//   090320 
//   Sara Vanini - Padua University
//   Modifications by Ignazio - Trento University
//
//--------------------------------------------------
#ifndef TrackerStub_H
#define TrackerStub_H

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


class TrackerStub {
 public:
  // trivial default constructor needed by dictionary building (Ignazio)
  TrackerStub()
    {
      _id = StackedTrackerDetId();     
      _x = NAN;
      _y = NAN;
      _z = NAN;
      _rho = NAN;      
      _phi = NAN;
      _theta = NAN;
      _direction = GlobalVector();
      _valid = false;  //Ignazio
      _PTflag = false; //PLZ  
      _MCid = 0; //PLZ      
    }
  // default constructor (with "direction" added by Ignazio)
  TrackerStub(StackedTrackerDetId const id, 
              float const x, float const y, float const z, 
	      GlobalVector const direction, float const phi, 
	      float const theta, bool PTflag,int const MCid) 
    {
      _id = id;      
      _x = x;
      _y = y;
      _z = z;
      _rho = sqrt(x*x + y*y);
      _direction = direction;      
      _phi = phi;
      _theta = theta;
      _valid = true;  //Ignazio   
     _PTflag = PTflag;   
     _MCid = MCid;
    }

  // copy constructor (Ignazio)
  TrackerStub(const TrackerStub& TrSt) {
    _id = TrSt.id();
    _phi = TrSt.phi();
    _theta = TrSt.theta();    
    _x = TrSt.x();
    _y = TrSt.y();
    _z = TrSt.z();
    _rho = TrSt.rho();
    _valid = true;     
    _direction = TrSt.direction(); 
    _PTflag = TrSt.PTflag();  
    _MCid = TrSt.MCid();   
  }
  
  // destructor
  ~TrackerStub(){;} // explicitly trivial (was implicit; Ignazio) 
  
  //return functions 
  inline StackedTrackerDetId id() const	{ return _id; }
  inline float  x()		  const	{ return _x; }
  inline float  y()		  const	{ return _y; }
  inline float  z()		  const	{ return _z; }
  inline float  rho()		  const	{ return _rho; }
  inline float 	phi() 		  const	{ return _phi; }
  inline float 	theta() 	  const	{ return _theta; }
  inline GlobalVector position()  const { return GlobalVector(_x,_y,_z); } //Ignazio
  inline GlobalVector direction() const { return _direction; }             //Ignazio 
  inline int    layer()		  const	{ return _id.layer(); } 
  inline bool   valid()           const { return _valid; }                 //Ignazio
  inline bool PTflag()            const { return _PTflag;}
  inline int MCid()               const { return _MCid;}

 private:
  StackedTrackerDetId _id;
  float _phi, _theta;
  float _x, _y, _z, _rho;
  bool _valid;
  GlobalVector _direction;                                                 // Ignazio  
  bool _PTflag;
  int _MCid;
};


struct lt_stub 
{
  bool operator()(const TrackerStub* a, const TrackerStub* b) const {
    return ( (a->id()).rawId() < (b->id()).rawId() );
  }
};


typedef std::set<TrackerStub*, lt_stub> DTMatchingStubSet; //  StubTracklet;

#endif

