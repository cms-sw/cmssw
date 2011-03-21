//---------------------------------------------------------------------------
//
//   \class DTStubCollection
/**
 *   Description:  collection to store DTStubMatch  triggers and Tracker Stubs 
*/
//   090320 
//   Sara Vanini - Padua University
//   from June 2009 on: Ignazio Lazzizzera - Trento University
//-----------------------------------------------------------------------------
#ifndef DTStubCollection_H
#define DTStubCollection_H

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <map>
#include <algorithm>
#include <vector>

#include "SimDataFormats/SLHC/interface/DTStubMatch.h"
#include "SimDataFormats/SLHC/interface/DTTrackerStub.h"
#include "SimDataFormats/SLHC/interface/DTBtiTrigger.h"
#include "SimDataFormats/SLHC/interface/DTTSPhiTrigger.h"
#include "L1Trigger/DTTriggerServerPhi/interface/DTChambPhSegm.h"
#include "L1Trigger/DTTriggerServerTheta/interface/DTChambThSegm.h"

using namespace std;

//              ---------------------
//              -- Class Interface --
//              ---------------------


typedef int DTmatchIdx;



class DTStubMatchesCollection {

 public:
  // default constructor
  DTStubMatchesCollection() 	
    { 
      _dtmatches.reserve(10); 
      _stubs.reserve(10); 
      _dtmatches_st1=0; // to be save!
      _dtmatches_st2=0; // to be save!
      return; 
    }

  // destructor
  ~DTStubMatchesCollection() 	
    { 
      clear(); 
      return; 
    } 
  
  //return functions
  inline int numDt() const { return _dtmatches.size(); }
  inline int numDt(int station)	const { 
    if(station==1) return _dtmatches_st1; 
    else if(station==2) return _dtmatches_st2; 
    return 0;
  }

  inline DTStubMatch* dtmatch(int i) const { return _dtmatches[i]; }

  inline TrackerStub* stub(int i)    const { return _stubs[i]; }
  inline size_t numStubs()	     const { return _stubs.size(); }

  // utility functions
  inline void addDT(DTStubMatch* dtmatch) { 
    _dtmatches.push_back(dtmatch); 
    if(dtmatch->station()==1) _dtmatches_st1++; 
    if(dtmatch->station()==2) _dtmatches_st2++;
    return; 
  }

  // Ignazio:
  void addDT(DTBtiTrigger const bti, 
	     DTChambPhSegm const tsphi, 
	     bool debug_dttrackmatch = false);

  void addDT(const DTBtiTrigger& bti, 
	     const DTTSPhiTrigger& tsphi,
	     bool debug_dttrackmatch);
  /*
  void addDT(const DTChambThSegm& tstheta, 
	     const DTTSPhiTrigger& tsphi,
	     const DTChamber* chamb,
	     bool debug_dttrackmatch);
  */
  // end Ignazio


  inline void addStub(TrackerStub* stub) { _stubs.push_back(stub); return; }


  inline void clear() { 
    _dtmatches_st1=0; 
    _dtmatches_st2=0; 
    _dtmatches.clear(); 
    _stubs.clear(); 
    return; 
  }

  int nstubsInWindow(int phi, int theta, int sdtphi, int sdttheta, int lay) const; 
  void getAllStubsInWindow(int phi, int theta, int sdtphi, int sdttheta, int lay) const; 
  TrackerStub* getClosestStub(int phi, int theta, int sdtphi, int sdttheta, int lay) const; 
  TrackerStub* getClosestPhiStub(int phi, int lay) const;
  TrackerStub* getClosestThetaStub(int theta, int lay) const;
  TrackerStub* getStub(int lay) const;
  int  countStubs(int lay) const;
  void orderDTTriggers();
  void extrapolateDTToTracker();
  void removeRedundantDTStubMatch();
  void eraseDTStubMatch(int dm);

 private:
  // DT Phi-Theta Match then completed with TrackerStub matches
  vector<DTStubMatch*> _dtmatches;

  // Tracker Stub vector
  vector<TrackerStub*> _stubs;
 
  // record number of DTStubMatch per station
  int _dtmatches_st1;
  int _dtmatches_st2;

};



#endif

