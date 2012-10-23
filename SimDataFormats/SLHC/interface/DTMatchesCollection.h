//---------------------------------------------------------------------------
//
//   \class DTStubCollection
/**
 *   Description:  collection to store DTMatch  triggers and Tracker Stubs 
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

#include "SimDataFormats/SLHC/interface/DTMatch.h"
#include "SimDataFormats/SLHC/interface/DTTrackerStub.h"
#include "SimDataFormats/SLHC/interface/DTTrackerTracklet.h"
#include "SimDataFormats/SLHC/interface/DTTrackerTrack.h"
#include "SimDataFormats/SLHC/interface/DTBtiTrigger.h"
#include "SimDataFormats/SLHC/interface/DTTSPhiTrigger.h"
#include "L1Trigger/DTTriggerServerPhi/interface/DTChambPhSegm.h"
#include "L1Trigger/DTTriggerServerTheta/interface/DTChambThSegm.h"

using namespace std;

//              ---------------------
//              -- Class Interface --
//              ---------------------


typedef int DTmatchIdx;



class DTMatchesCollection {
  /*
    The main data member are:
    vector<DTMatch*> _dtmatches, that is the set of those DT triggers
    satisfying internal criteria of quality and correlation;
    vector<TrackerStub*> _stubs, the set of all stubs;
    vector<TrackerTracklet*> _tracklets, the set of all tracklets.
    Once this data is collected by the DTL1SimOperations methods
    getDTSimTrigger, getTrackerGlobalStubs and getTrackerTracklets,
    the other DTL1SimOperations method getDTPrimitivesToTrackerObjectsMatches
    assigns to each DTMatch object the appropriately best matching 
    TrackerStub and TrackerTracklet objects.
   */
 public:
  // default constructor
  DTMatchesCollection() 	
    { 
      _dtmatches.reserve(10); 
      _stubs.reserve(10); 
      _tracklets.reserve(10); 
      _tracks.reserve(10); 
      _dtmatches_st1=0; // to be safe!
      _dtmatches_st2=0; // to be safe!
      return; 
    }

  // destructor
  ~DTMatchesCollection() 	
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

  inline DTMatch* dtmatch(int i) const { return _dtmatches[i]; }

  inline TrackerStub* stub(int i)    const { return _stubs[i]; }
  inline size_t numStubs()	     const { return _stubs.size(); }
  
  inline TrackerTracklet* tracklet(int i)    const { return _tracklets[i]; }
  inline size_t numTracklets()	     const { return _tracklets.size(); }
  
  inline TrackerTrack* track(int i)    const { return _tracks[i]; }
  inline size_t numTracks()	     const { return _tracks.size(); }


  // utility functions
  inline void addDT(DTMatch* dtmatch) { 
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
  inline void addTracklet(TrackerTracklet* tracklet) { 
    _tracklets.push_back(tracklet); return; 
  }
  inline void addTrack(TrackerTrack* track) { 
    _tracks.push_back(track); return; 
  }

  inline void clear() { 
    _dtmatches_st1=0; 
    _dtmatches_st2=0; 
    _dtmatches.clear(); 
    _stubs.clear(); 
    _tracklets.clear(); 
    _tracks.clear(); 
    return; 
  }

  int nstubsInWindow(int phi, int theta, int sdtphi, int sdttheta, int lay) const;   
  int ntrackletsInWindow(int phi, int theta, int sdtphi, int sdttheta, int lay) const;   
  int ntracksInWindow(int phi, int theta, int sdtphi, int sdttheta) const; 
  void getAllStubsInWindow(int phi, int theta, int sdtphi, int sdttheta, int lay) const; 
  TrackerStub* 
    getClosestStub(int phi, int theta, int sdtphi, int sdttheta, int lay) const; 
  TrackerStub* getClosestPhiStub(int phi, int lay) const;
  TrackerStub* getClosestThetaStub(int theta, int lay) const;
  TrackerStub* getStub(int lay) const;
  int countStubs(int lay) const;  
  TrackerTracklet* 
    getClosestTracklet(int phi, int theta, int sdtphi, int sdttheta, int superlay) const;
  void getAllTracksInWindow(int phi, int theta, int sdtphi, int sdttheta, vector<TrackerTrack*>& Tracks_in_window, int ntracks) const; 
  void orderDTTriggers();
  void extrapolateDTToTracker();
  void removeRedundantDTMatch();
  void eraseDTMatch(int dm);

 private:
  // DT Phi-Theta Match then completed with TrackerStub matches
  // TrackerTracklet matches added (PLZ: 110608)
  vector<DTMatch*> _dtmatches;

  // Tracker Stub vector
  vector<TrackerStub*> _stubs;
  // Tracker Tracklet vector
  vector<TrackerTracklet*> _tracklets;
  // Tracker Track vector
  vector<TrackerTrack*> _tracks;
 
  // record number of DTMatch per station
  int _dtmatches_st1;
  int _dtmatches_st2;

};



#endif

