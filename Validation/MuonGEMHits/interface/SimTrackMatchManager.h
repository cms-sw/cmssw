#ifndef GEMValidation_SimTrackMatchManager_h
#define GEMValidation_SimTrackMatchManager_h

/**\class SimTrackMatchManager

 Description: Matching of SIM and Trigger info for a SimTrack in CSC & GEM

 It's a manager-matcher class, as it uses specialized matching classes to match SimHits, various digis and stubs.

 Original Author:  "Vadim Khotilovich"
 $Id: SimTrackMatchManager.h,v 1.1 2013/02/11 07:33:07 khotilov Exp $

*/

#include "Validation/MuonGEMDigis/interface/BaseMatcher.h"
#include "Validation/MuonGEMDigis/interface/SimHitMatcher.h"

class SimTrackMatchManager
{
public:
  
  SimTrackMatchManager(const SimTrack& t, const SimVertex& v,
      const edm::ParameterSet& ps, const edm::Event& ev, const edm::EventSetup& es);
  
  ~SimTrackMatchManager();

  const SimHitMatcher& simhits() const {return simhits_;}
  
private:

  SimHitMatcher simhits_;
};

#endif
