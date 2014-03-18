#ifndef GEMValidation_SimTrackMatchManager_h
#define GEMValidation_SimTrackMatchManager_h

/**\class SimTrackMatchManager

 Description: Matching of SIM and Trigger info for a SimTrack in CSC & GEM

 It's a manager-matcher class, as it uses specialized matching classes to match SimHits, various digis and stubs.

 Original Author:  "Vadim Khotilovich"
 $Id: SimTrackMatchManager.h,v 1.1 2013/02/11 07:33:07 khotilov Exp $

*/

#include "Validation/MuonGEMHits/interface/BaseMatcher.h"
#include "Validation/MuonGEMHits/interface/SimHitMatcher.h"
#include "Validation/MuonGEMHits/interface/GEMDigiMatcher.h"

class SimTrackMatchManager
{
public:
  
  SimTrackMatchManager(const SimTrack& t, const SimVertex& v,
      const edm::ParameterSet& ps, const edm::Event& ev, const edm::EventSetup& es, const GEMGeometry* gem_geo);
  
  ~SimTrackMatchManager();

  const SimHitMatcher& simhits() const {return simhits_;}
  const GEMDigiMatcher& gemDigis() const {return gem_digis_;}
  
private:

  SimHitMatcher simhits_;
  GEMDigiMatcher gem_digis_;
};

#endif
