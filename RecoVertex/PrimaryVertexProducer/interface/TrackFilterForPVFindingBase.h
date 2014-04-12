#ifndef TrackFilterForPVFindingBase_h
#define TrackFilterForPVFindingBase_h

/**\class TrackFilterForPVFindingBase
 
  Description: base class for track selection

*/

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>

class TrackFilterForPVFindingBase {

public:

  TrackFilterForPVFindingBase(){};
  TrackFilterForPVFindingBase(const edm::ParameterSet& conf){};
  virtual std::vector<reco::TransientTrack> select (const std::vector<reco::TransientTrack>& tracks)const=0;
  virtual ~TrackFilterForPVFindingBase(){};
};

#endif
