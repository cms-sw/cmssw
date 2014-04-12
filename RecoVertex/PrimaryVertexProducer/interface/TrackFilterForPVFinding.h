#ifndef TrackFilterForPVFinding_h
#define TrackFilterForPVFinding_h

/**\class TrackFilterForPVFinding 
 
  Description: track selection for PV finding

*/
#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFindingBase.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>

class TrackFilterForPVFinding : public TrackFilterForPVFindingBase  {

public:

  TrackFilterForPVFinding(const edm::ParameterSet& conf);
  bool operator() (const reco::TransientTrack & tracks)const;
  std::vector<reco::TransientTrack> select (const std::vector<reco::TransientTrack>& tracks)const;

private:

  float maxD0Sig_, minPt_;
  int minSiLayers_, minPxLayers_;
  float maxNormChi2_;
  reco::TrackBase::TrackQuality quality_;
};

#endif
