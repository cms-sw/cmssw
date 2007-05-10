#ifndef TrackFilterForPVFinding_h
#define TrackFilterForPVFinding_h

/**\class TrackFilterForPVFinding 
 
  Description: selects tracks for primary vertex reconstruction

*/

#include "RecoVertex/PrimaryVertexProducer/interface/BeamTransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class TrackFilterForPVFinding {

public:

  TrackFilterForPVFinding(const edm::ParameterSet& conf);

  bool operator() (const reco::TransientTrack & tk) const;
  bool operator() (const BeamTransientTrack & tk) const;

  float minPt() const;
  float maxD0Significance() const;

private:

  edm::ParameterSet theConfig;

};

#endif
