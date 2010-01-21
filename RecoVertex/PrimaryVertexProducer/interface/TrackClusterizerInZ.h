#ifndef TrackClusterizerInZ_h
#define TrackClusterizerInZ_h

/**\class TrackClusterizerInZ 
 
  Description: separates event tracks into clusters along the beam line

*/

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>


class TrackClusterizerInZ {

public:

  TrackClusterizerInZ(const edm::ParameterSet& conf);

  std::vector< std::vector<reco::TransientTrack> >
  clusterize(const std::vector<reco::TransientTrack> & tracks) const;

  float zSeparation() const;

private:

  float zSep;
};

#endif
