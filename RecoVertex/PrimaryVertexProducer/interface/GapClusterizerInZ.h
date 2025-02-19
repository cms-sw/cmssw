#ifndef GapClusterizerInZ_h
#define GapClusterizerInZ_h

/**\class GapClusterizerInZ
 
  Description: separates event tracks into clusters along the beam line

*/
#include "RecoVertex/PrimaryVertexProducer/interface/TrackClusterizerInZ.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"




class GapClusterizerInZ : public TrackClusterizerInZ {


public:
  GapClusterizerInZ(const edm::ParameterSet& conf);

  std::vector< std::vector<reco::TransientTrack> >
    clusterize(const std::vector<reco::TransientTrack> & tracks)const;

  float zSeparation() const;

  std::vector< TransientVertex >
    vertices(const std::vector<reco::TransientTrack> & tracks)const;

  ~GapClusterizerInZ(){};
  
private:
  float zSep;
  bool verbose_;

};

#endif
