#ifndef TrackClusterizerInZ_h
#define TrackClusterizerInZ_h

/**\class TrackClusterizerInZ 
 
  Description: separates event tracks into clusters along the beam line

*/

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>
#include "DataFormats/Math/interface/Error.h"


struct track_t{
  double z;
  double dz2;
  double ip,dip;  // test
  const reco::TransientTrack* tt;
  double Z;           // Z[i]
};

struct vertex_t{
  double x;    // y               x coordinate
  double y;    // y               y coordinate
  double z;    // z               z coordinate
  double py;   // py              vertex weight
  double *ptk;   // p[i][k]         assignment probability track to vertex
  double Tc;   //   current critical temperature
  double epsilon;
  double arg;   // temporary 
};


class TrackClusterizerInZ {


public:

  TrackClusterizerInZ(const edm::ParameterSet& conf);

  std::vector< std::vector<reco::TransientTrack> >
    clusterize(const std::vector<reco::TransientTrack> & tracks)const;
  std::vector< std::vector<reco::TransientTrack> >
    clusterize0(const std::vector<reco::TransientTrack> & tracks)const;
  float zSeparation() const;

  std::vector< std::vector<reco::TransientTrack> >
    clusterizeDA(const std::vector<reco::TransientTrack> & tracks)const;
  std::vector<track_t> fill(const std::vector<reco::TransientTrack> & tracks)const;
  void updateWeights(
		     double beta,
		     std::vector<track_t> & tks,
		     std::vector<vertex_t> & y
		     )const;
  double fit(double beta, std::vector<track_t> & tks, std::vector<vertex_t> & y)const;
  void dump(const double beta, std::vector<vertex_t> & y, const std::vector<track_t> & tks, const int verbosity=0)const;
  bool merge(std::vector<vertex_t> &)const;
  double split(
	       double beta,
	       std::vector<track_t> & tks,
	       std::vector<vertex_t> & y
	       )const;
  
private:
  float zSep;
  bool verbose_;
  bool DEBUG;
  int maxIterations_;
  float betamax_;
};

#endif
