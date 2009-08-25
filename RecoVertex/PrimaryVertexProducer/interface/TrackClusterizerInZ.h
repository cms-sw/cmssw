#ifndef TrackClusterizerInZ_h
#define TrackClusterizerInZ_h

/**\class TrackClusterizerInZ 
 
  Description: separates event tracks into clusters along the beam line

*/

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>
#include "DataFormats/Math/interface/Error.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"




class TrackClusterizerInZ {


public:

struct track_t{
  double z;              // z at pca
  double dz2;            // square of the error of z(pca)
  //double w;            // 
  double ip,dip,pt,x,y;  // test
  int clu;               // cluster assignment (for modified gap clustering)
  const reco::TransientTrack* tt;  // a pointer to the Transient Track
  double Z;              // Z[i]   for DA clustering
};

struct vertex_t{
  double x;    // y               x coordinate
  double y;    // y               y coordinate
  double z;    // z               z coordinate
  double py;   // py              vertex weight
  double *ptk;   // p[i][k]         assignment probability track to vertex
  double Tc;   //   current critical temperature
  double epsilon;
};



  TrackClusterizerInZ();
  TrackClusterizerInZ(const edm::ParameterSet& conf);

  std::vector< std::vector<reco::TransientTrack> >
    clusterize(const std::vector<reco::TransientTrack> & tracks)const;
  std::vector< std::vector<reco::TransientTrack> >
    clusterize0(const std::vector<reco::TransientTrack> & tracks)const;
  std::vector< std::vector<reco::TransientTrack> >
    clusterize1(const std::vector<reco::TransientTrack> & tracks)const;

  float zSeparation() const;

/*   std::vector< std::vector<reco::TransientTrack> > */
/*     clusterizeDA(const std::vector<reco::TransientTrack> & tracks, double zClusterSep=0.2)const; */

  std::vector< TransientVertex >
    vertices(const std::vector<reco::TransientTrack> & tracks, const double Tmin=0)const;

  std::vector<track_t> fill(const std::vector<reco::TransientTrack> & tracks)const;
  void updateWeights(
		     double beta,
		     std::vector<track_t> & tks,
		     std::vector<vertex_t> & y
		     )const;
  double fit(double beta, std::vector<track_t> & tks, std::vector<vertex_t> & y)const;
  void dump(const double beta, const std::vector<vertex_t> & y, const std::vector<track_t> & tks, const int verbosity=0)const;
  bool merge(std::vector<vertex_t> &,int )const;
  double split(
	       double beta,
	       std::vector<track_t> & tks,
	       std::vector<vertex_t> & y
	       )const;

  double beta0(
	       const double betamax,
	       std::vector<track_t> & tks,
	       std::vector<vertex_t> & y
	       )const;
  
private:
  float zSep;
  std::string algorithm;
  bool verbose_;
  bool DEBUG;
  int maxIterations_;
  float betamax_;
  double coolingFactor_;
  VertexDistanceXY theVertexDistance_;

};

#endif
