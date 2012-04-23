#ifndef DAClusterizerInZ_h
#define DAClusterizerInZ_h

/**\class DAClusterizerInZ 
 
  Description: separates event tracks into clusters along the beam line

*/

#include "RecoVertex/PrimaryVertexProducer/interface/TrackClusterizerInZ.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>
#include "DataFormats/Math/interface/Error.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"




class DAClusterizerInZ : public TrackClusterizerInZ {


public:

struct track_t{
  double z;              // z-coordinate at point of closest approach to the beamline
  double dz2;            // square of the error of z(pca)
  const reco::TransientTrack* tt;  // a pointer to the Transient Track
  double Z;              // Z[i]   for DA clustering
  double pi;             // track weight
};


struct vertex_t{
  double z;    //           z coordinate
  double pk;   //           vertex weight for "constrained" clustering
  // --- temporary numbers, used during update
  double ei;
  double sw;
  double swz;
  double se;
};




 DAClusterizerInZ(const edm::ParameterSet& conf);

  std::vector< std::vector<reco::TransientTrack> >
    clusterize(const std::vector<reco::TransientTrack> & tracks)const;


  std::vector< TransientVertex >
    vertices(const std::vector<reco::TransientTrack> & tracks, const int verbosity=0)const;


  std::vector<track_t> fill(const std::vector<reco::TransientTrack> & tracks)const;

  double update(
		     double beta,
		     std::vector<track_t> & tks,
		     std::vector<vertex_t> & y
		     )const;

  double update(
		     double beta,
		     std::vector<track_t> & tks,
		     std::vector<vertex_t> & y,
		     double &
		     )const;

  void dump(const double beta, const std::vector<vertex_t> & y, const std::vector<track_t> & tks, const int verbosity=0)const;
  bool merge(std::vector<vertex_t> &,int )const;
  bool purge(std::vector<vertex_t> &, std::vector<track_t> & , double &, const double )const;

  void splitAll(
	       std::vector<track_t> & tks,
	       std::vector<vertex_t> & y
	       )const;

  double beta0(
	       const double betamax,
	       std::vector<track_t> & tks,
	       std::vector<vertex_t> & y
	       )const;

  double Eik(const track_t & t, const vertex_t & k)const;

  
private:
  bool verbose_;
  float vertexSize_;
  int maxIterations_;
  double coolingFactor_;
  float betamax_;
  float betastop_;
  double dzCutOff_;
  double d0CutOff_;
};

#endif
