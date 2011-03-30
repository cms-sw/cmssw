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
  double npi;            // normalized track weight
};


struct vertex_t{
  double z;    //           z coordinate
  double pk;   //           vertex weight for "constrained" clustering
  std::vector<double> pik;//
  bool   update;
  double z1,pk1;
  // --- temporary numbers, used during update
  double Ei;       // =log (pik) = (z_i - z_k)**2
  double pi;       // pk*exp(-beta Ei).
  //  double ei;       // exponential exp(-beta Ei).... obsolete
  double sw;
  double swz;
  double se;
  double logpk;
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
		     std::vector<vertex_t> & y,
		     double Z0=0
		     )const;
  double update1(
		     double beta,
		     const bool forceUpdate,
		     std::vector<track_t> & tks,
		     std::vector<vertex_t> & y,
		     double Z0=0
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

  inline double Eik(const track_t & t, const vertex_t & k)const{double dz=t.z-k.z; return dz*dz/t.dz2;};

  //   inline double fexp(const double  z) const{// (z is <=0 
  //     return (-z)<ecutoff_ ? 1.+2.*z/(2.-z+z*z/6.) : 0.;
  //   } 
  
private:
  bool verbose_;
  float vertexSize_;
  int maxIterations_;
  double coolingFactor_;
  float betamax_;
  float betastop_;
  double mu0_;
  double deltamax_;
  bool splitMergedClusters_;
  bool mergeAfterAnnealing_;
  bool useTrackResolutionAfterFreezeOut_;
  bool full_;
};

#endif
