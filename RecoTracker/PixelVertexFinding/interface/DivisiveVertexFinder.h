#ifndef RecoTracker_PixelVertexFinding_DivisiveVertexFinder_h
#define RecoTracker_PixelVertexFinding_DivisiveVertexFinder_h
/** \class DivisiveVertexFinder DivisiveVertexFinder.h RecoTracker/PixelVertexFinding/interface/DivisiveVertexFinder.h 

 Description: Fits a primary vertex in 1D (z) using the "divisive method"

 Implementation: 
 This class was ported from ORCA by me (Aaron).  It was originally written by ...
 Find the PV candidates with a simple divisive method.
 Divide the luminosity region in several regions according 
 to the track distance and for each of them make a PVCluster. 
 Iteratively discard tracks and recover them in a new PVCluster.
 Return a sorted vector<Vertex> (aka VertexCollection) with the z coordinate of PV candidates
 \param ntkmin Minimum number of tracks required to form a cluster.
 \param useError physical distances or weighted distances.
 \param zsep Maximum distance between two adjacent tracks that belong
 to the same initial cluster.
 \param wei Compute the cluster "center" with an unweighted or a weighted
 average of the tracks. Weighted means weighted with the error
 of the data point.

 \author Aaron Dominguez (UNL)
*/
#include <vector>
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
//#include "CommonTools/Clustering1D/interface/DivisiveClusterizer1D.h"
#include "RecoTracker/PixelVertexFinding/interface/DivisiveClusterizer1D.h"

#include "RecoTracker/PixelVertexFinding/interface/PVClusterComparer.h"

class DivisiveVertexFinder {
public:
  DivisiveVertexFinder(double track_pt_min,
                       double track_pt_max,
                       double track_chi2_max,
                       double track_prob_min,
                       double zOffset = 5.0,
                       int ntrkMin = 5,
                       bool useError = true,
                       double zSeparation = 0.05,
                       bool wtAverage = true,
                       int verbosity = 0);
  ~DivisiveVertexFinder();

  /// Run the divisive algorithm and return a vector of vertexes for the input track collection
  bool findVertexes(const reco::TrackRefVector &trks,     // input
                    reco::VertexCollection &vertexes);    // output
  bool findVertexesAlt(const reco::TrackRefVector &trks,  // input
                       reco::VertexCollection &vertexes,
                       const math::XYZPoint &bs);  // output
private:
  /// Cuts on vertex formation and other options
  double zOffset_, zSeparation_;
  int ntrkMin_;
  bool useError_, wtAverage_;

  /// We use Wolfgang's templated class that implements the actual divisive method
  pixeltemp::DivisiveClusterizer1D<reco::Track> divmeth_;
  //  DivisiveClusterizer1D< reco::Track > divmeth_;

  // How loud should I be?
  int verbose_;

  PVClusterComparer *pvComparer_;
};
#endif
