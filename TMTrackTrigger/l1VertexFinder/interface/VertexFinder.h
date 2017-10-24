#ifndef __TMTrackTrigger_VertexFinder_VertexFinder_h__
#define __TMTrackTrigger_VertexFinder_VertexFinder_h__

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "TMTrackTrigger/VertexFinder/interface/L1fittedTrack.h"
#include "TMTrackTrigger/VertexFinder/interface/RecoVertex.h"

#include <vector>


namespace vertexFinder {

typedef std::vector<const L1fittedTrack*> FitTrackCollection;
typedef std::vector<RecoVertex> RecoVertexCollection;

class VertexFinder {

public:
  // Copy fitted tracks collection into class
  VertexFinder(FitTrackCollection fitTracks, Settings* settings){fitTracks_ = fitTracks; settings_ = settings;}
  ~VertexFinder(){}

  struct SortTracksByZ0{
    inline bool operator() (const L1fittedTrack* track0, const L1fittedTrack* track1){
      return(track0->z0() < track1->z0());
    }
  };

  struct SortTracksByPt{
    inline bool operator() (const L1fittedTrack* track0, const L1fittedTrack* track1){
      return(fabs(track0->pt()) > fabs(track1->pt()));
    }
  };

  struct SortVertexByZ0{
    inline bool operator() (const RecoVertex vertex0, const RecoVertex vertex1){
      return(vertex0.z0() < vertex1.z0());
    }
  };

  /// Returns the z positions of the reconstructed primary vertices
  std::vector<RecoVertex> Vertices()        const {return vertices_;}
  /// Number of reconstructed vertices
  unsigned int numVertices()                const {return vertices_.size();}
  /// Reconstructed Primary Vertex
  RecoVertex  PrimaryVertex()               const {if(pv_index_ < vertices_.size()) return vertices_[pv_index_]; else{ std::cout << "No Primary Vertex has been found."; return RecoVertex();} }
  /// Reconstructed Primary Vertex as in TDR
  RecoVertex  TDRPrimaryVertex()            const {return tdr_vertex_;}
  /// Reconstructed Primary Vertex Id
  unsigned int PrimaryVertexId()            const { return pv_index_;}
  /// Storage for tracks out of the L1 Track finder
  FitTrackCollection FitTracks()            const { return fitTracks_;}
  /// Storage for tracks out of the L1 Track finder
  FitTrackCollection TDRPileUpTracks()      const { return tdr_pileup_tracks_;}
  /// Find the primary vertex
  void  FindPrimaryVertex();
  /// Gap Clustering Algorithm
  void GapClustering();
  /// Find maximum distance in two clusters of tracks
  float MaxDistance(RecoVertex cluster0, RecoVertex cluster1);
  /// Simple Merge Algorithm
  void SimpleMergeClustering();
  /// DBSCAN algorithm
  void DBSCAN();
  /// Principal Vertex Reconstructor algorithm
  void PVR();
  /// Adaptive Vertex Reconstruction algorithm 
  void AdaptiveVertexReconstruction();
  /// High PT Vertex Algorithm
  void HPV();
  /// Histogramming algorithmn as in the TDR
  void TDRalgorithm();
  /// Sort Vertices in z
  void SortVerticesInZ0()                   {std::sort(vertices_.begin(), vertices_.end(), SortVertexByZ0());}
  /// Number of iterations
  unsigned int NumIterations()				const{ return iterations_;}
  /// Number of iterations
  unsigned int IterationsPerTrack()			const{ return double(iterations_)/double(fitTracks_.size());}

private:

  Settings* settings_;
  std::vector<RecoVertex> vertices_;
  unsigned int numMatchedVertices_;
  FitTrackCollection fitTracks_;
  unsigned int pv_index_;
  unsigned int iterations_;

  RecoVertex tdr_vertex_;
  FitTrackCollection tdr_pileup_tracks_;

};

} // end namespace vertexFinder

#endif
