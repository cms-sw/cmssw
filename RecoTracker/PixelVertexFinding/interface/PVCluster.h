#ifndef RecoTracker_PixelVertexFinding_PVCluster_h
#define RecoTracker_PixelVertexFinding_PVCluster_h
/** \class PVCluster PVCluster.h RecoTracker/PixelVertexFinding/PVCluster.h 
 * A simple collection of tracks that represents a physical clustering
 * of charged particles, ie a vertex, in one dimension.  This
 * (typedef) class is used by the Pixel standalone vertex finding
 * classes found in RecoTracker/PixelVertexFinding.
 *
 *  \author Aaron Dominguez (UNL)
 */
#include "CommonTools/Clustering1D/interface/Cluster1D.h"
#include "DataFormats/TrackReco/interface/Track.h"

typedef Cluster1D<reco::Track> PVCluster;

#endif
