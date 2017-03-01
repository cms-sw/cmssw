#ifndef SimpleVertexTree_H
#define SimpleVertexTree_H

#include <string>

#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/KalmanVertexFit/interface/VertexFitterResult.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "TString.h"

/**
 * Basic class to do vertex fitting and smothing studies.<br>
 * For vertex resolution studies, it produces a TTree and a few basic histograms.<br>
 * The TTree contains only the positions of the simulated and reconstructed 
 * vertices, total chi**2, chi**2 probability and number of degrees of freedom.
 * The histograms present the residuals and pulls along the three axis, the
 * nomalized chi**2 and the chi**2 probability.
 * The only thing to be done is to call the method fill for each vertex.<br>
 *
 * WARNING: there is no track info in the tree yet! so what follows is not yet true! <br>
 * For smoothing studies (track refit with vertex constraint after the vertex
 * fit per se), the TTree is expanded with the track paramater info. 
 * For each vertex, for each matched track, the simulated, reconstructed (before
 * smoothing), and refitted (after smoothing) parameters and errors are included
 * in the TTree.
 * This information is provided only if the tracks have been smoothed by the
 * vertex fitter, and if the SimpleConfigurable <i>SimpleVertexTree:trackTest</i>
 * is set to true.
 * No statistics will be printed for the track parameters at the end of the job.
 *
 * A simpe root analysis is given in the test directory (simpleVertexAnalysis)
 * to produce vertex and track parameter resolution and error plots.
 * It is described in more details in the userguide.
 */

class TFile;
class TTree;

class SimpleVertexTree {
public:

  /**
   * The constructor<br>
   * \param fitterName The name of the TTree, and of the associated histograms. 
   */

  SimpleVertexTree(const char * fitterName = "VertexFitter",
  		   const MagneticField * magField = 0);
  virtual ~SimpleVertexTree();

  /**
   * Entry for a RecVertex. If the vertex was not associated to a TkSimVertex,
   * an empty pointer can be given (would be identical to the next method).
   * Timing information for the fit can also be provided.
   */

  void fill(const TransientVertex & recv, const TrackingVertex *simv = 0, 
  	    reco::RecoToSimCollection *recSimColl = 0,
  	    const float &time = 0.);

  void fill(const TransientVertex & recv, const TrackingVertex *simv = 0,
  	    const float &time = 0.);

  /**
   * Entry for a RecVertex, without associated vertex.
   * Timing information for the fit can also be provided.
   */

  void fill(const TransientVertex & recv, const float &time = 0.);

  /**
   * Entry for a TkSimVertex, without RecVertex.
   */

  void fill(const TrackingVertex *simv);

//   void fill(const TransientVertex & recVertex, const std::vector < RecTrack > & recTrackV,
// 			const SimVertex * simv, const float &time);
// 
//   void fill(const std::vector < RecTrack > & recTrackV, const TkSimVertex * simv = 0, 
//   			const float &time = 0.);

  /**
   * To be used if one wants to record "Failed Fits", e.g. to synchronise two Trees
   */
  void fill();

private:

  void defineTrackBranch(const TString& prefix, const TString& type,
			const float* (VertexFitterResult::*pfunc)(const int) const,
			const TString& index);

  float simPos[3];
  float recPos[3];
  float recErr[3];
  float chiTot, ndf, chiProb;
  int numberOfVertices;
  TTree* vertexTree;
  VertexFitterResult* result;
  TString theFitterName;

  bool trackTest;
  int maxTrack;
  TString* parameterNames[5];
};
#endif

