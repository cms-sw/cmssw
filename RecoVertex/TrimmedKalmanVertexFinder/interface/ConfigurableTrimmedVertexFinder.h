#ifndef _ConfigurableTrimmedVertexFinder_H_
#define _ConfigurableTrimmedVertexFinder_H_

#include "RecoVertex/VertexPrimitives/interface/VertexReconstructor.h"
#include "RecoVertex/TrimmedKalmanVertexFinder/interface/TrimmedVertexFinder.h"
#include "RecoVertex/TrimmedKalmanVertexFinder/interface/TrimmedTrackFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>

/** Algorithm to find a series of distinct vertices among the given set 
 *  of tracks. The technique is: <BR>
 *    1) use `TrimmedTrackFilter` to select tracks 
 *    above a certain pT; <BR>
 *    2) use `TrimmedVertexFinder` to split the set of tracks into 
 *    a cluster of compatible tracks and a set of remaining tracks; <BR>
 *    3) repeat 2) with the remaining set, and repeat as long as 
 *    (a) a cluster of compatible tracks can be found or (b) 
 *    the maximum number of clusters asked for is reached; <BR>
 *    4) reject vertices with a low fit probability. <BR>
 *
 *  This algorithm has 5 parameters that can be set at runtime 
 *  via the corresponding set() methods, or a ParameterSet: <BR>
 *   - "ptCut" (default: 1.5 GeV/c) 
 *  which defines the minimum pT of the tracks used to make vertices. 
 *  This value overrides the corresponding configurable parameter 
 *  of the TrimmedTrackFilter used internally. <BR>
 *   - "trackCompatibilityCut" (default: 0.05) 
 *  which defines the probability below which a track is considered 
 *  incompatible with the 1st vertex candidate formed. 
 *  This value overrides the corresponding configurable parameter 
 *  of the TrimmedVertexFinder used internally. <BR>
 *   - "trackCompatibilityToSV" (default: 0.01) 
 *  which defines the probability below which a track is considered 
 *  incompatible with the next vertex candidates formed. 
 *  This value overrides the corresponding configurable parameter 
 *  of the TrimmedVertexFinder used internally. <BR>
 *   - "vertexFitProbabilityCut" (default: 0.01) 
 *  which defines the probability below which a vertex is rejected. <BR>
 *   - "maxNbOfVertices" (default: 0) 
 *  which defines the maximum number of vertices searched for. 
 *  0 means all vertices which can be found. 
 */

class ConfigurableTrimmedVertexFinder : public VertexReconstructor {
public:
  ConfigurableTrimmedVertexFinder(const VertexFitter<5>* vf,
                                  const VertexUpdator<5>* vu,
                                  const VertexTrackCompatibilityEstimator<5>* ve);

  ~ConfigurableTrimmedVertexFinder() override {}

  std::vector<TransientVertex> vertices(const std::vector<reco::TransientTrack>& tracks) const override;

  std::vector<TransientVertex> vertices(const std::vector<reco::TransientTrack>& tracks,
                                        const reco::BeamSpot& spot) const override;

  std::vector<TransientVertex> vertices(const std::vector<reco::TransientTrack>& tracks,
                                        std::vector<reco::TransientTrack>& unused,
                                        const reco::BeamSpot& spot,
                                        bool use_spot) const;

  /** Access to parameters
   */
  float ptCut() const { return theFilter.ptCut(); }
  const TrimmedTrackFilter& trackFilter() const { return theFilter; }
  float trackCompatibilityCut() const { return theTrackCompatibilityToPV; }
  float trackCompatibilityToSV() const { return theTrackCompatibilityToSV; }
  float vertexFitProbabilityCut() const { return theVtxFitProbCut; }
  int maxNbOfVertices() const { return theMaxNbOfVertices; }

  /** Set parameters
   */
  void setParameters(const edm::ParameterSet&);

  void setPtCut(float cut) { theFilter.setPtCut(cut); }
  void setTrackCompatibilityCut(float cut) { theTrackCompatibilityToPV = cut; }
  void setTrackCompatibilityToSV(float cut) { theTrackCompatibilityToSV = cut; }
  void setVertexFitProbabilityCut(float cut) { theVtxFitProbCut = cut; }
  void setMaxNbOfVertices(int max) { theMaxNbOfVertices = max; }

  /** Clone method
   */
  ConfigurableTrimmedVertexFinder* clone() const override { return new ConfigurableTrimmedVertexFinder(*this); }

protected:
  virtual void resetEvent(const std::vector<reco::TransientTrack>& tracks) const {}

  virtual void analyseInputTracks(const std::vector<reco::TransientTrack>& tracks) const {}

  virtual void analyseClusterFinder(const std::vector<TransientVertex>& vts,
                                    const std::vector<reco::TransientTrack>& remain) const {}

  virtual void analyseVertexCandidates(const std::vector<TransientVertex>& vts) const {}

  virtual void analyseFoundVertices(const std::vector<TransientVertex>& vts) const {}

private:
  using VertexReconstructor::vertices;

  // separate tracks passing the filter
  //  void separateTracks(std::vector<TransientTrack>& filtered,
  //		      std::vector<TransientTrack>& unused) const;

  // find vertex candidates
  std::vector<TransientVertex> vertexCandidates(const std::vector<reco::TransientTrack>& tracks,
                                                std::vector<reco::TransientTrack>& unused,
                                                const reco::BeamSpot& spot,
                                                bool use_spot) const;

  // remove bad candidates
  std::vector<TransientVertex> clean(const std::vector<TransientVertex>& candidates) const;

  //
  mutable TrimmedVertexFinder theClusterFinder;
  float theVtxFitProbCut;
  float theTrackCompatibilityToPV;
  float theTrackCompatibilityToSV;
  int theMaxNbOfVertices;
  TrimmedTrackFilter theFilter;
};

#endif
