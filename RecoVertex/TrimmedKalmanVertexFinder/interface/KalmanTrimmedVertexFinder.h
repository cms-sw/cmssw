#ifndef _KalmanTrimmedVertexFinder_H_
#define _KalmanTrimmedVertexFinder_H_

#include "RecoVertex/TrimmedKalmanVertexFinder/interface/ConfigurableTrimmedVertexFinder.h"

/** User-friendly wrapper around ConfigurableTrimmedVertexFinder. <BR>
 *  Chooses the KalmanVertexFit classes as vertex fitting classes 
 *  used by the TrimmedVertexFinder. <BR>
 *  KalmanTrimmedVertexFinder is configurable 
 *  using the same set() methods as ConfigurableTrimmedVertexFinder. 
 */

class KalmanTrimmedVertexFinder 
  : public VertexReconstructor {

public:

  KalmanTrimmedVertexFinder();
  KalmanTrimmedVertexFinder(
    const KalmanTrimmedVertexFinder & other);
  virtual ~KalmanTrimmedVertexFinder();

  /** Clone method
   */
  virtual KalmanTrimmedVertexFinder * clone() const {
    return new KalmanTrimmedVertexFinder(*this);
  }

  virtual inline std::vector<TransientVertex> 
    vertices(const std::vector<reco::TransientTrack> & tracks) const { 
    return theFinder->vertices(tracks); 
  }
  
  virtual inline std::vector<TransientVertex> 
    vertices(const std::vector<reco::TransientTrack> & tracks,
        const reco::BeamSpot & s ) const { 
    return theFinder->vertices(tracks,s); 
  }

  inline std::vector<TransientVertex> 
    vertices( const std::vector<reco::TransientTrack> & tracks, 
	      std::vector<reco::TransientTrack>& unused) const {
    return theFinder->vertices(tracks, unused, reco::BeamSpot(), false );
  }
  
  inline std::vector<TransientVertex> 
    vertices( const std::vector<reco::TransientTrack> & tracks, 
	      std::vector<reco::TransientTrack>& unused,
        const reco::BeamSpot & spot, bool usespot=false ) const {
    return theFinder->vertices(tracks, unused, spot, usespot );
  }

  /** Access to parameters
   */
  inline float ptCut() const { return theFinder->ptCut(); }
  inline float trackCompatibilityCut() const { 
    return theFinder->trackCompatibilityCut();
  }
  inline float trackCompatibilityToSV() const { 
    return theFinder->trackCompatibilityToSV();
  }
  inline float vertexFitProbabilityCut() const { 
    return theFinder->vertexFitProbabilityCut();
  }
  inline int maxNbOfVertices() const { return theFinder->maxNbOfVertices(); }

  /** Set parameters
   */
   
  void setParameters ( const edm::ParameterSet & );   
   
  inline void setPtCut(float cut) { theFinder->setPtCut(cut); }
  inline void setTrackCompatibilityCut(float cut) {
    theFinder->setTrackCompatibilityCut(cut);
  }
  inline void setTrackCompatibilityToSV(float cut) {
    theFinder->setTrackCompatibilityToSV(cut);
  }
  inline void setVertexFitProbabilityCut(float cut) { 
    theFinder->setVertexFitProbabilityCut(cut);
  }
  inline void setMaxNbOfVertices(int max) { 
    theFinder->setMaxNbOfVertices(max);
  }

private:

  ConfigurableTrimmedVertexFinder * theFinder;

};

#endif
