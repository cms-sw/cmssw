#ifndef _TrimmedVertexFitter_H_
#define _TrimmedVertexFitter_H_

#include "RecoVertex/VertexPrimitives/interface/VertexFitter.h"
#include "RecoVertex/TrimmedKalmanVertexFinder/interface/KalmanTrimmedVertexFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
/*
 *  Turn the TrimmedVertexFinder into a VertexFitter.
 */

class TrimmedVertexFitter : public VertexFitter {

public:

  TrimmedVertexFitter(const edm::ParameterSet & pSet);

  virtual ~TrimmedVertexFitter(){}

  virtual CachingVertex vertex(const vector<reco::TransientTrack> & tracks) const;

  virtual CachingVertex vertex(const vector<RefCountedVertexTrack> & tracks) const;

  virtual CachingVertex vertex(const vector<reco::TransientTrack> & tracks,
  			const GlobalPoint& linPoint) const;

  virtual CachingVertex vertex(const vector<reco::TransientTrack> & tracks,
  			const GlobalPoint& priorPos,
			const GlobalError& priorError) const;

  virtual CachingVertex vertex(const vector<RefCountedVertexTrack> & tracks,
	 		const GlobalPoint& priorPos,
			const GlobalError& priorError) const;

  virtual CachingVertex vertex(const vector<reco::TransientTrack> & tracks,
		const reco::BeamSpot& beamSpot) const;


   // Clone method
  TrimmedVertexFitter * clone() const;


  void setPtCut ( float cut );
  void setTrackCompatibilityCut ( float cut );
  void setVertexFitProbabilityCut ( float cut );

private:
  KalmanTrimmedVertexFinder theRector;
  double ptcut;
};

#endif
