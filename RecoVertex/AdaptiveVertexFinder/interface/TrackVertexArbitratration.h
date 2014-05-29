#include <memory>
#include <set>


#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"


#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexUpdator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackCompatibilityEstimator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexSmoother.h"
//#include "DataFormats/Math/interface/deltaR.h"

//#define VTXDEBUG

class TrackVertexArbitration{
    public:
	TrackVertexArbitration(const edm::ParameterSet &params);


	reco::VertexCollection trackVertexArbitrator(
          edm::Handle<reco::BeamSpot> &beamSpot, 
	  const reco::Vertex &pv,
	  edm::ESHandle<TransientTrackBuilder> &trackBuilder,
	  const edm::RefVector< reco::TrackCollection > & selectedTracks,
	  reco::VertexCollection & secondaryVertices
	);
	
	
    private:
	bool trackFilterArbitrator(const reco::TrackRef &track) const;

	edm::InputTag				primaryVertexCollection;
	edm::InputTag				secondaryVertexCollection;
	edm::InputTag				trackCollection;
        edm::InputTag                           beamSpotCollection;
        double 					dRCut;
        double					distCut;
        double					sigCut;
	double					dLenFraction;
	double 					fitterSigmacut; // = 3.0;
	double 					fitterTini; // = 256.;
	double 					fitterRatio;//    = 0.25;
	int 					trackMinLayers;
	double					trackMinPt;
	int					trackMinPixels;   
};
