#ifndef TrackingTools_IPTools_h
#define TrackingTools_IPTools_h
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include <utility>
#include "DataFormats/CLHEP/interface/Migration.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "RecoVertex/VertexTools/interface/VertexDistance.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"


 
namespace IPTools
{
   /**
   *  Returns the unsigned transverse impact parameter
   *  The track is extrapolated to the closest point to the primary vertex in transverse plane
   *  then the impact parameter and its error are computed
   */
        std::pair<bool,Measurement1D> absoluteImpactParameter3D(const reco::TransientTrack & transientTrack, const  reco::Vertex & vertex);

   /**
   *  Returns the unsigned 3D impact parameter
   *  The track is extrapolated to the closest point to the primary vertex in 3d space
   *  then the impact parameter and its error are computed
   */

        std::pair<bool,Measurement1D> absoluteTransverseImpactParameter(const reco::TransientTrack & transientTrack, const  reco::Vertex & vertex);

   /**
   *  Returns life time signed transverse impact parameter
   *  The track is extrapolated to the closest point to the primary vertex in transverse plane
   *  then the impact parameter and its error are computed
   */
	std::pair<bool,Measurement1D> signedTransverseImpactParameter(const reco::TransientTrack & track,
 	         const GlobalVector & direction, const  reco::Vertex & vertex);

   /**
   *  Returns life time signed 3D impact parameter
   *  The track is extrapolated to the closest point to the primary vertex in 3d space
   *  then the impact parameter and its error are computed
   */

	std::pair<bool,Measurement1D> signedImpactParameter3D(const reco::TransientTrack & track,
 	         const GlobalVector & direction, const  reco::Vertex & vertex);

        /// Impact parameter without direction (internally used)
        std::pair<bool,Measurement1D> absoluteImpactParameter(const TrajectoryStateOnSurface & tsos  , const  reco::Vertex & vertex, VertexDistance & distanceComputer) ;


 	inline	TrajectoryStateOnSurface transverseExtrapolate(const TrajectoryStateOnSurface & track,const GlobalPoint & vertexPosition, const MagneticField * field)
	{
 		 TransverseImpactPointExtrapolator extrapolator(field);
 		 return extrapolator.extrapolate(track, vertexPosition);
	}



	TrajectoryStateOnSurface closestApproachToJet(const TrajectoryStateOnSurface & state, const reco::Vertex & vertex,
		 const GlobalVector& aJetDirection,const MagneticField * field);

	GlobalVector linearImpactParameter(const TrajectoryStateOnSurface & aTSOS, const GlobalPoint & point);

        
        std::pair<bool,Measurement1D> linearizedSignedImpactParameter3D(const TrajectoryStateOnSurface & state,
                 const GlobalVector & direction, const  reco::Vertex & vertex);
        
	inline std::pair<bool,Measurement1D> linearizedSignedImpactParameter3D(const reco::TransientTrack & transientTrack,
                 const GlobalVector & direction, const  reco::Vertex & vertex)
        {
	  // extrapolate to the point of closest approach to the jet axis
	  TrajectoryStateOnSurface closestToJetState = closestApproachToJet(transientTrack.impactPointState(), vertex, direction,transientTrack.field());
  	 return linearizedSignedImpactParameter3D(closestToJetState,direction,vertex);
	}
	
       std::pair<bool,Measurement1D> signedDecayLength3D(const TrajectoryStateOnSurface & state,
                 const GlobalVector & direction, const  reco::Vertex & vertex);

       inline std::pair<bool,Measurement1D> signedDecayLength3D(const reco::TransientTrack & transientTrack,
                 const GlobalVector & direction, const  reco::Vertex & vertex)
        {
          // extrapolate to the point of closest approach to the jet axis
          TrajectoryStateOnSurface closestToJetState = closestApproachToJet(transientTrack.impactPointState(), vertex, direction,transientTrack.field());
         return signedDecayLength3D(closestToJetState,direction,vertex);
        }
 

	std::pair<double,Measurement1D> jetTrackDistance(const reco::TransientTrack & track, const GlobalVector & direction,
                 const reco::Vertex & vertex);
}

#endif
