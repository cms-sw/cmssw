#ifndef CosmicTrackingRegion_H
#define CosmicTrackingRegion_H

/** \class CosmicTrackingRegion
 * A concrete implementation of TrackingRegion. 
 * Apart of vertex constraint from TrackingRegion in this implementation
 * the region of interest is further constrainted in phi and eta 
 * around the direction of the region 
 */

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionBase.h"
#include "RecoTracker/TkTrackingRegions/interface/TkTrackingRegionsMargin.h"
//#include "CommonDet/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "RecoTracker/TkTrackingRegions/interface/HitRZConstraint.h"
#include "RecoTracker/TkTrackingRegions/interface/OuterHitPhiPrediction.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include <vector>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class OuterEstimator;
class BarrelDetLayer;
class ForwardDetLayer;
typedef TransientTrackingRecHit::ConstRecHitPointer SeedingHit;

class CosmicTrackingRegion : public TrackingRegionBase {
public:
  
  //  typedef TkTrackingRegionsMargin<float> Margin;

 virtual ~CosmicTrackingRegion() { }
 /** constructor (symmetric eta and phi margins). <BR>
  * dir        - the direction around which region is constructed <BR>
  *              the initial direction of the momentum of the particle 
  *              should be in the range <BR> 
  *              phi of dir +- deltaPhi <BR>
  *              eta of dir +- deltaEta <BR> 
  *              
  * vertexPos  - the position of the vertex (origin) of the of the region.<BR>
  *              It is a centre of cylinder constraind with rVertex, zVertex.
  *              The track of the particle should cross the cylinder <BR>
  *              WARNING: in the current implementaion the vertexPos is
  *              supposed to be placed on the beam line, i.e. to be of the form
  *              (0,0,float)
  *
  * ptMin      - minimal pt of interest <BR>
  * rVertex    - radius of the cylinder around beam line where the tracks
  *              of interest should point to. <BR>
  * zVertex    - half height of the cylinder around the beam line
  *              where the tracks of interest should point to.   <BR>
  * deltaEta   - allowed deviation of the initial direction of particle
  *              in eta in respect to direction of the region <BR>
  *  deltaPhi  - allowed deviation of the initial direction of particle
  *              in phi in respect to direction of the region 
  */
  CosmicTrackingRegion( const GlobalVector & dir, 
			const GlobalPoint & vertexPos,
			float ptMin, float rVertex, float zVertex,
			float deltaEta, float deltaPhi,
			float dummy = 0.)			
    : TrackingRegionBase( dir, vertexPos, Range( -1/ptMin, 1/ptMin), 
			  rVertex, zVertex),
    measurementTrackerName_("")
    { }

    CosmicTrackingRegion(const GlobalVector & dir,
			 const GlobalPoint & vertexPos,
			 float ptMin, float rVertex, float zVertex,
			 float deltaEta, float deltaPhi,
			 const edm::ParameterSet & extra)
      : TrackingRegionBase( dir, vertexPos, Range( -1/ptMin, 1/ptMin),
			    rVertex, zVertex)
      {
	measurementTrackerName_ = extra.getParameter<std::string>("measurementTrackerName");
      }
      


   virtual TrackingRegion::Hits hits(
      const edm::Event& ev,  
      const edm::EventSetup& es, 
      const ctfseeding::SeedingLayer* layer) const;

   virtual HitRZCompatibility* checkRZ(
      const DetLayer* layer,
      const Hit & outerHit,
      const edm::EventSetup& iSetup, 
      const DetLayer* outerlayer=0,
      float lr=0, float gz=0, float dr=0, float dz=0) const {return 0; }
   
   CosmicTrackingRegion * clone() const {     return new CosmicTrackingRegion(*this);  }
   
   std::string name() const { return "CosmicTrackingRegion"; }

private:

   std::string measurementTrackerName_;
};

#endif
