#ifndef TSiTrackerMultiRecHit_h
#define TSiTrackerMultiRecHit_h

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerMultiRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/HelpertRecHit2DLocalPos.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

/*
A TransientTrackingRecHit for the SiTrackerMultiRecHit
*/

class TSiTrackerMultiRecHit : public TransientTrackingRecHit {
	public:
	//virtual ~TSiTrackerMultiRecHit() {delete theHitData;}
	virtual ~TSiTrackerMultiRecHit() {}
	
	virtual AlgebraicVector parameters() const {return theHitData.parameters();}
	virtual AlgebraicSymMatrix parametersError() const {
		return HelpertRecHit2DLocalPos().parError( theHitData.localPositionError(), *det());
	    	//return theHitData.parametersError();
	}

        virtual void getKfComponents( KfComponentsHolder & holder ) const {
            HelpertRecHit2DLocalPos().getKfComponents(holder, theHitData, *det()); 
        }
	virtual DetId geographicalId() const {return theHitData.geographicalId();}
	virtual AlgebraicMatrix projectionMatrix() const {return theHitData.projectionMatrix();}
	virtual int dimension() const {return theHitData.dimension();}

        virtual LocalPoint localPosition() const {return theHitData.localPosition();}
	virtual LocalError localPositionError() const {return theHitData.localPositionError();}

	virtual const TrackingRecHit * hit() const {return &theHitData;};
	const SiTrackerMultiRecHit* specificHit() const {return &theHitData;}

	virtual bool isValid() const{return theHitData.isValid();}

	virtual std::vector<const TrackingRecHit*> recHits() const {
		return theHitData.recHits();
	} 
	virtual std::vector<TrackingRecHit*> recHits() {
    		return theHitData.recHits();
  	}	

  	virtual const GeomDetUnit* detUnit() const;

  	virtual bool canImproveWithTrack() const {return true;}

  	virtual RecHitPointer clone(const TrajectoryStateOnSurface& ts) const;

        virtual ConstRecHitContainer transientHits() const {return theComponents;};

	static RecHitPointer build( const GeomDet * geom, const SiTrackerMultiRecHit* rh, 
				    const ConstRecHitContainer& components, float annealing=1.){
		return RecHitPointer(new TSiTrackerMultiRecHit( geom, rh, components, annealing));
	}
	private:
	SiTrackerMultiRecHit theHitData;
	//holds the TransientTrackingRecHit components of the MultiRecHit with up-to-date weights 
   	ConstRecHitContainer theComponents;   
	
	TSiTrackerMultiRecHit(const GeomDet * geom, const SiTrackerMultiRecHit* rh,  
			      const ConstRecHitContainer& components, float annealing):
    		TransientTrackingRecHit(geom,1, annealing), theHitData(*rh), theComponents(components){}
      
	virtual TSiTrackerMultiRecHit* clone() const {
    		return new TSiTrackerMultiRecHit(*this);
  	}

};		

#endif
