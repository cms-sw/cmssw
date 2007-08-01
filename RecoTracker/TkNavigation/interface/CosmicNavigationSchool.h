#ifndef TkNavigation_CosmicNavigationSchool_H
#define TkNavigation_CosmicNavigationSchool_H

//#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include <vector>
//Giulio
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"

class FakeDetLayer : public BarrelDetLayer{
        public:
        FakeDetLayer(BoundPlane* cp, BoundCylinder* cyl) : thePlane(cp){
		setSurface(cyl);
        }
        virtual ~FakeDetLayer(){};
 
        // GeometricSearchDet interface
        virtual const BoundSurface&  surface() const { return *thePlane;}
 
        virtual std::pair<bool, TrajectoryStateOnSurface>
                compatible( const TrajectoryStateOnSurface& ts , const Propagator& prop, 
                            const MeasurementEstimator& est) const{
                        std::cout << "In FakeDetLayer::compatible" << std::endl;
                        TrajectoryStateOnSurface myState = prop.propagate( ts, surface());
                        return std::make_pair( myState.isValid(), myState);
        };
        
        virtual const std::vector<const GeomDet*>& basicComponents() const {return theBasicComps;}
        virtual const std::vector<const GeometricSearchDet*>& components() const {std::cout << "ask components()"<< std::endl;return theComps;} 

        virtual std::vector<GeometricSearchDet::DetWithState> 
                compatibleDets( const TrajectoryStateOnSurface& tsos,
                   const Propagator& prop, 
                   const MeasurementEstimator& est) const {std::vector<GeometricSearchDet::DetWithState> none; return none;} 
 
        virtual std::vector<DetGroup> 
                groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
                          const Propagator& prop,
                          const MeasurementEstimator& est) const {std::vector<DetGroup> none; return none;}
 
        virtual Location   location()   const {return GeomDetEnumerators::barrel;}

        virtual bool hasGroups() const {return true;}; 

        virtual SubDetector subDetector() const {return GeomDetEnumerators::TIB;} 
	bool contains( const Local3DPoint& p) const {return thePlane->bounds().inside(p); }  

        private:
        ReferenceCountingPointer<BoundPlane>  thePlane;
        std::vector<const GeomDet*> theBasicComps;
        std::vector<const GeometricSearchDet*> theComps;



};


//Giulio




//class FakeDetLayer;


/** Concrete navigation school for cosmics in the Tracker
 */

class CosmicNavigationSchool : public SimpleNavigationSchool {
public:
  
  CosmicNavigationSchool(const GeometricSearchTracker* theTracker,
			 const MagneticField* field);
 
  ~CosmicNavigationSchool();
private:
  FakeDetLayer* theFakeDetLayer;
  void linkBarrelLayers( SymmetricLayerFinder& symFinder);
  void linkToAllRegularBarrelLayer(BDLC&);
  void establishInverseRelations();
};

#endif // CosmicNavigationSchool_H
