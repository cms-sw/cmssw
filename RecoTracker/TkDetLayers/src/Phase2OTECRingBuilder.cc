#include "Phase2OTECRingBuilder.h"

using namespace edm;
using namespace std;

Phase2OTECRing* Phase2OTECRingBuilder::build(const GeometricDet* aPhase2OTECRing,
			 const TrackerGeometry* theGeomDetGeometry)
{
  vector<const GeometricDet*>  theGeometricDets = aPhase2OTECRing->components();
  //edm::LogInfo(TkDetLayers) << "theGeometricDets.size(): " << theGeometricDets.size() ;


  vector<const GeomDet*> innerGeomDets;
  vector<const GeomDet*> outerGeomDets;
  vector<const GeomDet*> innerGeomDetBrothers;
  vector<const GeomDet*> outerGeomDetBrothers;

  //---- to evaluate meanZ
  double meanZ = 0;
  for(vector<const GeometricDet*>::const_iterator it=theGeometricDets.begin();
      it!=theGeometricDets.end();it++){
    meanZ = meanZ + (*it)->positionBounds().z();
  }
  meanZ = meanZ/theGeometricDets.size();
  //----

  unsigned int counter=0;
  for(vector<const GeometricDet*>::const_iterator it=theGeometricDets.begin();
      it!=theGeometricDets.end();it++,counter++){

    if(counter%2 == 0) {

      const GeomDet* theGeomDet = theGeomDetGeometry->idToDet( (*it)->geographicalID() );
      
      if( std::abs( (*it)->positionBounds().z() ) < std::abs(meanZ))
	innerGeomDets.push_back(theGeomDet);
      
      if( std::abs( (*it)->positionBounds().z() ) > std::abs(meanZ))
	outerGeomDets.push_back(theGeomDet);      
    }
    else {

      const GeomDet* theGeomDet = theGeomDetGeometry->idToDet( (*it)->geographicalID() );
      
      if( std::abs( (*it)->positionBounds().z() ) < std::abs(meanZ))
	innerGeomDetBrothers.push_back(theGeomDet);
      
      if( std::abs( (*it)->positionBounds().z() ) > std::abs(meanZ))
	outerGeomDetBrothers.push_back(theGeomDet);      
    }
  }

  //edm::LogInfo(TkDetLayers) << "innerGeomDets.size(): " << innerGeomDets.size() ;
  //edm::LogInfo(TkDetLayers) << "outerGeomDets.size(): " << outerGeomDets.size() ;

  return new Phase2OTECRing(innerGeomDets,outerGeomDets,innerGeomDetBrothers,outerGeomDetBrothers);
}
