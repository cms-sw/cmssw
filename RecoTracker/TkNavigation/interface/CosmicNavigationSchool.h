#ifndef TkNavigation_CosmicNavigationSchool_H
#define TkNavigation_CosmicNavigationSchool_H

#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>

//class FakeDetLayer;


/** Concrete navigation school for cosmics in the Tracker
 */

class CosmicNavigationSchool : public SimpleNavigationSchool {
public:
  CosmicNavigationSchool(const GeometricSearchTracker* theTracker,
			 const MagneticField* field);
  ~CosmicNavigationSchool(){ cleanMemory();}

  class CosmicNavigationSchoolConfiguration{
  public:
    CosmicNavigationSchoolConfiguration() : noPXB(false), noPXF(false), noTOB(false), noTIB(false), noTEC(false), noTID(false) , self(false), allSelf(false) {}
    CosmicNavigationSchoolConfiguration(const edm::ParameterSet conf);
    bool noPXB;
    bool noPXF;
    bool noTOB;
    bool noTIB;
    bool noTEC;
    bool noTID;
    
    bool self;
    bool allSelf;
  };

  void build(const GeometricSearchTracker* theTracker,
	     const MagneticField* field,
	     const CosmicNavigationSchoolConfiguration conf);
 
protected:
  CosmicNavigationSchool(){}
private:

  //FakeDetLayer* theFakeDetLayer;
  void linkBarrelLayers( SymmetricLayerFinder& symFinder);
  //void linkForwardLayers( SymmetricLayerFinder& symFinder); 
  void establishInverseRelations( SymmetricLayerFinder& symFinder );
  void buildAdditionalBarrelLinks();
  void buildAdditionalForwardLinks(SymmetricLayerFinder& symFinder);
};

#endif // CosmicNavigationSchool_H
