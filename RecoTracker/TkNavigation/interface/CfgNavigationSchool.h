#ifndef TkNavigation_CfgNavigationSchool_H
#define TkNavigation_CfgNavigationSchool_H

#include "TrackingTools/DetLayers/interface/NavigationSchool.h" 
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"    
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
namespace edm{
  class ParameterSet;
}

class MagneticField;

class CfgNavigationSchool : public SimpleNavigationSchool { 
 public:
  CfgNavigationSchool(){};
  CfgNavigationSchool(const edm::ParameterSet & cfg,
		      const GeometricSearchTracker* theTracker,
		      const MagneticField* field);
  
  ~CfgNavigationSchool(){ cleanMemory();}
  
 protected:
  void makeBwdLinks(std::string & lname, BDLC & reachableBL, FDLC & reachableFL);
  void makeFwdLinks(std::string & lname, BDLC & reachableBL, FDLC & reachableFL);
  void addLayer(std::string & lname, BDLC & reachableBL, FDLC & reachableFL);
  DetLayer * layer(std::string & lname);
};

#endif
