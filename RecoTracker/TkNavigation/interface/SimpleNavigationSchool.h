#ifndef TkNavigation_SimpleNavigationSchool_H
#define TkNavigation_SimpleNavigationSchool_H

#include "TrackingTools/DetLayers/interface/NavigationSchool.h"

#include <vector>

class DetLayer;
class BarrelDetLayer;
class ForwardDetLayer;
class SymmetricLayerFinder;
class SimpleBarrelNavigableLayer;
class SimpleForwardNavigableLayer;
class MagneticField;

/** Concrete navigation school for the Tracker
 */

class SimpleNavigationSchool : public NavigationSchool {
public:
  
  SimpleNavigationSchool(const MagneticField* field);
  
  // from base class
  virtual StateType navigableLayers() const;

private:

  typedef vector<const DetLayer*>              DLC;
  typedef vector<BarrelDetLayer*>              BDLC;
  typedef vector<ForwardDetLayer*>             FDLC;
  typedef DLC::iterator                        DLI;
  typedef BDLC::iterator                       BDLI;
  typedef FDLC::iterator                       FDLI;
  typedef BDLC::const_iterator                 ConstBDLI;
  typedef FDLC::const_iterator                 ConstFDLI;
 
  BDLC theBarrelLayers;
  FDLC theForwardLayers;  
  FDLC theRightLayers;
  FDLC theLeftLayers;
  float theBarrelLength;

  typedef vector< SimpleBarrelNavigableLayer*>   BNLCType;
  typedef vector< SimpleForwardNavigableLayer*>  FNLCType;
  BNLCType  theBarrelNLC;
  FNLCType  theForwardNLC;

  void linkBarrelLayers( SymmetricLayerFinder& symFinder);
  void linkForwardLayers( SymmetricLayerFinder& symFinder);

  void linkNextForwardLayer( BarrelDetLayer*, FDLC&);

  void linkNextLargerLayer( BDLI, BDLI, BDLC&);

  void linkNextBarrelLayer( ForwardDetLayer* fl, BDLC&);

  void linkOuterGroup( ForwardDetLayer* fl,
		       const FDLC& group,
		       FDLC& reachableFL);

  void linkWithinGroup( FDLI fl, const FDLC& group, FDLC& reachableFL);
  
  ConstFDLI outerRadiusIncrease( FDLI fl, const FDLC& group);

  vector<FDLC> splitForwardLayers();

  float barrelLength();

  void establishInverseRelations();

  void linkNextLayerInGroup( FDLI fli, const FDLC& group, FDLC& reachableFL);

  const MagneticField* theField;
};

#endif // SimpleNavigationSchool_H
