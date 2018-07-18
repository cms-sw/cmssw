#ifndef TkNavigation_SimpleNavigableLayer_H
#define TkNavigation_SimpleNavigableLayer_H

#include "TrackingTools/DetLayers/interface/NavigableLayer.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/DetLayers/interface/TkLayerLess.h"

#include <vector>

class GtfPropagator;

/** A partial implementation of the NavigableLayer
 */

class dso_hidden SimpleNavigableLayer : public NavigableLayer {
public:

  typedef std::vector<const DetLayer*>              DLC;
  typedef std::vector<const BarrelDetLayer*>        BDLC;
  typedef std::vector<const ForwardDetLayer*>       FDLC;

  SimpleNavigableLayer( const MagneticField* field,float eps,bool checkCrossingSide=true) :
    theField(field), theEpsilon(eps),theCheckCrossingSide(checkCrossingSide),theSelfSearch(false) {}

  virtual void setInwardLinks(const BDLC&, const FDLC&, TkLayerLess sorter = TkLayerLess(outsideIn)) = 0;
  
  virtual void setAdditionalLink(const DetLayer*, NavigationDirection direction=insideOut) = 0;

  void setCheckCrossingSide(bool docheck) {theCheckCrossingSide = docheck;}

  using NavigableLayer::compatibleLayers;
  std::vector< const DetLayer * > compatibleLayers (const FreeTrajectoryState &fts, 
							    PropagationDirection timeDirection,
							    int& counter) const  final;
  
protected:
  
  const MagneticField * theField;
  float                     theEpsilon;
  bool theCheckCrossingSide;
public:
  bool theSelfSearch;
protected:
 
 typedef BDLC::iterator                       BDLI;
  typedef FDLC::iterator                       FDLI;
  typedef BDLC::const_iterator                 ConstBDLI;
  typedef FDLC::const_iterator                 ConstFDLI;
  typedef TrajectoryStateOnSurface             TSOS;


  bool wellInside( const FreeTrajectoryState& fts, PropagationDirection dir,
		   const BarrelDetLayer* bl, DLC& result) const dso_internal;

  bool wellInside( const FreeTrajectoryState& fts, PropagationDirection dir,
		   const ForwardDetLayer* bl, DLC& result) const dso_internal;

  bool wellInside( const FreeTrajectoryState& fts, PropagationDirection dir,
		   ConstBDLI begin, ConstBDLI end, DLC& result) const dso_internal;

  bool wellInside( const FreeTrajectoryState& fts, PropagationDirection dir,
		   const DLC& layers, DLC& result) const dso_internal;

  bool wellInside( const FreeTrajectoryState& fts, PropagationDirection dir,
		   ConstFDLI begin, ConstFDLI end, DLC& result) const dso_internal;

  AnalyticalPropagator propagator( PropagationDirection dir) const{
    AnalyticalPropagator aPropagator(theField);
    aPropagator.setPropagationDirection(dir);
    return aPropagator;
  }

  TSOS crossingState(const FreeTrajectoryState& fts,PropagationDirection dir) const dso_internal;
  
};

#endif // SimpleNavigableLayer_H
