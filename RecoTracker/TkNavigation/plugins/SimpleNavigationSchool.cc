#include "SimpleNavigationSchool.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimpleBarrelNavigableLayer.h"
#include "SimpleForwardNavigableLayer.h"
#include "SimpleNavigableLayer.h"
#include "SymmetricLayerFinder.h"

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/DetLayers/src/DetBelowZ.h"
#include "TrackingTools/DetLayers/src/DetLessZ.h"

#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <algorithm>
#include <map>
#include <cmath>

using namespace std;

void SimpleNavigationSchool::init()
{

  theAllDetLayersInSystem=&theTracker->allLayers();
  theAllNavigableLayer.resize(theTracker->allLayers().size(),nullptr);


  // Get barrel layers
  for( auto i : theTracker->barrelLayers()) {
    theBarrelLayers.push_back( i );
  }

  // get forward layers
  for( auto i : theTracker->forwardLayers()) {
    theForwardLayers.push_back( i );
  }
  
  FDLI middle = find_if( theForwardLayers.begin(), theForwardLayers.end(),
			 not1(DetBelowZ(0)));
  theLeftLayers  = FDLC( theForwardLayers.begin(), middle);
  theRightLayers = FDLC( middle, theForwardLayers.end());
  
  SymmetricLayerFinder symFinder( theForwardLayers);

  // only work on positive Z side; negative by mirror symmetry later
  linkBarrelLayers( symFinder);
  linkForwardLayers( symFinder);
  establishInverseRelations();
}

void SimpleNavigationSchool::cleanMemory(){
  // free the memory allocated to the SimpleNavigableLayers
  for ( vector< SimpleBarrelNavigableLayer*>::const_iterator
          ib = theBarrelNLC.begin(); ib != theBarrelNLC.end(); ib++) {
    delete (*ib);
  }
  theBarrelNLC.clear();
  for ( vector< SimpleForwardNavigableLayer*>::const_iterator 
	  ifl = theForwardNLC.begin(); ifl != theForwardNLC.end(); ifl++) {
    delete (*ifl);
  }
  theForwardNLC.clear();
}

SimpleNavigationSchool::StateType 
SimpleNavigationSchool::navigableLayers()
{
  StateType result;
  for ( vector< SimpleBarrelNavigableLayer*>::const_iterator 
	  ib = theBarrelNLC.begin(); ib != theBarrelNLC.end(); ib++) {
    result.push_back( *ib);
  }
  for ( vector< SimpleForwardNavigableLayer*>::const_iterator 
	  ifl = theForwardNLC.begin(); ifl != theForwardNLC.end(); ifl++) {
    result.push_back( *ifl);
  }
  return result;
}

void SimpleNavigationSchool::
linkBarrelLayers( SymmetricLayerFinder& symFinder)
{
  // Link barrel layers outwards
  for ( BDLI i = theBarrelLayers.begin(); i != theBarrelLayers.end(); i++) {
    BDLC reachableBL;
    FDLC leftFL;
    FDLC rightFL;

    // always add next barrel layer first
    if ( i+1 != theBarrelLayers.end()) reachableBL.push_back(*(i+1));
 
    // Add closest reachable forward layer (except for last BarrelLayer)
    if (i != theBarrelLayers.end() - 1) {
      linkNextForwardLayer( *i, rightFL);
    }

    // Add next BarrelLayer with length larger than the current BL
    if ( i+2 < theBarrelLayers.end()) {
      linkNextLargerLayer( i, theBarrelLayers.end(), reachableBL);
    }

    theBarrelNLC.push_back( new 
       SimpleBarrelNavigableLayer( *i, reachableBL,
				   symFinder.mirror(rightFL),
				   rightFL,theField, 5.));
  }
}

void SimpleNavigationSchool::linkNextForwardLayer( const BarrelDetLayer* bl, 
						   FDLC& rightFL)
{
  // find first forward layer with larger Z and larger outer radius
  float length = bl->surface().bounds().length() / 2.;
  float radius = bl->specificSurface().radius();
  for ( FDLI fli = theRightLayers.begin();
	fli != theRightLayers.end(); fli++) {
    if ( length < (**fli).position().z() &&
	 radius < (**fli).specificSurface().outerRadius()) {
      //search if there are any sovrapposition between forward layers
      for ( FDLI fliNext = fli; fliNext != theRightLayers.end(); fliNext++) {
        if ( (**fliNext).position().z() < (**fli).position().z() && (**fliNext).specificSurface().innerRadius() < (**fli).specificSurface().outerRadius()) {
          rightFL.push_back( *fliNext);
          return;
        }
      }
      rightFL.push_back( *fli);
      return;
    }
  }
}

void SimpleNavigationSchool::linkNextLargerLayer( BDLI bli, BDLI end,
						  BDLC& reachableBL)
{
  // compare length of next layer with length of following ones
  float length = (**(bli+1)).surface().bounds().length();
  float epsilon = 0.1;

  for ( BDLI i = bli+2; i < end; i++) {
    if ( length + epsilon < (**i).surface().bounds().length()) {
      reachableBL.push_back( *i);
      return;
    }
  }
}

void SimpleNavigationSchool::
linkForwardLayers( SymmetricLayerFinder& symFinder)
{

  // handle right side first, groups are only on the right 
  vector<FDLC> groups = splitForwardLayers();

  LogDebug("TkNavigation") << "SimpleNavigationSchool, Forward groups size = " << groups.size() ;
  for (vector<FDLC>::iterator g = groups.begin(); g != groups.end(); g++) {
    LogDebug("TkNavigation") << "group " << g - groups.begin() << " has " 
			     << g->size() << " layers " ;
  }

  for ( vector<FDLC>::iterator group = groups.begin();
	group != groups.end(); group++) {

    for ( FDLI i = group->begin(); i != group->end(); i++) {

      BDLC reachableBL;
      FDLC reachableFL;
 
      // Always connect to next barrel layer first, if exists
      linkNextBarrelLayer( *i, reachableBL);

      // Then always connect to next forward layer of "same" size, 
      // and layers of larger inner Radius
      linkNextLayerInGroup( i, *group, reachableFL);

      // Then connect to next N fw layers of next size
      if ( group+1 != groups.end()) {
	linkOuterGroup( *i, *(group+1), reachableFL);
      }

      // or connect within the group if outer radius increases
      linkWithinGroup( i, *group, reachableFL);

      theForwardNLC.push_back( new SimpleForwardNavigableLayer( *i,reachableBL,
								reachableFL,
								theField,
								5.));
      theForwardNLC.push_back( new SimpleForwardNavigableLayer( symFinder.mirror(*i),
								reachableBL,
								symFinder.mirror(reachableFL),
								theField,
								5.));

    }
  }

//    // now the left side by symmetry
//    for ( FDLI ileft = theLeftLayers.begin(); 
//  	ileft != theLeftLayers.end(); ileft++) {
//      ForwardDetLayer* right = symFinder.mirror( *ileft);
    
//      theForwardNLC.push_back( new 
//         SimpleForwardNavigableLayer( *ileft , right->nextBarrelLayers(),
//  	                      symFinder.mirror(right->nextForwardLayers())));
//    }
}

void SimpleNavigationSchool::linkNextBarrelLayer( const ForwardDetLayer* fl,
						  BDLC& reachableBL)
{
  if ( fl->position().z() > barrelLength()) return;

  float outerRadius = fl->specificSurface().outerRadius();
  float zpos        = fl->position().z();
  for ( BDLI bli = theBarrelLayers.begin(); bli != theBarrelLayers.end(); bli++) {
    if ( outerRadius < (**bli).specificSurface().radius() &&
	 zpos        < (**bli).surface().bounds().length() / 2.) {
      reachableBL.push_back( *bli);
      return;
    }
  }
}


void SimpleNavigationSchool::linkNextLayerInGroup( FDLI fli,
						   const FDLC& group,
						   FDLC& reachableFL)
{
  // Always connect to next forward layer of "same" size, if exists
  if ( fli+1 != group.end()) {
    reachableFL.push_back( *(fli+1));
    // If that layer has an inner radius larger then the current one
    // also connect ALL next disks of same radius.
    float innerRThis = (**fli).specificSurface().innerRadius();
    float innerRNext =  (**(fli+1)).specificSurface().innerRadius();
    const float epsilon = 2.f;

    if (innerRNext > innerRThis + epsilon) {
      // next disk is smaller, so it doesn't cover fully subsequent ones
      // of same radius

      int i = 2;
      while ( (fli+i) != group.end()) {
	if ( (**(fli+i)).specificSurface().innerRadius() < 
	     innerRNext + epsilon) {
	  // following disk has not increased in ineer radius 
	  reachableFL.push_back( *(fli+i));
	  i++;
	} else {
	  break;
	}
      }
    }
  }
}


void SimpleNavigationSchool::linkOuterGroup( const ForwardDetLayer* fl,
					     const FDLC& group,
					     FDLC& reachableFL)
{

  // insert N layers with Z grater than fl

  ConstFDLI first = find_if( group.begin(), group.end(), 
			     not1( DetBelowZ( fl->position().z())));
  if ( first != group.end()) {

    // Hard-wired constant!!!!!!
    ConstFDLI last = min( first + 7, group.end());

    reachableFL.insert( reachableFL.end(), first, last);
  }
}

void SimpleNavigationSchool::linkWithinGroup( FDLI fl,
					      const FDLC& group,
					      FDLC& reachableFL)
{
  ConstFDLI biggerLayer = outerRadiusIncrease( fl, group);
  if ( biggerLayer != group.end() && biggerLayer != fl+1) {
    reachableFL.push_back( *biggerLayer);
  }
}

SimpleNavigationSchool::ConstFDLI
SimpleNavigationSchool::outerRadiusIncrease( FDLI fl, const FDLC& group)
{
  const float epsilon = 5.f;
  float outerRadius = (**fl).specificSurface().outerRadius();
  while ( ++fl != group.end()) {
    if ( (**fl).specificSurface().outerRadius() > outerRadius + epsilon) {
      return fl;
    }
  }
  return fl;
}

vector<SimpleNavigationSchool::FDLC> 
SimpleNavigationSchool::splitForwardLayers() 
{
  // only work on positive Z side; negative by mirror symmetry later

  FDLC myRightLayers( theRightLayers);
  FDLI begin = myRightLayers.begin();
  FDLI end   = myRightLayers.end();

  // sort according to inner radius, but keeping the ordering in z!
  std::stable_sort ( begin, end, []( const ForwardDetLayer* a, const ForwardDetLayer* b)
          { return a->specificSurface().innerRadius() < b->specificSurface().innerRadius();});

  // partition in cylinders
  vector<FDLC> result;
  FDLC current;
  current.push_back( *begin);
  for ( FDLI i = begin+1; i != end; i++) {

#ifdef EDM_ML_DEBUG
    LogDebug("TkNavigation") << "(**i).specificSurface().innerRadius()      = "
			     << (**i).specificSurface().innerRadius() << endl
			     << "(**(i-1)).specificSurface().outerRadius()) = "
			     << (**(i-1)).specificSurface().outerRadius() ;
    LogDebug("TkNavigation") << "(**i).specificSurface().position().z()      = "
                            << (**i).specificSurface().position().z() << endl
                            << "(**(i-1)).specificSurface().position().z() = "
                            << (**(i-1)).specificSurface().position().z() ;
#endif

    // if inner radius of i is larger than outer radius of i-1 then split!
    // FIXME: The solution found for phase2 is a bit dirty, we can do better.
    // For phase2 we compare the EXTENDED pixel with the TID to get the assignment right!
    if ( (**i).specificSurface().innerRadius() > (**(i-1)).specificSurface().outerRadius() ||
         (theTracker->posPixelForwardLayers().back()->specificSurface().position().z() > theTracker->posTidLayers().front()->specificSurface().position().z() &&
          (**i).specificSurface().position().z() < (**(i-1)).specificSurface().position().z()) ){

      LogDebug("TkNavigation") << "found break between groups" ;

      // sort layers in group along Z
      std::stable_sort ( current.begin(), current.end(), isDetLessZ);

      result.push_back(current);
      current.clear();
    }
    current.push_back(*i);
  }
  result.push_back(current); // save last one too 

  // now sort subsets in Z
  for ( vector<FDLC>::iterator ivec = result.begin();
	ivec != result.end(); ivec++) {
      std::stable_sort( ivec->begin(), ivec->end(), isDetLessZ);
  }

  return result;
}

float SimpleNavigationSchool::barrelLength() 
{
  if ( theBarrelLength < 1.) {
    for (BDLI i=theBarrelLayers.begin(); i!=theBarrelLayers.end(); i++) {
      theBarrelLength = max( theBarrelLength,
			     (**i).surface().bounds().length() / 2.f);
    }

    LogDebug("TkNavigation") << "The barrel length is " << theBarrelLength ;
  }
  return theBarrelLength;
}

void SimpleNavigationSchool::establishInverseRelations() {

  // NavigationSetter setter(*this);

  setState(navigableLayers());

    // find for each layer which are the barrel and forward
    // layers that point to it
    typedef map<const DetLayer*, vector<const BarrelDetLayer*>, less<const DetLayer*> > BarrelMapType;
    typedef map<const DetLayer*, vector<const ForwardDetLayer*>, less<const DetLayer*> > ForwardMapType;


    BarrelMapType reachedBarrelLayersMap;
    ForwardMapType reachedForwardLayersMap;


    for ( auto bli :  theBarrelLayers) {
      auto reachedLC = nextLayers(*bli, insideOut);
      for ( auto i : reachedLC) {
        reachedBarrelLayersMap[i].push_back(bli);
      }
    }

    for ( auto fli : theForwardLayers) {
      auto reachedLC = nextLayers(*fli, insideOut);
      for ( auto i : reachedLC) {
        reachedForwardLayersMap[i].push_back(fli);
      }
    }

    /*
    vector<DetLayer*> lc = theTracker->allLayers();
    for ( vector<DetLayer*>::iterator i = lc.begin(); i != lc.end(); i++) {
      SimpleNavigableLayer* navigableLayer =
	dynamic_cast<SimpleNavigableLayer*>((**i).navigableLayer());
      navigableLayer->setInwardLinks( reachedBarrelLayersMap[*i],reachedForwardLayersMap[*i] );
    }
    */
    
    
    for(auto nl : theAllNavigableLayer) {
      if (!nl) continue;
      auto navigableLayer = static_cast<SimpleNavigableLayer*>(nl);
      auto dl = nl->detLayer();
      navigableLayer->setInwardLinks( reachedBarrelLayersMap[dl],reachedForwardLayersMap[dl] );
    }
    
    
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

 
#include "NavigationSchoolFactory.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
DEFINE_EDM_PLUGIN(NavigationSchoolFactory, SimpleNavigationSchool, "SimpleNavigationSchool");

