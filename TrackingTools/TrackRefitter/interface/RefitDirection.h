#ifndef TrackingTools_TrackRefitter_RefitDirection_H
#define TrackingTools_TrackRefitter_RefitDirection_H

/** \class RefitDirection
 *  Help class in order to handle the different refit possibilities
 *
 *  $Date: 2008/11/04 14:46:37 $
 *  $Revision: 1.2 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class RefitDirection {

public:
  
  enum GeometricalDirection{insideOut, outsideIn, undetermined};
  
  /// Constructor
  RefitDirection(){
    thePropagationDirection = anyDirection;
    theGeoDirection = undetermined;
  }

  RefitDirection(std::string& type){ 

    thePropagationDirection = anyDirection;
    theGeoDirection = undetermined;
 
    if (type == "alongMomentum") thePropagationDirection = alongMomentum;
    else if (type == "oppositeToMomentum") thePropagationDirection = oppositeToMomentum;
    else if (type == "insideOut") theGeoDirection = insideOut;
    else if (type == "outsideIn") theGeoDirection = outsideIn;
    else 
      throw cms::Exception("RefitDirection") 
	<<"Wrong refit direction chosen in TrackTransformer ParameterSet"
	<< "\n"
	<< "Possible choices are:"
	<< "\n"
	<< "RefitDirection = [alongMomentum, oppositeToMomentum, insideOut, outsideIn]";
  }
  
  /// Destructor
  virtual ~RefitDirection(){};

  // Operations
  inline GeometricalDirection geometricalDirection() const {
    if(theGeoDirection == undetermined) LogTrace("Reco|TrackingTools|TrackTransformer") << "Try to use undetermined geometrical direction";
    return theGeoDirection;
  }
  inline PropagationDirection propagationDirection() const {
    if(thePropagationDirection == anyDirection) LogTrace("Reco|TrackingTools|TrackTransformer") << "Try to use anyDirection as propagation direction";
    return thePropagationDirection;
  }
  
protected:
  
private:
  GeometricalDirection theGeoDirection;
  PropagationDirection thePropagationDirection;
};
#endif

