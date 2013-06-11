/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
///                                      ///
/// Andrew W. Rose, IC                   ///
/// Nicola Pozzobon, UNIPD               ///
///                                      ///
/// 2008                                 ///
/// 2010, May                            ///
/// 2011, June                           ///
/// ////////////////////////////////////////

#ifndef HIT_MATCHING_ALGO_BASE_H
#define HIT_MATCHING_ALGO_BASE_H

#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"

#include <sstream>
#include <string>
#include <map>
#include "classNameFinder.h"
  /** ************************ **/
  /**                          **/
  /**   DECLARATION OF CLASS   **/
  /**                          **/
  /** ************************ **/

  template< typename T >
  class HitMatchingAlgorithm
  {
    protected:
      /// Data members
      const StackedTrackerGeometry *theStackedTracker;
      std::string className_;
    public:
      /// Constructors
      HitMatchingAlgorithm( const StackedTrackerGeometry *aStackedTracker,
			    std::string fName )
        : theStackedTracker( aStackedTracker ){
	className_=classNameFinder<T>(fName);
      }


      /// Destructor
      virtual ~HitMatchingAlgorithm(){}

      /// Matching operations
      virtual void CheckTwoMemberHitsForCompatibility( bool &aConfirmation, int &aDisplacement, int &anOffset, const L1TkStub< T > &aL1TkStub ) const {}

      /// Algorithm name
      virtual std::string AlgorithmName() const { return className_; }

  }; /// Close class



#endif

