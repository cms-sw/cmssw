/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
///                                      ///
/// Andrew W. Rose, IC                   ///
/// Nicola Pozzobon, UNIPD               ///
///                                      ///
/// 2008                                 ///
/// 2010, May                            ///
/// ////////////////////////////////////////

#ifndef HIT_MATCHING_ALGORITHM_a_H
#define HIT_MATCHING_ALGORITHM_a_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithm.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithmRecord.h"


#include <boost/shared_ptr.hpp>
#include <memory>
#include <string>
#include <map>

  /** ************************ **/
  /**                          **/
  /**   DECLARATION OF CLASS   **/
  /**                          **/
  /** ************************ **/

  template< typename T >
  class HitMatchingAlgorithm_a : public HitMatchingAlgorithm< T >
  {
    private:
      /// Data members

    public:
      /// Constructor
      HitMatchingAlgorithm_a( const StackedTrackerGeometry *aStackedTracker )
        : HitMatchingAlgorithm< T >( aStackedTracker,__func__ ) {}

      /// Destructor
      ~HitMatchingAlgorithm_a(){}

      /// Matching operations
      void CheckTwoMemberHitsForCompatibility( bool &aConfirmation, int &aDisplacement, int &anOffset, const L1TkStub< T > &aL1TkStub ) const
      {
        aConfirmation = true; 
      }


  }; /// Close class



/** ********************** **/
/**                        **/
/**   DECLARATION OF THE   **/
/**    ALGORITHM TO THE    **/
/**       FRAMEWORK        **/
/**                        **/
/** ********************** **/

template<  typename T  >
class  ES_HitMatchingAlgorithm_a : public edm::ESProducer
{
  private:
    /// Data members
    boost::shared_ptr< HitMatchingAlgorithm< T > > _theAlgo;

  public:
    /// Constructor
    ES_HitMatchingAlgorithm_a( const edm::ParameterSet & p )
    {
      setWhatProduced( this );
    }

    /// Destructor
    virtual ~ES_HitMatchingAlgorithm_a(){}

    boost::shared_ptr< HitMatchingAlgorithm< T > > produce( const HitMatchingAlgorithmRecord & record )
    { 
      edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
      record.getRecord< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );

      HitMatchingAlgorithm< T >* HitMatchingAlgo =
        new HitMatchingAlgorithm_a< T >( &(*StackedTrackerGeomHandle) );

      _theAlgo = boost::shared_ptr< HitMatchingAlgorithm< T > >( HitMatchingAlgo );
      return _theAlgo;
    }

};

#endif

