/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
///                                      ///
/// Kristofer Henriksson                 ///
/// Nicola Pozzobon, UNIPD               ///
///                                      ///
/// 2009                                 ///
/// 2010, May                            ///
/// 2011, June                           ///
/// ////////////////////////////////////////

#ifndef HIT_MATCHING_ALGORITHM_pixelray_H
#define HIT_MATCHING_ALGORITHM_pixelray_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithm.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithmRecord.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithm_pixelray_helper.h"

#include "CLHEP/Units/PhysicalConstants.h"
#include <boost/shared_ptr.hpp>
#include <memory>
#include <string>
#include <map>

  /** ************************ **/
  /**                          **/
  /**   DECLARATION OF CLASS   **/
  /**                          **/
  /** ************************ **/

  template < typename T >
  class HitMatchingAlgorithm_pixelray : public HitMatchingAlgorithm< T >
  {
    private:
      /// Data members
      double                       mCompatibilityScalingFactor;
      double                       mIPWidth;

    public:
      /// Constructor
      HitMatchingAlgorithm_pixelray( const StackedTrackerGeometry *aStackedTracker,
                                     double aCompatibilityScalingFactor,
                                     double aIPWidth )
        : HitMatchingAlgorithm< T >( aStackedTracker,__func__ )
      {
        mCompatibilityScalingFactor = aCompatibilityScalingFactor;
        mIPWidth = aIPWidth;
      }

      /// Destructor
      ~HitMatchingAlgorithm_pixelray(){}

      /// Matching operations
      void CheckTwoMemberHitsForCompatibility( bool &aConfirmation, int &aDisplacement, int &anOffset, const L1TkStub< T > &aL1TkStub ) const;


  }; /// Close class

  /** ***************************** **/
  /**                               **/
  /**   IMPLEMENTATION OF METHODS   **/
  /**                               **/
  /** ***************************** **/

  /// Matching operations
  template< typename T >
  void HitMatchingAlgorithm_pixelray< T >::CheckTwoMemberHitsForCompatibility( bool &aConfirmation, int &aDisplacement, int &anOffset, const L1TkStub< T > &aL1TkStub) const
  {
    /// Convert DetId
    StackedTrackerDetId stDetId( aL1TkStub.getDetId() );

    /// Force this to be a BARREL-only algorithm
    if ( stDetId.isEndcap() )
    {
      aConfirmation = false;
      return;
    }

    /// Prepare pixelray
    std::pair< double, double > * rayEndpoints;
        
    /// Just call the helper function to do all the work
    rayEndpoints = getPixelRayEndpoints( aL1TkStub,
                                         HitMatchingAlgorithm< T >::theStackedTracker,
                                         mCompatibilityScalingFactor );

    /// If positive define window
    if (rayEndpoints)
    {
      /// Establish the valid window
      double positiveZBoundary =  mIPWidth/2;
      double negativeZBoundary = -mIPWidth/2;

      /// Is it really within the window?
      if ( ( rayEndpoints->second > negativeZBoundary ) &&
           ( rayEndpoints->first < positiveZBoundary ) )
      {
        aConfirmation = true;

        /// Calculate output
        /// NOTE this assumes equal pitch in both sensors!
        MeasurementPoint mp0 = aL1TkStub.getClusterPtr(0)->findAverageLocalCoordinates();
        MeasurementPoint mp1 = aL1TkStub.getClusterPtr(1)->findAverageLocalCoordinates();
        aDisplacement = 2*(mp1.x() - mp0.x()); /// In HALF-STRIP units!

        /// By default, assigned as ZERO
        anOffset = 0;

        delete rayEndpoints;
      }
    }
    else
      delete rayEndpoints;
  }



/** ********************** **/
/**                        **/
/**   DECLARATION OF THE   **/
/**    ALGORITHM TO THE    **/
/**       FRAMEWORK        **/
/**                        **/
/** ********************** **/

template < typename T >
class  ES_HitMatchingAlgorithm_pixelray : public edm::ESProducer
{
  private:
    /// Data members
    boost::shared_ptr< HitMatchingAlgorithm< T > > _theAlgo;
    double mPtThreshold;
    double mIPWidth;

  public:
    /// Constructor
    ES_HitMatchingAlgorithm_pixelray( const edm::ParameterSet & p )
      : mPtThreshold( p.getParameter< double >("minPtThreshold") ),
        mIPWidth( p.getParameter< double >("ipWidth") )
    {
      setWhatProduced(this);
    }

    /// Destructor
    virtual ~ES_HitMatchingAlgorithm_pixelray(){}

    /// Implement the producer
    boost::shared_ptr< HitMatchingAlgorithm< T > > produce( const HitMatchingAlgorithmRecord & record )
    {
      /// Get magnetic field
      edm::ESHandle< MagneticField > magnet;
      record.getRecord< IdealMagneticFieldRecord >().get(magnet);
      double mMagneticFieldStrength = magnet->inTesla(GlobalPoint(0,0,0)).z();

      /// Calculate scaling factor based on B and Pt threshold
      double mCompatibilityScalingFactor = (CLHEP::c_light * mMagneticFieldStrength) / (100.0 * 2.0e+9 * mPtThreshold);

      edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
      record.getRecord<StackedTrackerGeometryRecord>().get( StackedTrackerGeomHandle );

      HitMatchingAlgorithm< T >* HitMatchingAlgo =
        new HitMatchingAlgorithm_pixelray< T >( &(*StackedTrackerGeomHandle),
                                                             mCompatibilityScalingFactor,
                                                             mIPWidth );

      _theAlgo = boost::shared_ptr< HitMatchingAlgorithm< T > >( HitMatchingAlgo );
      return _theAlgo;
    }

}; /// Close class

#endif

