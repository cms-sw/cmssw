/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
///                                      ///
/// Written by:                          ///
/// Andrew W. Rose, IC                   ///
/// Nicola Pozzobon, UNIPD               ///
///                                      ///
/// 2008                                 ///
/// 2010, May                            ///
/// 2011, June                           ///
/// ////////////////////////////////////////

#ifndef HIT_MATCHING_ALGORITHM_globalgeometry_H
#define HIT_MATCHING_ALGORITHM_globalgeometry_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "CLHEP/Units/PhysicalConstants.h"
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
  class HitMatchingAlgorithm_globalgeometry : public HitMatchingAlgorithm< T >
  {
    private:
      /// Data members
      double                       mCompatibilityScalingFactor;
      double                       mIPWidth;

    public:
      /// Constructor
      HitMatchingAlgorithm_globalgeometry( const StackedTrackerGeometry *aStackedTracker,
                                           double aCompatibilityScalingFactor,
                                           double aIPWidth )
        : HitMatchingAlgorithm< T >( aStackedTracker,__func__ )
      {
        mCompatibilityScalingFactor = aCompatibilityScalingFactor;
        mIPWidth = aIPWidth;
      }

      /// Destructor
      ~HitMatchingAlgorithm_globalgeometry(){}

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
  void HitMatchingAlgorithm_globalgeometry< T >::CheckTwoMemberHitsForCompatibility( bool &aConfirmation, int &aDisplacement, int &anOffset, const L1TkStub< T > &aL1TkStub ) const
  { 
    /// Convert DetId
    StackedTrackerDetId stDetId( aL1TkStub.getDetId() );

    /// Force this to be a BARREL-only algorithm
    if ( stDetId.isEndcap() )
    {
      aConfirmation = false;
      return;
    }

    /// Get average position of Clusters composing the Stub
    GlobalPoint innerHitPosition = (*HitMatchingAlgorithm< T >::theStackedTracker).findAverageGlobalPosition( aL1TkStub.getClusterPtr(0).get() );
    GlobalPoint outerHitPosition = (*HitMatchingAlgorithm< T >::theStackedTracker).findAverageGlobalPosition( aL1TkStub.getClusterPtr(1).get() );

    /// Get useful quantities
    double outerPointRadius = outerHitPosition.perp();
    double innerPointRadius = innerHitPosition.perp();
    double outerPointPhi = outerHitPosition.phi();
    double innerPointPhi = innerHitPosition.phi();

    /// Check for seed compatibility given a pt cut
    /// Threshold computed from radial location of hits
    double deltaRadius = outerPointRadius - innerPointRadius;
    double deltaPhiThreshold = deltaRadius * mCompatibilityScalingFactor;  

    /// Calculate angular displacement from hit phi locations
    /// and renormalize it, if needed
    double deltaPhi = outerPointPhi - innerPointPhi;
    if ( deltaPhi < 0 ) deltaPhi = -deltaPhi;
    if ( deltaPhi > M_PI ) deltaPhi = 2*M_PI - deltaPhi;

    /// Apply selection based on Pt
    if ( deltaPhi < deltaPhiThreshold )
    {
      /// Check for backprojection to beamline
      double innerPointZ = innerHitPosition.z();
      double outerPointZ = outerHitPosition.z();
      double positiveZBoundary =  (mIPWidth - outerPointZ) * deltaRadius;
      double negativeZBoundary = -(mIPWidth + outerPointZ) * deltaRadius;
      double multipliedLocation = (innerPointZ - outerPointZ) * outerPointRadius;

      /// Apply selection based on backprojected Z
      if ( (multipliedLocation < positiveZBoundary) && (multipliedLocation > negativeZBoundary) )
      {  
        aConfirmation = true;

        /// Calculate output
        /// NOTE this assumes equal pitch in both sensors!
        MeasurementPoint mp0 = aL1TkStub.getClusterPtr(0)->findAverageLocalCoordinates(); 
        MeasurementPoint mp1 = aL1TkStub.getClusterPtr(1)->findAverageLocalCoordinates();                                  
        aDisplacement = 2*(mp1.x() - mp0.x()); /// In HALF-STRIP units!

        /// By default, assigned as ZERO
        anOffset = 0;

      } /// End of selection based on Z
    } /// End of selection based on Pt
  }



/** ********************** **/
/**                        **/
/**   DECLARATION OF THE   **/
/**    ALGORITHM TO THE    **/
/**       FRAMEWORK        **/
/**                        **/
/** ********************** **/

template< typename T >
class  ES_HitMatchingAlgorithm_globalgeometry : public edm::ESProducer
{
  private:
    /// Data members
    boost::shared_ptr< HitMatchingAlgorithm< T > > _theAlgo;
    double mPtThreshold;
    double mIPWidth;

  public:
    /// Constructor
    ES_HitMatchingAlgorithm_globalgeometry( const edm::ParameterSet & p ) :
                                            mPtThreshold( p.getParameter< double >("minPtThreshold") ),
                                            mIPWidth( p.getParameter< double >("ipWidth") )
    {
      setWhatProduced( this );
    }

    /// Destructor
    virtual ~ES_HitMatchingAlgorithm_globalgeometry(){}

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
      record.getRecord< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );
  
      HitMatchingAlgorithm< T >* HitMatchingAlgo =
        new HitMatchingAlgorithm_globalgeometry< T >( &(*StackedTrackerGeomHandle),
                                                                   mCompatibilityScalingFactor,
                                                                   mIPWidth );

      _theAlgo = boost::shared_ptr< HitMatchingAlgorithm< T > >( HitMatchingAlgo );
      return _theAlgo;
    }

}; /// Close class

#endif

