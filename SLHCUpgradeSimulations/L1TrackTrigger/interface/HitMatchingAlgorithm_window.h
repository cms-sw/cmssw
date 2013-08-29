/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
///                                      ///
/// Andrew W. Rose, IC                   ///
/// Nicola Pozzobon,  UNIPD              ///
///                                      ///
/// 2008                                 ///
/// 2010, May                            ///
/// 2011, June                           ///
/// ////////////////////////////////////////

#ifndef HIT_MATCHING_ALGORITHM_window_H
#define HIT_MATCHING_ALGORITHM_window_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithm.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithmRecord.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/WindowFinder.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "Geometry/CommonTopologies/interface/Topology.h" 

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
  class HitMatchingAlgorithm_window : public HitMatchingAlgorithm< T >
  {
    private:
      /// Data members
      WindowFinder    *mWindowFinder;

    public:
      /// Constructor
      HitMatchingAlgorithm_window( const StackedTrackerGeometry *aStackedTracker,
                                   double aPtScalingFactor,
                                   double aIPwidth,
                                   double aRowResolution,
                                   double aColResolution )
        : HitMatchingAlgorithm< T >( aStackedTracker, __func__ ),
          mWindowFinder( new WindowFinder( aStackedTracker,
					   aPtScalingFactor,
					   aIPwidth,
					   aRowResolution,
					   aColResolution ) ) {}

      /// Destructor
      ~HitMatchingAlgorithm_window(){}

      /// Matching operations
      void CheckTwoMemberHitsForCompatibility( bool &aConfirmation, int &aDisplacement, int &anOffset, const L1TkStub< T > &aL1TkStub ) const;


  }; /// Close class



/** ***************************** **/
/**                               **/
/**   IMPLEMENTATION OF METHODS   **/
/**                               **/
/** ***************************** **/

/// Matching operations
/// Default is for PixelDigis
template< typename T >
void HitMatchingAlgorithm_window< T >::CheckTwoMemberHitsForCompatibility( bool &aConfirmation, int &aDisplacement, int &anOffset, const L1TkStub< T > &aL1TkStub ) const
{
  /// Convert DetId
  StackedTrackerDetId stDetId( aL1TkStub.getDetId() );

  //move this out of the if to ensure that it gets set to something regardless
  aConfirmation = false;

  /// Force this to be a BARREL-only algorithm
  if ( stDetId.isEndcap() ) return;

  typename std::vector< T >::const_iterator hitIter;

  /// Calculate average coordinates col/row for inner Cluster
  double averageRow = 0.0;
  double averageCol = 0.0;
  const std::vector< T > &lhits0 = aL1TkStub.getClusterPtr(0)->getHits();
  if ( lhits0.size() != 0 )
  {
    for ( hitIter = lhits0.begin();
          hitIter != lhits0.end();
          hitIter++ )
    {
      averageRow +=  (**hitIter).row();
      averageCol +=  (**hitIter).column();
    }
    averageRow /= lhits0.size();
    averageCol /= lhits0.size();
  }

  /// Calculate window based on the average row and column
  StackedTrackerWindow window = mWindowFinder->getWindow( stDetId, averageRow, averageCol );

  /// Calculate average coordinates col/row for outer Cluster
  averageRow = 0.0;
  averageCol = 0.0;
  const std::vector< T > &lhits1 = aL1TkStub.getClusterPtr(1)->getHits();
  if ( lhits1.size() != 0 )
  {
    for ( hitIter = lhits1.begin();
          hitIter != lhits1.end();
          hitIter++ )
    {
      averageRow += (**hitIter).row();
      averageCol +=  (**hitIter).column();
    }
    averageRow /= lhits1.size();
    averageCol /= lhits1.size();
  }

  /// Check if the window criteria are satisfied
  if ( ( averageRow >= window.mMinrow ) && ( averageRow <= window.mMaxrow ) &&
       ( averageCol >= window.mMincol ) && ( averageCol <= window.mMaxcol ) )
    {
      aConfirmation = true;
      
      /// Calculate output
      /// NOTE this assumes equal pitch in both sensors!
      MeasurementPoint mp0 = aL1TkStub.getClusterPtr(0)->findAverageLocalCoordinates();
      MeasurementPoint mp1 = aL1TkStub.getClusterPtr(1)->findAverageLocalCoordinates();
      aDisplacement = 2*(mp1.x() - mp0.x()); /// In HALF-STRIP units!
            
      /// By default, assigned as ZERO
      anOffset = 0;
    }

}

/** ********************** **/
/**                        **/
/**   DECLARATION OF THE   **/
/**    ALGORITHM TO THE    **/
/**       FRAMEWORK        **/
/**                        **/
/** ********************** **/

template< typename T >
class ES_HitMatchingAlgorithm_window : public edm::ESProducer
{
  private:
    /// Data members
    boost::shared_ptr< HitMatchingAlgorithm< T > > _theAlgo;
    double mPtThreshold;
    double mIPWidth;
    double mRowResolution;
    double mColResolution;

  public:
    /// Constructor
    ES_HitMatchingAlgorithm_window( const edm::ParameterSet & p )
      : mPtThreshold( p.getParameter< double >("minPtThreshold") ),
        mIPWidth( p.getParameter< double >("ipWidth") ),
        mRowResolution( p.getParameter< double >("RowResolution") ),
        mColResolution( p.getParameter< double >("ColResolution") )
    {
      setWhatProduced( this );
    }

    /// Destructor
    virtual ~ES_HitMatchingAlgorithm_window(){}

    /// Implement the producer
    boost::shared_ptr< HitMatchingAlgorithm< T > > produce( const HitMatchingAlgorithmRecord & record )
    { 
      /// Get magnetic field
      edm::ESHandle< MagneticField > magnet;
      record.getRecord< IdealMagneticFieldRecord >().get(magnet);
      double mMagneticFieldStrength = magnet->inTesla(GlobalPoint(0,0,0)).z();

      /// Calculate scaling factor based on B and Pt threshold
      double mPtScalingFactor = 0.0015*mMagneticFieldStrength/mPtThreshold;

      edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
      record.getRecord< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );
  
      HitMatchingAlgorithm< T >* HitMatchingAlgo =
        new HitMatchingAlgorithm_window< T >( &(*StackedTrackerGeomHandle),
                                                           mPtScalingFactor,
                                                           mIPWidth,
                                                           mRowResolution,
                                                           mColResolution );

      _theAlgo = boost::shared_ptr< HitMatchingAlgorithm< T > >( HitMatchingAlgo );
      return _theAlgo;
    } 

};

#endif

