/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
///                                      ///
/// Nicola Pozzobon,  UNIPD              ///
///                                      ///
/// 2012, August, October                ///
/// ////////////////////////////////////////

#ifndef HIT_MATCHING_ALGORITHM_window2012_H
#define HIT_MATCHING_ALGORITHM_window2012_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithm.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithmRecord.h"


#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"

#include "Geometry/CommonTopologies/interface/Topology.h" 
#include "CLHEP/Units/PhysicalConstants.h"

#include <boost/shared_ptr.hpp>
#include <memory>
#include <string>
#include <map>
#include <typeinfo>

  /** ************************ **/
  /**                          **/
  /**   DECLARATION OF CLASS   **/
  /**                          **/
  /** ************************ **/

  template< typename T >
  class HitMatchingAlgorithm_window2012 : public HitMatchingAlgorithm< T >
  {
    private:
      /// Data members
      double                       mPtScalingFactor;
      std::string className_;

    public:
      /// Constructor
      HitMatchingAlgorithm_window2012( const StackedTrackerGeometry *aStackedTracker,
                                       double aPtScalingFactor )
        : HitMatchingAlgorithm< T >( aStackedTracker,__func__ )
      {
        mPtScalingFactor = aPtScalingFactor;
      }

      /// Destructor
      ~HitMatchingAlgorithm_window2012(){}

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
void HitMatchingAlgorithm_window2012< T >::CheckTwoMemberHitsForCompatibility( bool &aConfirmation, int &aDisplacement, int &anOffset, const L1TkStub< T > &aL1TkStub ) const
{
  /// Calculate average coordinates col/row for inner/outer Cluster
  /// These are already corrected for being at the center of each pixel
  MeasurementPoint mp0 = aL1TkStub.getClusterPtr(0)->findAverageLocalCoordinates();
  MeasurementPoint mp1 = aL1TkStub.getClusterPtr(1)->findAverageLocalCoordinates();

  /// Get the module position in global coordinates
  StackedTrackerDetId stDetId( aL1TkStub.getDetId() );
  const GeomDetUnit* det0 = HitMatchingAlgorithm< T >::theStackedTracker->idToDetUnit( stDetId, 0 );
  const GeomDetUnit* det1 = HitMatchingAlgorithm< T >::theStackedTracker->idToDetUnit( stDetId, 1 );

  /// Find pixel pitch and topology related information
  const PixelGeomDetUnit* pix0 = dynamic_cast< const PixelGeomDetUnit* >( det0 );
  const PixelGeomDetUnit* pix1 = dynamic_cast< const PixelGeomDetUnit* >( det1 );
  const PixelTopology* top0 = dynamic_cast< const PixelTopology* >( &(pix0->specificTopology()) );
  const PixelTopology* top1 = dynamic_cast< const PixelTopology* >( &(pix1->specificTopology()) );
  std::pair< float, float > pitch0 = top0->pitch();
  std::pair< float, float > pitch1 = top1->pitch();
  
  /// Get the Stack radius and z and displacements
  double R0 = det0->position().perp();
  double R1 = det1->position().perp();
  double Z0 = det0->position().z();
  double Z1 = det1->position().z();

  double DR = R1-R0;
  double DZ = Z1-Z0;

  /// Scale factor is already present in
  /// double mPtScalingFactor = (floor(mMagneticFieldStrength*10.0 + 0.5))/10.0*0.0015/mPtThreshold;
  /// hence the formula iis something like
  /// displacement < Delta * 1 / sqrt( ( 1/(mPtScalingFactor*R) )** 2 - 1 )
  double denominator = sqrt( 1/( mPtScalingFactor*mPtScalingFactor*R0*R0 ) - 1 );

  if (stDetId.isBarrel())
  {
    /// All of these are calculated in terms of pixels in outer sensor
    /// 0) Calculate window in terms of multiples of outer sensor pitch
    int window = floor( (DR/denominator) / pitch1.first ) + 1;
    /// POSITION IN TERMS OF PITCH MULTIPLES:
    ///       0 1 2 3 4 5 5 6 8 9 ...
    /// COORD: 0 1 2 3 4 5 6 7 8 9 ...
    /// OUT   | | | | | |x| | | | | | | | | |
    ///
    /// IN    | | | |x|x| | | | | | | | | | |
    ///             THIS is 3.5 (COORD) and 4.0 (POS)
    /// 1) disp is the difference between average row coordinates
    ///    in inner and outer stack member, in terms of outer member pitch
    ///    (in case they are the same, this is just a plain coordinate difference)
    double dispD = mp1.x() - mp0.x() * (pitch0.first / pitch1.first);
    int dispI = ((dispD>0)-(dispD<0))*floor(fabs(dispD));
    /// 2) offset is the projection with a straight line of the innermost
    ///    hit towards the ourermost stack member, still in terms of outer member pitch
    ///    NOTE: in terms of coordinates, the center of the module is at NROWS/2-0.5 to
    ///    be consistent with the definition given above 
    double offsetD = DR/R0 * ( mp0.x() - (top0->nrows()/2 - 0.5) ) * (pitch0.first / pitch1.first);
    int offsetI = ((offsetD>0)-(offsetD<0))*floor(fabs(offsetD));

    /// Accept the stub if the post-offset correction displacement is smaller than the half-window
    if ( abs(dispI - offsetI) < window )
    {
/*
      double DP = fabs((dispI-offsetI) * pitch1.first / R0) + 0.00000000001;
      double roughPt = DR * mPtFactor / DP;
      a.first = true;
      a.second = roughPt;
*/

      /// Accept the stub if the rough backprojection is fine
//      if (fabs(Z0 - R0*DZ/DR)<200) {
// NON SI PUO` perche' Z0, R0 ecc sono le coordinate dei MODULI

        aConfirmation = true;
        aDisplacement = 2*dispI; /// In HALF-STRIP units!
        anOffset = 2*offsetI; /// In HALF-STRIP units!

//      }

    } /// End of stub is accepted
  }
  else if (stDetId.isEndcap())
  {
    /// All of these are calculated in terms of pixels in outer sensor
    /// 0) Calculate window in terms of multiples of outer sensor pitch
    int window = floor( R0/Z0 * (DZ/denominator) / pitch1.first ) + 1;
    /// 1) disp is the difference between average row coordinates
    ///    in inner and outer stack member, in terms of outer member pitch
    ///    (in case they are the same, this is just a plain coordinate difference)
    double dispD = mp1.x() - mp0.x() * (pitch0.first / pitch1.first);
    int dispI = ((dispD>0)-(dispD<0))*floor(fabs(dispD));
    /// 2) offset is the projection with a straight line of the innermost
    ///    hit towards the ourermost stack member, still in terms of outer member pitch
    ///    NOTE: in terms of coordinates, the center of the module is at NROWS/2-0.5 to
    ///    be consistent with the definition given above 
    double offsetD = DZ/Z0 * ( mp0.x() - (top0->nrows()/2 - 0.5) ) * (pitch0.first / pitch1.first);
    int offsetI = ((offsetD>0)-(offsetD<0))*floor(fabs(offsetD));

    /// Accept the stub if the post-offset correction displacement is smaller than the half-window
    if ( abs(dispI - offsetI) < window )
    {
/*
      double DP = fabs((dispI-offsetI) * pitch1.first / R0) + 0.00000000001;
      double roughPt = DZ * R0 / Z0 * mPtFactor / DP;
      a.first = true;
      a.second = roughPt;
*/

        aConfirmation = true;
        aDisplacement = 2*dispI; /// In HALF-STRIP units!
        anOffset = 2*offsetI; /// In HALF-STRIP units!

    } /// End of stub is accepted
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
class ES_HitMatchingAlgorithm_window2012 : public edm::ESProducer
{
  private:
    /// Data members
    boost::shared_ptr< HitMatchingAlgorithm< T > > _theAlgo;
    double mPtThreshold;
    double mIPWidth;

  public:
    /// Constructor
    ES_HitMatchingAlgorithm_window2012( const edm::ParameterSet & p )
      : mPtThreshold( p.getParameter< double >("minPtThreshold") )
    //                                mIPWidth( p.getParameter<double>("ipWidth") ),
    {
      setWhatProduced( this );
    }

    /// Destructor
    virtual ~ES_HitMatchingAlgorithm_window2012(){}

    /// Implement the producer
    boost::shared_ptr< HitMatchingAlgorithm< T > > produce( const HitMatchingAlgorithmRecord & record )
    { 
      /// Get magnetic field
      edm::ESHandle< MagneticField > magnet;
      record.getRecord< IdealMagneticFieldRecord >().get(magnet);
      double mMagneticFieldStrength = magnet->inTesla(GlobalPoint(0,0,0)).z();

      /// Calculate scaling factor based on B and Pt threshold
      //double mPtScalingFactor = 0.0015*mMagneticFieldStrength/mPtThreshold;
      //double mPtScalingFactor = (CLHEP::c_light * mMagneticFieldStrength) / (100.0 * 2.0e+9 * mPtThreshold);
      double mPtScalingFactor = (floor(mMagneticFieldStrength*10.0 + 0.5))/10.0*0.0015/mPtThreshold;

      edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
      record.getRecord< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );
  
      HitMatchingAlgorithm< T >* HitMatchingAlgo =
        new HitMatchingAlgorithm_window2012< T >( &(*StackedTrackerGeomHandle),
                                                               mPtScalingFactor );

      _theAlgo = boost::shared_ptr< HitMatchingAlgorithm< T > >( HitMatchingAlgo );
      return _theAlgo;
    } 

};

#endif

