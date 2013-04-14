/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
///                                      ///
/// Anders Ryd, Cornell                  ///
/// Emmanuele Salvati, Cornell           ///
/// Nicola Pozzobon,  UNIPD              ///
///                                      ///
/// 2012                                 ///
/// 2013, January                        ///
/// ////////////////////////////////////////

#ifndef TRACKING_ALGO_PROVA_H
#define TRACKING_ALGO_PROVA_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/TrackingAlgorithm.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/TrackingAlgorithmRecord.h"

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
  class TrackingAlgorithm_PROVA : public TrackingAlgorithm< T >
  {
    private :
      typedef std::pair< unsigned int, unsigned int >                                    StubKey;     /// This is the key
      typedef std::map< StubKey, std::vector< edm::Ptr< L1TkStub< T > > > > L1TkStubMap; /// This is the map

      /// Data members
      double                       mMagneticField;

    public:
      /// Constructors
      TrackingAlgorithm_PROVA( const StackedTrackerGeometry *aStackedGeom,
                               double aMagneticField )
        : TrackingAlgorithm< T > ( aStackedGeom, __func__ )
      {
        mMagneticField = aMagneticField;
      }

      /// Destructor
      ~TrackingAlgorithm_PROVA(){}

      /// Seed creation
      void CreateSeeds( std::vector< L1TkTrack< T > > &output, edm::Handle< std::vector< L1TkStub< T > > > &input ) const;

      /// Match a Stub to a Seed/Track
      void AttachStubToSeed( L1TkTrack< T > &seed, edm::Ptr< L1TkStub< T > > &candidate ) const;

  }; /// Close class



/** ***************************** **/
/**                               **/
/**   IMPLEMENTATION OF METHODS   **/
/**                               **/
/** ***************************** **/

template< typename T >
void TrackingAlgorithm_PROVA< T >::CreateSeeds( std::vector< L1TkTrack< T > > &output, edm::Handle< std::vector< L1TkStub< T > > > &input ) const
{
  /// Prepare output
  output.clear();

  /// Map the Barrel Stubs per layer/rod
  L1TkStubMap stubBarrelMap;
  stubBarrelMap.clear();

  typename std::vector< L1TkStub< T > >::const_iterator inputIter;
  unsigned int j = 0; /// Counter needed to build the edm::Ptr to the L1TkStub

  for ( inputIter = input->begin();
        inputIter != input->end();
        ++inputIter )
  {
    /// Make the pointer to be put in the map and, later on, in the Track
    edm::Ptr< L1TkStub< T > > tempStubPtr( input, j++ );

    StackedTrackerDetId detIdStub( inputIter->getDetId() );
    if ( detIdStub.isBarrel() )
    {
      /// Build the key to the map (by DoubleStack)
      StubKey mapkey = std::make_pair( detIdStub.iLayer()/2 + 1, detIdStub.iPhi() );

      /// If an entry already exists for this key, just add the cluster
      /// to the vector, otherwise create the entry
      if ( stubBarrelMap.find( mapkey ) == stubBarrelMap.end() )
      {
        /// New entry
        std::vector< edm::Ptr< L1TkStub< T > > > tempStubVec;
        tempStubVec.clear();
        tempStubVec.push_back( tempStubPtr );
        stubBarrelMap.insert( std::pair< StubKey, std::vector< edm::Ptr< L1TkStub< T > > > > ( mapkey, tempStubVec ) );
      }
      else
      {
        /// Already existing entry
        stubBarrelMap[mapkey].push_back( tempStubPtr );
      }
    }
  }

  /// Loop over the map
  /// Create Seeds and map detectors
  typename L1TkStubMap::const_iterator mapIter;
  for ( mapIter = stubBarrelMap.begin();
        mapIter != stubBarrelMap.end();
        ++mapIter )
  {
    /// Here we have ALL the stubs in one single Hermetic ROD (if applicable) 
    std::vector< edm::Ptr< L1TkStub< T > > > tempStubVec = mapIter->second;

    for ( unsigned int i = 0; i < tempStubVec.size(); i++ )
    {
      StackedTrackerDetId detId1( tempStubVec.at(i)->getDetId() );
      GlobalPoint pos1 = TrackingAlgorithm< T >::theStackedTracker->findAverageGlobalPosition( tempStubVec.at(i)->getClusterPtr(0).get()  );

      for ( unsigned int k = i+1; k < tempStubVec.size(); k++ )
      {
        StackedTrackerDetId detId2( tempStubVec.at(k)->getDetId() );
        GlobalPoint pos2 = TrackingAlgorithm< T >::theStackedTracker->findAverageGlobalPosition( tempStubVec.at(k)->getClusterPtr(0).get() );

        /// Skip different rod pairs
        if ( detId1.iPhi() != detId2.iPhi() ) continue;
        /// Skip same layer pairs
        if ( detId1.iLayer()+1 != detId2.iLayer() ) continue;

        /// NOTE This approach requires global coordinates

        /// Perform standard trigonometric operations
        double deltaPhi = pos2.phi() - pos1.phi();
        if ( fabs(deltaPhi) >= M_PI )
        {
          if ( deltaPhi>0 )
            deltaPhi = deltaPhi - 2*M_PI;
          else
            deltaPhi = 2*M_PI + deltaPhi;
        }

        double distance = sqrt( pos2.perp2() + pos1.perp2() - 2*pos2.perp()*pos1.perp()*cos(deltaPhi) );
        double rInvOver2 = sin(deltaPhi)/distance; /// Sign is mantained to keep track of the charge

        /// Perform cut on Pt
        //double seedPt = mMagneticField*0.0015 / rInvOver2;
        //if ( seedPt < 2.0 ) continue;
        if ( fabs(rInvOver2) > mMagneticField*0.0015/2.0 ) continue;

/*
            double vertexphi = acos(outerPointRadius/twoRadius);
            vertexphi = outerPointPhi - charge*vertexphi;
            vertexphi = vertexphi + charge*0.5*M_PI;
            if ( vertexphi > M_PI ) vertexphi -= 2*M_PI;
            else if ( vertexphi <= -M_PI ) vertexphi += 2*M_PI;
            tempShortTracklet.setMomentum( GlobalVector( roughPt*cos(vertexphi),
                                                         roughPt*sin(vertexphi),
                                                         roughPt*(outerStubPosition.z()-innerStubPosition.z())/deltaRadius ) );
*/

        /// Calculate projected vertex
        double rhoPsi1 = asin( pos1.perp()*rInvOver2 )/rInvOver2;
        double rhoPsi2 = asin( pos2.perp()*rInvOver2 )/rInvOver2;
        double tanTheta0 = ( pos1.z() - pos2.z() ) / ( rhoPsi1 - rhoPsi2 );
        double z0 = pos2.z() - rhoPsi2 * tanTheta0;

        /// Perform projected vertex cut
        if ( fabs(z0) > 30.0 ) continue;

        /// Calculate direction at vertex
        double phi0 = pos2.phi() + asin( pos2.perp() * rInvOver2 );

        /// Calculate Pt
        double roughPt = fabs( mMagneticField*0.0015 / rInvOver2 );

        /// Create the Seed in the form of a Track and store it in the output
        std::vector< edm::Ptr< L1TkStub< T > > > tempVec;
        tempVec.push_back( tempStubVec.at(i) );
        tempVec.push_back( tempStubVec.at(k) );
        L1TkTrack< T > tempTrack( tempVec );
        tempTrack.setRInv( 2*rInvOver2 );
        tempTrack.setMomentum( GlobalVector( roughPt*cos(phi0),
                                             roughPt*sin(phi0),
                                             roughPt*tanTheta0 ) );
        output.push_back( tempTrack );
      }
    } /// End of double loop over pairs of stubs

  } /// End of loop over map elements
}

/// Match a Stub to a Seed/Track
template< typename T >
void TrackingAlgorithm_PROVA< T >::AttachStubToSeed( L1TkTrack< T > &seed, edm::Ptr< L1TkStub< T > > &candidate ) const
{
  /// Get the track Stubs
  std::vector< edm::Ptr< L1TkStub< T > > > theStubs = seed.getStubPtrs();

  /// Check that the candidate is NOT the one under examination
  for ( unsigned int i = 0; i < theStubs.size(); i++ )
  {
    if ( theStubs.at(i) == candidate )
      return;
  }

  /// Get the track momentum and propagate it
  GlobalVector curMomentum = seed.getMomentum();
  double curRInv = seed.getRInv();

  /// Get the candidate Stub position
  GlobalPoint candPos = candidate->getClusterPtr(0)->findAverageGlobalPosition( &(*TrackingAlgorithm< T >::theStackedTracker) );

  /// Propagate
  double propPhi = curMomentum.phi() - asin ( 0.5 * candPos.perp() / curRInv );

  /// Calculate displacement
  /// Perform standard trigonometric operations
  double deltaPhi = propPhi - candPos.phi();
  if ( fabs(deltaPhi) >= M_PI )
  {
    if ( deltaPhi>0 )
      deltaPhi = deltaPhi - 2*M_PI;
    else
      deltaPhi = 2*M_PI + deltaPhi;
  }
  double deltaRPhi = fabs( deltaPhi * candPos.perp() );







}


/** ********************** **/
/**                        **/
/**   DECLARATION OF THE   **/
/**    ALGORITHM TO THE    **/
/**       FRAMEWORK        **/
/**                        **/
/** ********************** **/

template< typename T >
class ES_TrackingAlgorithm_PROVA : public edm::ESProducer
{
  private:
    /// Data members
    boost::shared_ptr< TrackingAlgorithm< T > > _theAlgo;

  public:
    /// Constructor
    ES_TrackingAlgorithm_PROVA( const edm::ParameterSet & p )
    //  : mPtThreshold( p.getParameter< double >("minPtThreshold") )
    //                                mIPWidth( p.getParameter<double>("ipWidth") ),
    {
      setWhatProduced( this );
    }

    /// Destructor
    virtual ~ES_TrackingAlgorithm_PROVA() {}

    /// Implement the producer
    boost::shared_ptr< TrackingAlgorithm< T > > produce( const TrackingAlgorithmRecord & record )
    {
      /// Get magnetic field
      edm::ESHandle< MagneticField > magnet;
      record.getRecord< IdealMagneticFieldRecord >().get(magnet);
      double mMagneticFieldStrength = magnet->inTesla(GlobalPoint(0,0,0)).z();
      double mMagneticFieldRounded = (floor(mMagneticFieldStrength*10.0 + 0.5))/10.0;

      /// Calculate scaling factor based on B and Pt threshold
      //double mPtScalingFactor = 0.0015*mMagneticFieldStrength/mPtThreshold;
      //double mPtScalingFactor = (KGMS_C * mMagneticFieldStrength) / (100.0 * 2.0e+9 * mPtThreshold);
      //double mPtScalingFactor = (floor(mMagneticFieldStrength*10.0 + 0.5))/10.0*0.0015/mPtThreshold;

      edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
      record.getRecord< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );

      TrackingAlgorithm< T >* TrackingAlgo =
        new TrackingAlgorithm_PROVA< T >( &(*StackedTrackerGeomHandle),
                                                       mMagneticFieldRounded );

      _theAlgo = boost::shared_ptr< TrackingAlgorithm< T > >( TrackingAlgo );
      return _theAlgo;
    }

};

#endif

