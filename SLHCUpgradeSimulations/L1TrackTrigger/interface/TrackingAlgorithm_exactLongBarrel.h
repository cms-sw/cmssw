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

#ifndef TRACKING_ALGO_EXACTLB_H
#define TRACKING_ALGO_EXACTLB_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/TrackingAlgorithm.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/TrackingAlgorithmRecord.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

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

template< typename T >
class TrackingAlgorithm_exactLongBarrel : public TrackingAlgorithm< T >
{
  private :
    /// Data members
    double                       mMagneticField;
    unsigned int                 nSectors;
    unsigned int                 nWedges;

    std::vector< std::vector< double > > tableRPhi;
    std::vector< std::vector< double > > tableZ;

  public:
    /// Constructors
    TrackingAlgorithm_exactLongBarrel( const StackedTrackerGeometry *aStackedGeom,
                                       double aMagneticField, unsigned int aSectors, unsigned int aWedges,
                                       std::vector< std::vector< double > > aTableRPhi,
                                       std::vector< std::vector< double > > aTableZ )
      : TrackingAlgorithm< T > ( aStackedGeom, __func__ )
    {
      mMagneticField = aMagneticField;
      nSectors = aSectors;
      nWedges = aWedges;

      tableRPhi = aTableRPhi;
      tableZ = aTableZ;
    }

    /// Destructor
    ~TrackingAlgorithm_exactLongBarrel(){}

    /// Seed creation
    void CreateSeeds( std::vector< L1TkTrack< T > > &output,
                      std::map< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< L1TkStub< T > > > > *outputSectorMap,
                      edm::Handle< std::vector< L1TkStub< T > > > &input ) const;

    /// Match a Stub to a Seed/Track
    void AttachStubToSeed( L1TkTrack< T > &seed, edm::Ptr< L1TkStub< T > > &candidate ) const;

    /// Return the number of Sectors
    unsigned int ReturnNumberOfSectors() const { return nSectors; } /// Phi
    unsigned int ReturnNumberOfWedges() const { return nWedges; } /// Eta

    /// Return the value of the magnetic field
    double ReturnMagneticField() const { return mMagneticField; }

    /// Fit the Track
    /// Take it from the parent class

}; /// Close class

/** ***************************** **/
/**                               **/
/**   IMPLEMENTATION OF METHODS   **/
/**                               **/
/** ***************************** **/

template< typename T >
void TrackingAlgorithm_exactLongBarrel< T >::CreateSeeds( std::vector< L1TkTrack< T > > &output,
                                                          std::map< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< L1TkStub< T > > > > *outputSectorMap,
                                                          edm::Handle< std::vector< L1TkStub< T > > > &input ) const
{
  /// Prepare output
  output.clear();

  /// Map the Barrel Stubs per layer/rod
  std::map< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< L1TkStub< T > > > > stubBarrelMap;
  stubBarrelMap.clear();

  /// Map the Barrel Stubs per sector
  outputSectorMap->clear();

  typename std::vector< L1TkStub< T > >::const_iterator inputIter;
  unsigned int j = 0; /// Counter needed to build the edm::Ptr to the L1TkStub

  for ( inputIter = input->begin();
        inputIter != input->end();
        ++inputIter )
  {
    /// Make the pointer to be put in the map and, later on, in the Track
    edm::Ptr< L1TkStub< T > > tempStubPtr( input, j++ );

    /// Calculate Sector
    /// From 0 to nSectors-1
    /// Sector 0 centered on Phi = 0 and symmetric around it
    double stubPhi = TrackingAlgorithm< T >::theStackedTracker->findGlobalPosition( tempStubPtr.get() ).phi();
    stubPhi += M_PI/nSectors;
    if ( stubPhi < 0 )
    {
      stubPhi += 2*M_PI;
    }
    unsigned int thisSector = floor( 0.5*stubPhi*nSectors/M_PI );
    /// Calculate Wedge
    /// From 0 to nWedges-1
    /// 
    double stubEta = TrackingAlgorithm< T >::theStackedTracker->findGlobalPosition( tempStubPtr.get() ).eta();
    stubEta += 2.5; /// bring eta = -2.5 to 0
    //stubEta += 2.5/nWedges;

    /// Accept only stubs within -2.5, 2.5 range
    if ( stubEta < 0.0 || stubEta > 5.0 )
      continue;

    unsigned int thisWedge = floor( stubEta*nWedges/5.0 );

    /// Build the key to the map (by Sector / Wedge)
    std::pair< unsigned int, unsigned int > mapkey = std::make_pair( thisSector, thisWedge );

    StackedTrackerDetId detIdStub( inputIter->getDetId() );
    if ( detIdStub.isBarrel() )
    {
      /// If an entry already exists for this key, just add the stub
      /// to the vector, otherwise create the entry
      if ( stubBarrelMap.find( mapkey ) == stubBarrelMap.end() )
      {
        /// New entry
        std::vector< edm::Ptr< L1TkStub< T > > > tempStubVec;
        tempStubVec.clear();
        tempStubVec.push_back( tempStubPtr );
        stubBarrelMap.insert( std::pair< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< L1TkStub< T > > > > ( mapkey, tempStubVec ) );
      }
      else
      {
        /// Already existing entry
        stubBarrelMap[mapkey].push_back( tempStubPtr );
      }

      /// If an entry already exists for this Sector, just add the stub
      /// to the vector, otherwise create the entry
      if ( outputSectorMap->find( mapkey ) == outputSectorMap->end() )
      {
        /// New entry
        std::vector< edm::Ptr< L1TkStub< T > > > tempStubVec;
        tempStubVec.clear();
        tempStubVec.push_back( tempStubPtr );
        outputSectorMap->insert( std::pair< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< L1TkStub< T > > > > ( mapkey, tempStubVec ) );
      }
      else
      {
        /// Already existing entry
        outputSectorMap->find( mapkey )->second.push_back( tempStubPtr );
      }
    }
  }

  /// Loop over the map
  /// Create Seeds and map detectors
  typename std::map< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< L1TkStub< T > > > >::const_iterator mapIter;
  for ( mapIter = stubBarrelMap.begin();
        mapIter != stubBarrelMap.end();
        ++mapIter )
  {
    /// Here we have ALL the stubs in one single Hermetic ROD (if applicable) 
    std::vector< edm::Ptr< L1TkStub< T > > > tempStubVec = mapIter->second;

    for ( unsigned int i = 0; i < tempStubVec.size(); i++ )
    {
      StackedTrackerDetId detId1( tempStubVec.at(i)->getDetId() );
      GlobalPoint pos1 = TrackingAlgorithm< T >::theStackedTracker->findAverageGlobalPosition( tempStubVec.at(i)->getClusterPtr(0).get() );

      for ( unsigned int k = i+1; k < tempStubVec.size(); k++ )
      {
        StackedTrackerDetId detId2( tempStubVec.at(k)->getDetId() );
        GlobalPoint pos2 = TrackingAlgorithm< T >::theStackedTracker->findAverageGlobalPosition( tempStubVec.at(k)->getClusterPtr(0).get() );
        /// Skip different rod pairs
        if ( detId1.iPhi() != detId2.iPhi() ) continue;
        /// Skip same layer pairs
        if ( detId1.iLayer()+1 != detId2.iLayer() ) continue;
        /// Skip off-hermetic-rod
        if ( detId1.iLayer()%2 == 0 ) continue;

///NP** NOTE This approach requires global coordinates
///Questo corrisponde a pagina 8 della nota di Anders

        /// Perform standard trigonometric operations
        double deltaPhi = pos1.phi() - pos2.phi();
        if ( fabs(deltaPhi) >= M_PI )
        {
          if ( deltaPhi>0 )
            deltaPhi = deltaPhi - 2*M_PI;
          else
            deltaPhi = 2*M_PI + deltaPhi;
        }

        double distance = sqrt( pos2.perp2() + pos1.perp2() - 2*pos2.perp()*pos1.perp()*cos(deltaPhi) );
        double rInvOver2 = sin(deltaPhi)/distance; /// Sign is maintained to keep track of the charge

///NP** NOTE questo deve essere ricalcolato
///secondo i dettami di pag 9 della nota di Anders

        /// Perform cut on Pt
        if ( fabs(rInvOver2) > mMagneticField*0.0015*0.5 ) continue;

///NP*** questo corrisponde alla prima parte di pagina 10 della nota di Anders

        /// Calculate projected vertex
        /// NOTE: cotTheta0 = Pz/Pt
        double rhoPsi1 = asin( pos1.perp()*rInvOver2 )/rInvOver2;
        double rhoPsi2 = asin( pos2.perp()*rInvOver2 )/rInvOver2;
        double cotTheta0 = ( pos1.z() - pos2.z() ) / ( rhoPsi1 - rhoPsi2 );
        double z0 = pos2.z() - rhoPsi2 * cotTheta0;

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
                                             roughPt*cotTheta0 ) );
        tempTrack.setVertex( GlobalPoint( 0, 0, z0 ) );
        tempTrack.setSector( mapIter->first.first );
        tempTrack.setWedge( mapIter->first.second );
        output.push_back( tempTrack );
      }
    } /// End of double loop over pairs of stubs

  } /// End of loop over map elements
}

/// Match a Stub to a Seed/Track
template< typename T >
void TrackingAlgorithm_exactLongBarrel< T >::AttachStubToSeed( L1TkTrack< T > &seed, edm::Ptr< L1TkStub< T > > &candidate ) const
{
  /// Get the track Stubs
  std::vector< edm::Ptr< L1TkStub< T > > > theStubs = seed.getStubPtrs();

  /// Compare SuperLayers
  unsigned int seedSuperLayer = (unsigned int)(( StackedTrackerDetId( theStubs.at(0)->getDetId() ).iLayer() + 1 )/2 );
  unsigned int targetSuperLayer = (unsigned int)(( StackedTrackerDetId( candidate->getDetId() ).iLayer() + 1 )/2 );

  if ( seedSuperLayer == targetSuperLayer )
    return;

  /// Skip if the seed and the stub are in the same
  /// SuperLayer in case of SL 3-4-5
  if ( seedSuperLayer > 2 && targetSuperLayer > 2 )
    return;

  unsigned int seedSL = ( seedSuperLayer > 2 ) ? 3 : seedSuperLayer;
  unsigned int targSL = ( targetSuperLayer > 2 ) ? 3 : targetSuperLayer;

  /// Check that the candidate is NOT the one under examination
  for ( unsigned int i = 0; i < theStubs.size(); i++ )
  {
    if ( theStubs.at(i) == candidate )
      return;
  }

  /// Get the track momentum and propagate it
  GlobalVector curMomentum = seed.getMomentum();
  GlobalPoint curVertex = seed.getVertex();
  double curRInv = seed.getRInv();

  /// Get the candidate Stub position
  GlobalPoint candPos = TrackingAlgorithm< T >::theStackedTracker->findAverageGlobalPosition( candidate->getClusterPtr(0).get() );

  /// Propagate
  double propPsi = asin( candPos.perp() * 0.5 * curRInv );
  double propRhoPsi = 2 * propPsi / curRInv;
  double propPhi = curMomentum.phi() - propPsi;
  double propZ = curVertex.z() + propRhoPsi * tan( M_PI_2 - curMomentum.theta() );

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
  double deltaZ = fabs( propZ - candPos.z() );

  /// First get the vector corresponding to the seed SL
  /// Then get in this vector the entry corresponding to the targer SL
  if ( deltaRPhi < (tableRPhi.at(seedSL)).at(targSL) && deltaZ < (tableZ.at(seedSL)).at(targSL) )
//  if ( deltaRPhi < 4 && deltaZ < 8 )
  {
    seed.addStubPtr( candidate );
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
class ES_TrackingAlgorithm_exactLongBarrel : public edm::ESProducer
{
  private:
    /// Data members
    boost::shared_ptr< TrackingAlgorithm< T > > _theAlgo;

    /// Number of Sectors
    unsigned int  mSectors;
    unsigned int  mWedges;

    /// projection windows
    std::vector< std::vector< double > > setRhoPhiWin;
    std::vector< std::vector< double > > setZWin;

  public:
    /// Constructor
    ES_TrackingAlgorithm_exactLongBarrel( const edm::ParameterSet & p )
      : mSectors( p.getParameter< int >("NumSectors") ), mWedges( p.getParameter< int >("NumWedges") )
    {
      std::vector< edm::ParameterSet > vPSet = p.getParameter< std::vector< edm::ParameterSet > >("ProjectionWindows");
      std::vector< edm::ParameterSet >::const_iterator iPSet;
      for ( iPSet = vPSet.begin(); iPSet != vPSet.end(); iPSet++ )
      {
        setRhoPhiWin.push_back( iPSet->getParameter< std::vector< double > >("RhoPhiWin") );
        setZWin.push_back( iPSet->getParameter< std::vector< double > >("ZWin") );
      }

      setWhatProduced( this );
    }

    /// Destructor
    virtual ~ES_TrackingAlgorithm_exactLongBarrel() {}

    /// Implement the producer
    boost::shared_ptr< TrackingAlgorithm< T > > produce( const TrackingAlgorithmRecord & record )
    {
      /// Get magnetic field
      edm::ESHandle< MagneticField > magnet;
      record.getRecord< IdealMagneticFieldRecord >().get(magnet);
      double mMagneticFieldStrength = magnet->inTesla(GlobalPoint(0,0,0)).z();
      double mMagneticFieldRounded = (floor(mMagneticFieldStrength*10.0 + 0.5))/10.0;

      edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
      record.getRecord< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );

      TrackingAlgorithm< T >* TrackingAlgo =
        new TrackingAlgorithm_exactLongBarrel< T >( &(*StackedTrackerGeomHandle),
                                                    mMagneticFieldRounded, mSectors, mWedges,
                                                    setRhoPhiWin, setZWin );

      _theAlgo = boost::shared_ptr< TrackingAlgorithm< T > >( TrackingAlgo );
      return _theAlgo;
    }

};

#endif

