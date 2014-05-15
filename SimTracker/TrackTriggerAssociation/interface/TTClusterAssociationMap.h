/*! \class   TTClusterAssociationMap
 *  \brief   Class to store the MC truth of L1 Track Trigger clusters
 *  \details After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 19
 *
 */

#ifndef L1_TRACK_TRIGGER_CLUSTER_ASSOCIATION_FORMAT_H
#define L1_TRACK_TRIGGER_CLUSTER_ASSOCIATION_FORMAT_H

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h" /// NOTE: this is needed even if it seems not
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

template< typename T >
class TTClusterAssociationMap
{
  public:
    /// Constructors
    TTClusterAssociationMap();

    /// Destructor
    ~TTClusterAssociationMap();

    /// Data members:   getABC( ... )
    /// Helper methods: findABC( ... )

    /// Maps
    std::map< edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > >, std::vector< edm::Ptr< TrackingParticle > > > getTTClusterToTrackingParticlesMap() const
      { return clusterToTrackingParticleVectorMap; }
    std::map< edm::Ptr< TrackingParticle >, std::vector< edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > > > > getTrackingParticleToTTClustersMap() const
      { return trackingParticleToClusterVectorMap; }

    void setTTClusterToTrackingParticlesMap( std::map< edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > >, std::vector< edm::Ptr< TrackingParticle > > > aMap )
      { clusterToTrackingParticleVectorMap = aMap; }
    void setTrackingParticleToTTClustersMap( std::map< edm::Ptr< TrackingParticle >, std::vector< edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > > > > aMap )
      { trackingParticleToClusterVectorMap = aMap; }

    /// Operations
    std::vector< edm::Ptr< TrackingParticle > >                                       findTrackingParticlePtrs( edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > > aCluster ) const;
    edm::Ptr< TrackingParticle >                                                      findTrackingParticlePtr( edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > > aCluster ) const;
    std::vector< edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > > > findTTClusterRefs( edm::Ptr< TrackingParticle > aTrackingParticle ) const;

    /// MC Truth methods
    bool isGenuine( edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > > aCluster ) const;
    bool isCombinatoric( edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > > aCluster ) const;
    bool isUnknown( edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > > aCluster ) const;

  private:
    /// Data members
    std::map< edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > >, std::vector< edm::Ptr< TrackingParticle > > > clusterToTrackingParticleVectorMap;
    std::map< edm::Ptr< TrackingParticle >, std::vector< edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > > > > trackingParticleToClusterVectorMap;

}; /// Close class

/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

/// Default Constructor
/// NOTE: to be used with setSomething(...) methods
template< typename T >
TTClusterAssociationMap< T >::TTClusterAssociationMap()
{
  /// Set default data members
  clusterToTrackingParticleVectorMap.clear();
  trackingParticleToClusterVectorMap.clear();
}

/// Destructor
template< typename T >
TTClusterAssociationMap< T >::~TTClusterAssociationMap(){}

/// Operations
template< typename T >
std::vector< edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > > > TTClusterAssociationMap< T >::findTTClusterRefs( edm::Ptr< TrackingParticle > aTrackingParticle ) const
{
  if ( trackingParticleToClusterVectorMap.find( aTrackingParticle ) != trackingParticleToClusterVectorMap.end() )
  {
    return trackingParticleToClusterVectorMap.find( aTrackingParticle )->second;
  }

  std::vector< edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > > > tempVector;
  tempVector.clear();
  return tempVector;
}

template< typename T >
std::vector< edm::Ptr< TrackingParticle > > TTClusterAssociationMap< T >::findTrackingParticlePtrs( edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > > aCluster ) const
{
  if ( clusterToTrackingParticleVectorMap.find( aCluster ) != clusterToTrackingParticleVectorMap.end() )
  {
    return clusterToTrackingParticleVectorMap.find( aCluster )->second;
  }

  std::vector< edm::Ptr< TrackingParticle > > tempVector;
  tempVector.clear();
  return tempVector;
}

/// MC truth
/// Table to define Genuine, Combinatoric and Unknown
///
/// N = number of NULL TP pointers
/// D = number of GOOD TP pointers different from each other
///
/// N / D--> | 0 | 1 | >1
/// ----------------------
/// 0        | U | G | C
/// ----------------------
/// >0       | U | C | C
///
template< typename T >
bool TTClusterAssociationMap< T >::isGenuine( edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > > aCluster ) const
{
  /// Get the TrackingParticles
  std::vector< edm::Ptr< TrackingParticle > > theseTrackingParticles = this->findTrackingParticlePtrs( aCluster );

  /// If the vector is empty, then the cluster is UNKNOWN
  if ( theseTrackingParticles.size() == 0 )
    return false;

  /// If we are here, it means there are some TrackingParticles
  unsigned int nullTPs = 0;
  unsigned int goodDifferentTPs = 0;
  std::vector< const TrackingParticle* > tpAddressVector;

  /// Loop over the TrackingParticles
  for ( unsigned int itp = 0; itp < theseTrackingParticles.size(); itp++ )
  {
    /// Get the TrackingParticle
    edm::Ptr< TrackingParticle > curTP = theseTrackingParticles.at(itp);

    /// Count the NULL TrackingParticles
    if ( curTP.isNull() )
    {
      nullTPs++;
    }
    else
    {
      /// Store the pointers (addresses) of the TrackingParticle
      /// to be able to count how many different there are
      tpAddressVector.push_back( curTP.get() );
    }
  }

  /// Count how many different TrackingParticle there are
  std::sort( tpAddressVector.begin(), tpAddressVector.end() );
  tpAddressVector.erase( std::unique( tpAddressVector.begin(), tpAddressVector.end() ), tpAddressVector.end() );
  goodDifferentTPs = tpAddressVector.size();

  /// GENUINE means no NULLs and only one good TP
  return ( nullTPs == 0 && goodDifferentTPs == 1 );
}

template< typename T >
bool TTClusterAssociationMap< T >::isUnknown( edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > > aCluster ) const
{
  /// Get the TrackingParticles
  std::vector< edm::Ptr< TrackingParticle > > theseTrackingParticles = this->findTrackingParticlePtrs( aCluster );

  /// If the vector is empty, then the cluster is UNKNOWN
  if ( theseTrackingParticles.size() == 0 )
    return true;

  /// If we are here, it means there are some TrackingParticles
  unsigned int goodDifferentTPs = 0;
  std::vector< const TrackingParticle* > tpAddressVector;

  /// Loop over the TrackingParticles
  for ( unsigned int itp = 0; itp < theseTrackingParticles.size(); itp++ )
  {
    /// Get the TrackingParticle
    edm::Ptr< TrackingParticle > curTP = theseTrackingParticles.at(itp);

    /// Count the non-NULL TrackingParticles
    if ( !curTP.isNull() )
    {
      /// Store the pointers (addresses) of the TrackingParticle
      /// to be able to count how many different there are
      tpAddressVector.push_back( curTP.get() );
    }
  }

  /// Count how many different TrackingParticle there are
  std::sort( tpAddressVector.begin(), tpAddressVector.end() );
  tpAddressVector.erase( std::unique( tpAddressVector.begin(), tpAddressVector.end() ), tpAddressVector.end() );
  goodDifferentTPs = tpAddressVector.size();

  /// UNKNOWN means no good TP is found
  return ( goodDifferentTPs == 0 );
}

template< typename T >
bool TTClusterAssociationMap< T >::isCombinatoric( edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > > aCluster ) const
{
  /// Get the TrackingParticles
  std::vector< edm::Ptr< TrackingParticle > > theseTrackingParticles = this->findTrackingParticlePtrs( aCluster );

  /// If the vector is empty, then the cluster is UNKNOWN
  if ( theseTrackingParticles.size() == 0 )
    return false;

  /// If we are here, it means there are some TrackingParticles
  unsigned int nullTPs = 0;
  unsigned int goodDifferentTPs = 0;
  std::vector< const TrackingParticle* > tpAddressVector;
  
  /// Loop over the TrackingParticles 
  for ( unsigned int itp = 0; itp < theseTrackingParticles.size(); itp++ )
  {
    /// Get the TrackingParticle 
    edm::Ptr< TrackingParticle > curTP = theseTrackingParticles.at(itp);

    /// Count the NULL TrackingParticles
    if ( curTP.isNull() )
    {
      nullTPs++;
    }
    else
    {
      /// Store the pointers (addresses) of the TrackingParticle
      /// to be able to count how many different there are
      tpAddressVector.push_back( curTP.get() );
    }
  }

  /// Count how many different TrackingParticle there are
  std::sort( tpAddressVector.begin(), tpAddressVector.end() );
  tpAddressVector.erase( std::unique( tpAddressVector.begin(), tpAddressVector.end() ), tpAddressVector.end() );
  goodDifferentTPs = tpAddressVector.size();

  /// COMBINATORIC means no NULLs and more than one good TP
  /// OR, in alternative, only one good TP but non-zero NULLS
  return ( ( nullTPs == 0 && goodDifferentTPs > 1 ) || ( nullTPs > 0 && goodDifferentTPs > 0 ) );
}

template< typename T >
edm::Ptr< TrackingParticle > TTClusterAssociationMap< T >::findTrackingParticlePtr( edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > > aCluster ) const
{
  if ( this->isGenuine( aCluster ) )
  {
    return this->findTrackingParticlePtrs( aCluster ).at(0);
  }

  edm::Ptr< TrackingParticle >* temp = new edm::Ptr< TrackingParticle >();
  return *temp;
}

#endif

