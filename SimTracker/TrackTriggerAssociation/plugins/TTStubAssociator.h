/*! \class   TTStubAssociator
 *  \brief   Plugin to create the MC truth for TTStubs.
 *  \details After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 19
 *
 */

#ifndef L1_TRACK_TRIGGER_STUB_ASSOCIATOR_H
#define L1_TRACK_TRIGGER_STUB_ASSOCIATOR_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TrackTrigger/interface/TTStub.h"

#include "L1Trigger/TrackTrigger/interface/classNameFinder.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerDetUnit.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <memory>
#include <map>
#include <vector>

template< typename T >
class TTStubAssociator : public edm::EDProducer
{
  /// NOTE since pattern hit correlation must be performed within a stacked module, one must store
  /// Clusters in a proper way, providing easy access to them in a detector/member-wise way
  public:
    /// Constructors
    explicit TTStubAssociator( const edm::ParameterSet& iConfig );

    /// Destructor
    ~TTStubAssociator();

  private:
    /// Data members
    edm::InputTag TTStubsInputTag;
    edm::InputTag TTClusterTruthInputTag;

    /// Mandatory methods
    virtual void beginRun( const edm::Run& run, const edm::EventSetup& iSetup );
    virtual void endRun( const edm::Run& run, const edm::EventSetup& iSetup );
    virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup );

}; /// Close class

/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

/// Constructors
template< typename T >
TTStubAssociator< T >::TTStubAssociator( const edm::ParameterSet& iConfig )
{
  TTStubsInputTag = iConfig.getParameter< edm::InputTag >( "TTStubs" );
  TTClusterTruthInputTag = iConfig.getParameter< edm::InputTag >( "TTClusterTruth" );

  produces< TTStubAssociationMap< T > >();
}

/// Destructor
template< typename T >
TTStubAssociator< T >::~TTStubAssociator(){}

/// Begin run
template< typename T >
void TTStubAssociator< T >::beginRun( const edm::Run& run, const edm::EventSetup& iSetup )
{
  /// Print some information when loaded
  std::cout << std::endl;
  std::cout << "TTStubAssociator< " << templateNameFinder< T >() << " > loaded."
            << std::endl;
  std::cout << std::endl;
}

/// End run
template< typename T >
void TTStubAssociator< T >::endRun( const edm::Run& run, const edm::EventSetup& iSetup ){}

/// Implement the producer
template< typename T >
void TTStubAssociator< T >::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  /// Exit if real data
  if ( iEvent.isRealData() )
    return;

  /// Prepare output
  std::auto_ptr< TTStubAssociationMap< T > > AssociationMapForOutput( new TTStubAssociationMap< T > );

  /// Get the Stubs already stored away
  edm::Handle< std::vector< TTStub< T > > > TTStubHandle;
  iEvent.getByLabel( TTStubsInputTag, TTStubHandle );

  /// Get the Cluster MC truth
  edm::Handle< TTClusterAssociationMap< T > > TTClusterAssociationMapHandle;
  iEvent.getByLabel( TTClusterTruthInputTag, TTClusterAssociationMapHandle );

  /// Prepare the necessary maps
  std::map< edm::Ptr< TTStub< T > >, edm::Ptr< TrackingParticle > >                stubToTrackingParticleMap;
  std::map< edm::Ptr< TrackingParticle >, std::vector< edm::Ptr< TTStub< T > > > > trackingParticleToStubVectorMap;
  std::map< edm::Ptr< TrackingParticle >, std::vector< unsigned int > >            trackingParticleToStubIndexVectorMap;

  unsigned int j = 0; /// Counter needed to build the edm::Ptr to the TTStub
  typename std::vector< TTStub< T > >::const_iterator inputIter;
  for ( inputIter = TTStubHandle->begin();
        inputIter != TTStubHandle->end();
        ++inputIter )
  {
    /// Make the pointer to be put in the map
    edm::Ptr< TTStub< T > > tempStubPtr( TTStubHandle, j++ );

    /// Get the two clusters
    std::vector< edm::Ptr< TTCluster< T > > > theseClusters = tempStubPtr->getClusterPtrs();

    /// Fill the inclusive map which is careless of the stub classification
    for ( unsigned int ic = 0; ic < 2; ic++ )
    {
      std::vector< edm::Ptr< TrackingParticle > > tempTPs = TTClusterAssociationMapHandle->findTrackingParticlePtrs( theseClusters.at(ic) );

      for ( unsigned int itp = 0; itp < tempTPs.size(); itp++ )
      {
        edm::Ptr< TrackingParticle > testTP = tempTPs.at(itp);

        if ( testTP.isNull() )
          continue;

        /// Prepare the maps wrt TrackingParticle
        if ( trackingParticleToStubIndexVectorMap.find( testTP ) == trackingParticleToStubIndexVectorMap.end() )
        {
          std::vector< unsigned int > stubVector;
          stubVector.clear();
          trackingParticleToStubIndexVectorMap.insert( std::make_pair( testTP, stubVector ) );
        }
        trackingParticleToStubIndexVectorMap.find( testTP )->second.push_back( j-1 ); /// Fill the auxiliary map
      }
    }

    /// GENUINE for clusters means not combinatoric and
    /// not unknown: same MC truth content MUST be found
    /// in both clusters composing the stub
    if ( TTClusterAssociationMapHandle->isUnknown( theseClusters.at(0) ) ||
         TTClusterAssociationMapHandle->isUnknown( theseClusters.at(1) ) )
    {
      /*
      if ( TTClusterAssociationMapHandle->isUnknown( theseClusters.at(0) ) &&
           TTClusterAssociationMapHandle->isUnknown( theseClusters.at(1) ) )
      {
        /// If both are unknown, it means that no TP's were found at all!
        /// hence the stub is UNKNOWN
      }
      */

      /// If at least one cluster is unknown, it means
      /// either unknown, either combinatoric
      /// Do nothing, and go to the next Stub
      continue;
    }
    else
    {
      /// Here both are clusters are genuine/combinatoric
      /// If both clusters have some known SimTrack content
      /// they must be compared to each other
      if ( TTClusterAssociationMapHandle->isGenuine( theseClusters.at(0) ) &&
           TTClusterAssociationMapHandle->isGenuine( theseClusters.at(1) ) )
      {
        /// If both clusters are genuine, they must be associated to the same TrackingParticle
        /// in order to return a genuine stub. Period. Note we can perform safely
        /// this comparison because, if both clusters are genuine, their TrackingParticle shall NEVER be NULL
        if ( TTClusterAssociationMapHandle->findTrackingParticlePtr( theseClusters.at(0) ).get() ==
             TTClusterAssociationMapHandle->findTrackingParticlePtr( theseClusters.at(1) ).get() )
        {
          /// Two genuine clusters with same SimTrack content mean genuine
          edm::Ptr< TrackingParticle > testTP = TTClusterAssociationMapHandle->findTrackingParticlePtr( theseClusters.at(0) );

          /// Fill the map: by construction, this will always be the first time the
          /// stub is inserted into the map: no need for "find"
          stubToTrackingParticleMap.insert( std::make_pair( tempStubPtr, testTP ) );

          /// At this point, go to the next Stub
          continue;
        }
        else
        {
          /// It means combinatoric
          continue;
        }
      } /// End of two genuine clusters
      else
      {
        /// Here, at least one cluster is combinatoric
        TrackingParticle* prevTPAddress = NULL;
        unsigned int whichTP = 0;

        std::vector< edm::Ptr< TrackingParticle > > trackingParticles0 = TTClusterAssociationMapHandle->findTrackingParticlePtrs( theseClusters.at(0) );
        std::vector< edm::Ptr< TrackingParticle > > trackingParticles1 = TTClusterAssociationMapHandle->findTrackingParticlePtrs( theseClusters.at(1) );

        bool escape = false;

        for ( unsigned int i = 0; i < trackingParticles0.size() && !escape; i++ )
        {
          /// Skip NULL pointers
          if ( trackingParticles0.at(i).isNull() )
            continue;

          for ( unsigned int k = 0; k < trackingParticles1.size() && !escape; k++ )
          {
            /// Skip NULL pointers
            if ( trackingParticles1.at(k).isNull() )
              continue;

            if ( trackingParticles0.at(i).get() == trackingParticles1.at(k).get() )
            {
              /// Same SimTrack is present in both clusters
              if ( prevTPAddress == NULL )
              {
                prevTPAddress = const_cast< TrackingParticle* >(trackingParticles1.at(k).get());
                whichTP = k;
              }

              /// If two different SimTracks are found in both clusters,
              /// then the stub is for sure combinatoric
              if ( prevTPAddress != const_cast< TrackingParticle* >(trackingParticles1.at(k).get()) )
              {
                escape = true;
                continue;
              }
            }
          }
        } /// End of double loop over SimTracks of both clusters

        /// If two different SimTracks are found in both clusters,
        /// go to the next stub
        if ( escape )
          continue;

        if ( prevTPAddress == NULL )
        {
          /// No SimTracks were found to be in both clusters
          continue;
        }
        else
        {
          /// Only one SimTrack was found to be present in both clusters
          /// even if one of the clusters (or both) are combinatoric:
          /// this means there is only one track that participates in
          /// both clusters, hence the stub is genuine
          edm::Ptr< TrackingParticle > testTP = trackingParticles1.at(whichTP);

          /// Fill the map: by construction, this will always be the first time the
          /// stub is inserted into the map: no need for "find"
          stubToTrackingParticleMap.insert( std::make_pair( tempStubPtr, testTP ) );

          /// At this point, go to the next Stub
          continue;
        } /// End of one single SimTrack in both clusters
      } /// End of "at least one cluster is combinatoric"
    } /// End of "both clusters are known, somehow..."
  } /// End of loop over Stubs

  /// Clean the only map that needs cleaning
  /// Prepare the output map wrt TrackingParticle
  std::map< edm::Ptr< TrackingParticle >, std::vector< unsigned int > >::iterator iterMapToClean;
  for ( iterMapToClean = trackingParticleToStubIndexVectorMap.begin();
        iterMapToClean != trackingParticleToStubIndexVectorMap.end();
        ++iterMapToClean )
  {
    /// Get the vector of edm::Ptr< TTStub >
    std::vector< unsigned int > tempVector = iterMapToClean->second;

    /// Sort and remove duplicates
    std::sort( tempVector.begin(), tempVector.end() );
    tempVector.erase( std::unique( tempVector.begin(), tempVector.end() ), tempVector.end() );

    /// Create the vector for the output map
    std::vector< edm::Ptr< TTStub< T > > > outputVector;
    outputVector.clear();

    for ( unsigned int k = 0; k < tempVector.size(); k++ )
    {
      edm::Ptr< TTStub< T > > tempStubPtr( TTStubHandle, tempVector.at(k) );
      outputVector.push_back( tempStubPtr );
    }

    /// Put the vector in the output map
    trackingParticleToStubVectorMap.insert( std::make_pair( iterMapToClean->first, outputVector ) );
  }

  /// Also, create the pointer to the TTClusterAssociationMap
  edm::RefProd< TTClusterAssociationMap< T > > theCluAssoMap( TTClusterAssociationMapHandle ); 

  /// Put the maps in the association object
  AssociationMapForOutput->setTTStubToTrackingParticleMap( stubToTrackingParticleMap );
  AssociationMapForOutput->setTrackingParticleToTTStubsMap( trackingParticleToStubVectorMap );
  AssociationMapForOutput->setTTClusterAssociationMap( theCluAssoMap );

  /// Put output in the event
  iEvent.put( AssociationMapForOutput );
}

#endif

