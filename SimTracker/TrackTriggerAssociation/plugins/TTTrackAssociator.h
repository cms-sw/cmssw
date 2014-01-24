/*! \class   TTTrackAssociator
 *  \brief   Plugin to create the MC truth for TTTracks.
 *  \details After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 19
 *
 */

#ifndef L1_TRACK_TRIGGER_TRACK_ASSOCIATOR_H
#define L1_TRACK_TRIGGER_TRACK_ASSOCIATOR_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"

#include "L1Trigger/TrackTrigger/interface/classNameFinder.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTTrackAssociationMap.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerDetUnit.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <memory>
#include <map>
#include <vector>

template< typename T >
class TTTrackAssociator : public edm::EDProducer
{
  /// NOTE since pattern hit correlation must be performed within a stacked module, one must store
  /// Clusters in a proper way, providing easy access to them in a detector/member-wise way
  public:
    /// Constructors
    explicit TTTrackAssociator( const edm::ParameterSet& iConfig );

    /// Destructor
    ~TTTrackAssociator();

  private:
    /// Data members
    std::vector< edm::InputTag > TTTracksInputTags;
    edm::InputTag TTClusterTruthInputTag;
    edm::InputTag TTStubTruthInputTag;

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
TTTrackAssociator< T >::TTTrackAssociator( const edm::ParameterSet& iConfig )
{
  TTTracksInputTags = iConfig.getParameter< std::vector< edm::InputTag > >( "TTTracks" );
  TTClusterTruthInputTag = iConfig.getParameter< edm::InputTag >( "TTClusterTruth" );
  TTStubTruthInputTag = iConfig.getParameter< edm::InputTag >( "TTStubTruth" );

  for ( unsigned int iTag = 0; iTag < TTTracksInputTags.size(); iTag++ )
  {
    produces< TTTrackAssociationMap< T > >( TTTracksInputTags.at(iTag).instance() );
  }
}

/// Destructor
template< typename T >
TTTrackAssociator< T >::~TTTrackAssociator(){}

/// Begin run
template< typename T >
void TTTrackAssociator< T >::beginRun( const edm::Run& run, const edm::EventSetup& iSetup )
{
  /// Print some information when loaded
  std::cout << std::endl;
  std::cout << "TTTrackAssociator< " << templateNameFinder< T >() << " > loaded."
            << std::endl;
  std::cout << std::endl;
}

/// End run
template< typename T >
void TTTrackAssociator< T >::endRun( const edm::Run& run, const edm::EventSetup& iSetup ){}

/// Implement the producer
template< typename T >
void TTTrackAssociator< T >::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  /// Exit if real data
  if ( iEvent.isRealData() )
    return;

  /// Get the Stub and Cluster MC truth
  edm::Handle< TTClusterAssociationMap< T > > TTClusterAssociationMapHandle;
  iEvent.getByLabel( TTClusterTruthInputTag, TTClusterAssociationMapHandle );
  edm::Handle< TTStubAssociationMap< T > > TTStubAssociationMapHandle;
  iEvent.getByLabel( TTStubTruthInputTag, TTStubAssociationMapHandle );

  /// Loop over InputTags to handle multiple collections
  for ( unsigned int iTag = 0; iTag < TTTracksInputTags.size(); iTag++ )
  {
    /// Prepare output
    std::auto_ptr< TTTrackAssociationMap< T > > AssociationMapForOutput( new TTTrackAssociationMap< T > );

    /// Get the Tracks already stored away
    edm::Handle< std::vector< TTTrack< T > > > TTTrackHandle;
    iEvent.getByLabel( TTTracksInputTags.at(iTag), TTTrackHandle );

    /// Prepare the necessary maps
    std::map< edm::Ptr< TTTrack< T > >, edm::Ptr< TrackingParticle > >                trackToTrackingParticleMap;
    std::map< edm::Ptr< TrackingParticle >, std::vector< edm::Ptr< TTTrack< T > > > > trackingParticleToTrackVectorMap;
    trackToTrackingParticleMap.clear();
    trackingParticleToTrackVectorMap.clear();


    unsigned int j = 0; /// Counter needed to build the edm::Ptr to the TTTrack
    typename std::vector< TTTrack< T > >::const_iterator inputIter;
    for ( inputIter = TTTrackHandle->begin();
          inputIter != TTTrackHandle->end();
          ++inputIter )
    {
      /// Make the pointer to be put in the map
      edm::Ptr< TTTrack< T > > tempTrackPtr( TTTrackHandle, j++ );

      /// Get the stubs
      std::vector< edm::Ref< edmNew::DetSetVector< TTStub< T > >, TTStub< T > > > theseStubs = tempTrackPtr->getStubRefs();

      /// Auxiliary map to store TP addresses and TP edm::Ptr
      std::map< const TrackingParticle*, edm::Ptr< TrackingParticle > > auxMap;
      auxMap.clear();
      bool mayCombinUnknown = false;

      /// Fill the inclusive map which is careless of the stub classification
      for ( unsigned int is = 0; is < theseStubs.size(); is++ )
      {
        std::vector< edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > > > theseClusters = theseStubs.at(is)->getClusterRefs();
        for ( unsigned int ic = 0; ic < 2; ic++ )
        {
          std::vector< edm::Ptr< TrackingParticle > > tempTPs = TTClusterAssociationMapHandle->findTrackingParticlePtrs( theseClusters.at(ic) );
          for ( unsigned int itp = 0; itp < tempTPs.size(); itp++ )
          {
            edm::Ptr< TrackingParticle > testTP = tempTPs.at(itp);

            if ( testTP.isNull() )
              continue;

            /// Prepare the maps wrt TrackingParticle
            if ( trackingParticleToTrackVectorMap.find( testTP ) == trackingParticleToTrackVectorMap.end() )
            {
              std::vector< edm::Ptr< TTTrack< T > > > trackVector;
              trackVector.clear();
              trackingParticleToTrackVectorMap.insert( std::make_pair( testTP, trackVector ) );
            }
            trackingParticleToTrackVectorMap.find( testTP )->second.push_back( tempTrackPtr ); /// Fill the auxiliary map

            /// Fill the other auxiliary map
            if ( auxMap.find( testTP.get() ) == auxMap.end() )
            {
              auxMap.insert( std::make_pair( testTP.get(), testTP ) );
            }

          }
        } /// End of loop over the clusters

        /// Check if the stub is unknown
        if ( TTStubAssociationMapHandle->isUnknown( theseStubs.at(is) ) )
          mayCombinUnknown = true;

      } /// End of loop over the stubs

      /// If there is an unknown stub, go to the next track
      /// as this track may be COMBINATORIC or UNKNOWN
      if ( mayCombinUnknown )
        continue;

      /// If we are here, all the stubs are either combinatoric or genuine
      /// Loop over all the TrackingParticle
      std::vector< const TrackingParticle* > tpInAllStubs;
      std::map< const TrackingParticle*, edm::Ptr< TrackingParticle > >::const_iterator iterAuxMap;
      for ( iterAuxMap = auxMap.begin();
            iterAuxMap != auxMap.end();
            ++iterAuxMap )
      {
        /// Get all the stubs from this TrackingParticle
        std::vector< edm::Ref< edmNew::DetSetVector< TTStub< T > >, TTStub< T > > > tempStubs = TTStubAssociationMapHandle->findTTStubRefs( iterAuxMap->second );

        bool allFound = true;
        /// Loop over the stubs
        for ( unsigned int js = 0; js < theseStubs.size() && allFound; js++ )
        {
          /// We want that all the stubs of the track are included in the container of
          /// all the stubs produced by this particular TrackingParticle which we
          /// already know is one of the TrackingParticles that released hits
          /// in this track we are evaluating right now
          if ( std::find( tempStubs.begin(), tempStubs.end(), theseStubs.at(js) ) == tempStubs.end() )
          {
            allFound = false;
          }
        }

        /// If the TrackingParticle does not appear in all stubs
        /// then go to the next track
        if ( !allFound )
          continue;

        /// If we are here, it means that the TrackingParticle
        /// generates hits in all stubs of the current track
        /// so put it into the vector
        tpInAllStubs.push_back( iterAuxMap->first );
      }

      /// Count how many TrackingParticles we do have
      std::sort( tpInAllStubs.begin(), tpInAllStubs.end() );
      tpInAllStubs.erase( std::unique( tpInAllStubs.begin(), tpInAllStubs.end() ), tpInAllStubs.end() );
      unsigned int nTPs = tpInAllStubs.size();

      /// If only one TrackingParticle, GENUINE
      /// if more than one, COMBINATORIC
      if ( nTPs != 1 )
        continue;

      /// Here, the track may only be GENUINE
      /// Fill the map
      trackToTrackingParticleMap.insert( std::make_pair( tempTrackPtr, auxMap.find( tpInAllStubs.at(0) )->second ) ); 

    } /// End of loop over Tracks

    /// Clean the only map that needs cleaning
    /// Prepare the output map wrt TrackingParticle
    typename std::map< edm::Ptr< TrackingParticle >, std::vector< edm::Ptr< TTTrack< T > > > >::iterator iterMapToClean;
    for ( iterMapToClean = trackingParticleToTrackVectorMap.begin();
          iterMapToClean != trackingParticleToTrackVectorMap.end();
          ++iterMapToClean )
    {
      /// Get the vector of edm::Ptr< TTTrack >
      std::vector< edm::Ptr< TTTrack< T > > > tempVector = iterMapToClean->second;

      /// Sort and remove duplicates
      std::sort( tempVector.begin(), tempVector.end() );
      tempVector.erase( std::unique( tempVector.begin(), tempVector.end() ), tempVector.end() );

      iterMapToClean->second = tempVector;
    }

    /// Also, create the pointer to the TTClusterAssociationMap
    edm::RefProd< TTStubAssociationMap< T > > theStubAssoMap( TTStubAssociationMapHandle );

    /// Put the maps in the association object
    AssociationMapForOutput->setTTTrackToTrackingParticleMap( trackToTrackingParticleMap ); 
    AssociationMapForOutput->setTrackingParticleToTTTracksMap( trackingParticleToTrackVectorMap );
    AssociationMapForOutput->setTTStubAssociationMap( theStubAssoMap );

    /// Put output in the event
    iEvent.put( AssociationMapForOutput, TTTracksInputTags.at(iTag).instance() );

  } /// End of loop over InputTags
}

#endif

