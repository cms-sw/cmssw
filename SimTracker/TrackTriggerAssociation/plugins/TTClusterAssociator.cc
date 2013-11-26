/*! \brief   
 *  \details Here, in the source file, the methods which do depend
 *           on the specific type <T> that can fit the template.
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 19
 *
 */

#include "SimTracker/TrackTriggerAssociation/plugins/TTClusterAssociator.h"

/// Implement the producer
template< >
void TTClusterAssociator< Ref_PixelDigi_ >::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  /// Exit if real data
  if ( iEvent.isRealData() )
    return;

  /// Get the PixelDigiSimLink
  iEvent.getByLabel( "simSiPixelDigis", thePixelDigiSimLinkHandle );

  /// Get the TrackingParticles
  edm::Handle< std::vector< TrackingParticle > > TrackingParticleHandle;
  iEvent.getByLabel( "mix", "MergedTrackTruth", TrackingParticleHandle );

  /// Loop over InputTags to handle multiple collections
  for ( unsigned int iTag = 0; iTag < TTClustersInputTags.size(); iTag++ )
  {
    /// Prepare output
    std::auto_ptr< TTClusterAssociationMap< Ref_PixelDigi_ > > AssociationMapForOutput( new TTClusterAssociationMap< Ref_PixelDigi_ > );

    /// Get the Clusters already stored away
    edm::Handle< std::vector< TTCluster< Ref_PixelDigi_ > > > TTClusterHandle;
    iEvent.getByLabel( TTClustersInputTags.at(iTag), TTClusterHandle );

    /// Preliminary task: map SimTracks by TrackingParticle
    /// Prepare the map
    std::map< std::pair< unsigned int, EncodedEventId >, edm::Ptr< TrackingParticle > > simTrackUniqueToTPMap;
    simTrackUniqueToTPMap.clear();

    if ( TrackingParticleHandle->size() != 0 )
    {
      /// Loop over TrackingParticles
      unsigned int tpCnt = 0;
      std::vector< TrackingParticle >::const_iterator iterTPart;
      for ( iterTPart = TrackingParticleHandle->begin();
            iterTPart != TrackingParticleHandle->end();
            ++iterTPart )
      {
        /// Make the pointer to the TrackingParticle
        edm::Ptr< TrackingParticle > tempTPPtr( TrackingParticleHandle, tpCnt++ );

        /// Get the EncodedEventId
        EncodedEventId eventId = EncodedEventId( tempTPPtr->eventId() );

        /// Loop over SimTracks inside TrackingParticle
        std::vector< SimTrack >::const_iterator iterSimTrack;
        for ( iterSimTrack = tempTPPtr->g4Tracks().begin();
              iterSimTrack != tempTPPtr->g4Tracks().end();
              ++iterSimTrack )
        {
          /// Build the unique SimTrack Id (which is SimTrack ID + EncodedEventId)
          std::pair< unsigned int, EncodedEventId > simTrackUniqueId( iterSimTrack->trackId(), eventId );
          simTrackUniqueToTPMap.insert( std::make_pair( simTrackUniqueId, tempTPPtr ) );
        }
      } /// End of loop over TrackingParticles
    }

    /// Prepare the necessary maps
    std::map< edm::Ptr< TTCluster< Ref_PixelDigi_ > >, std::vector< edm::Ptr< TrackingParticle > > > clusterToTrackingParticleVectorMap;
    std::map< edm::Ptr< TrackingParticle >, std::vector< edm::Ptr< TTCluster< Ref_PixelDigi_ > > > > trackingParticleToClusterVectorMap;
    std::map< edm::Ptr< TrackingParticle >, std::vector< unsigned int > >                            trackingParticleToClusterIndexVectorMap;

    unsigned int j = 0; /// Counter needed to build the edm::Ptr to the TTCluster
    typename std::vector< TTCluster< Ref_PixelDigi_ > >::const_iterator inputIter;
    for ( inputIter = TTClusterHandle->begin();
          inputIter != TTClusterHandle->end();
          ++inputIter )
    {
      /// Make the pointer to be put in the map
      edm::Ptr< TTCluster< Ref_PixelDigi_ > > tempCluPtr( TTClusterHandle, j++ );

      /// Prepare the maps wrt TTCluster
      if ( clusterToTrackingParticleVectorMap.find( tempCluPtr ) == clusterToTrackingParticleVectorMap.end() )
      {
        std::vector< edm::Ptr< TrackingParticle > > tpVector;
        tpVector.clear();
        clusterToTrackingParticleVectorMap.insert( std::make_pair( tempCluPtr, tpVector ) );
      }

      /// Get the DetId
      const DetId detId = theStackedTrackers->idToDet( tempCluPtr->getDetId(), tempCluPtr->getStackMember() )->geographicalId();

      /// Get the PixelDigiSimLink
      edm::DetSet<PixelDigiSimLink> thisDigiSimLink = (*(thePixelDigiSimLinkHandle) )[detId.rawId()];
      edm::DetSet<PixelDigiSimLink>::const_iterator iterSimLink;

      /// Get the Digis and loop over them
      std::vector< Ref_PixelDigi_ > theseHits = tempCluPtr->getHits();
      for ( unsigned int i = 0; i < theseHits.size(); i++ )
      {
        /// Loop over PixelDigiSimLink
        for ( iterSimLink = thisDigiSimLink.data.begin();
              iterSimLink != thisDigiSimLink.data.end();
              iterSimLink++ )
        {
          /// Find the link and, if there's not, skip
          if ( (int)iterSimLink->channel() != theseHits.at(i)->channel() )
            continue;

          /// Get SimTrack Id and type
          unsigned int curSimTrkId = iterSimLink->SimTrackId();
          EncodedEventId curSimEvId = iterSimLink->eventId();

          /// Prepare the SimTrack Unique ID
          std::pair< unsigned int, EncodedEventId > thisUniqueId = std::make_pair( curSimTrkId, curSimEvId );

          /// Get the corresponding TrackingParticle
          if ( simTrackUniqueToTPMap.find( thisUniqueId ) != simTrackUniqueToTPMap.end() )
          {
            edm::Ptr< TrackingParticle > thisTrackingParticle = simTrackUniqueToTPMap.find( thisUniqueId )->second;

            /// Store the TrackingParticle
            clusterToTrackingParticleVectorMap.find( tempCluPtr )->second.push_back( thisTrackingParticle );

            /// Prepare the maps wrt TrackingParticle
            if ( trackingParticleToClusterIndexVectorMap.find( thisTrackingParticle ) == trackingParticleToClusterIndexVectorMap.end() )
            {
              std::vector< unsigned int > clusterVector;
              clusterVector.clear();
              trackingParticleToClusterIndexVectorMap.insert( std::make_pair( thisTrackingParticle, clusterVector ) );
            }
            trackingParticleToClusterIndexVectorMap.find( thisTrackingParticle )->second.push_back( j-1 ); /// Fill the auxiliary map
          }
          else 
          {
            /// In case no TrackingParticle is found, store a NULL pointer
            edm::Ptr< TrackingParticle >* tempTPPtr = new edm::Ptr< TrackingParticle >();
            clusterToTrackingParticleVectorMap.find( tempCluPtr )->second.push_back( *tempTPPtr );
          }
        } /// End of loop over PixelDigiSimLink
      } /// End of loop over all the hits composing the L1TkCluster
    } /// End of loop over all the TTClusters of the event

    /// Clean the maps that need cleaning
    /// Prepare the output map wrt TrackingParticle
    std::map< edm::Ptr< TrackingParticle >, std::vector< unsigned int > >::iterator iterMapToClean;
    for ( iterMapToClean = trackingParticleToClusterIndexVectorMap.begin();
          iterMapToClean != trackingParticleToClusterIndexVectorMap.end();
          ++iterMapToClean )
    {
      /// Get the vector of edm::Ptr< TTCluster >
      std::vector< unsigned int > tempVector = iterMapToClean->second;

      /// Sort and remove duplicates
      std::sort( tempVector.begin(), tempVector.end() );
      tempVector.erase( std::unique( tempVector.begin(), tempVector.end() ), tempVector.end() );

      /// Create the vector for the output map
      std::vector< edm::Ptr< TTCluster< Ref_PixelDigi_ > > > outputVector;
      outputVector.clear();

      for ( unsigned int k = 0; k < tempVector.size(); k++ )
      {
        edm::Ptr< TTCluster< Ref_PixelDigi_ > > tempCluPtr( TTClusterHandle, tempVector.at(k) );
        outputVector.push_back( tempCluPtr );
      }

      /// Put the vector in the output map
      trackingParticleToClusterVectorMap.insert( std::make_pair( iterMapToClean->first, outputVector ) );
    }

    /// Put the maps in the association object
    AssociationMapForOutput->setTTClusterToTrackingParticlesMap( clusterToTrackingParticleVectorMap );
    AssociationMapForOutput->setTrackingParticleToTTClustersMap( trackingParticleToClusterVectorMap );

    /// Put output in the event
    iEvent.put( AssociationMapForOutput, TTClustersInputTags.at(iTag).instance() );

  } /// End of loop over input tags
}

