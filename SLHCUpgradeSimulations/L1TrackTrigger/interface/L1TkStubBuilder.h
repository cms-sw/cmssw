/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
///                                      ///
/// Andrew W. Rose, IC                   ///
/// Nicola Pozzobon,  UNIPD              ///
///                                      ///
/// 2008                                 ///
/// 2010, June                           ///
/// 2011, July                           ///
/// ////////////////////////////////////////

#ifndef L1TK_STUB_BUILDER_H
#define L1TK_STUB_BUILDER_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithm.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithmRecord.h"
#include "classNameFinder.h"

#include <memory>
#include <map>
#include <vector>

/** ************************ **/
/**                          **/
/**   DECLARATION OF CLASS   **/
/**                          **/
/** ************************ **/

template<  typename T  >
class L1TkStubBuilder : public edm::EDProducer
{
  public:
    typedef std::pair< StackedTrackerDetId, unsigned int >                      ClusterKey;     /// This is the key
    typedef std::map< ClusterKey, std::vector< edm::Ptr< L1TkCluster< T > > > > L1TkClusterMap; /// This is the map

    /// Constructor
    explicit L1TkStubBuilder( const edm::ParameterSet& iConfig );

    /// Destructor;
    ~L1TkStubBuilder();

  private:
    /// Data members
    const StackedTrackerGeometry               *theStackedTracker;
    edm::ESHandle< HitMatchingAlgorithm< T > > MatchingAlgoHandle;
    edm::InputTag                                           L1TkClustersInputTag;

    /// Mandatory methods
    virtual void beginRun( const edm::Run& run, const edm::EventSetup& iSetup );
    virtual void endRun( const edm::Run& run, const edm::EventSetup& iSetup );
    virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup );

}; /// Close class

/** ***************************** **/
/**                               **/
/**   IMPLEMENTATION OF METHODS   **/
/**                               **/
/** ***************************** **/

/// Constructors
template< typename T >
L1TkStubBuilder< T >::L1TkStubBuilder( const edm::ParameterSet& iConfig )
{
  produces< std::vector< L1TkStub< T > > >( "StubsPass" );
  produces< std::vector< L1TkStub< T > > >( "StubsFail" );
  L1TkClustersInputTag = iConfig.getParameter< edm::InputTag >("L1TkClusters");
}

/// Destructor
template< typename T >
L1TkStubBuilder< T >::~L1TkStubBuilder(){}

/// Begin run
template< typename T >
void L1TkStubBuilder< T >::beginRun( const edm::Run& run, const edm::EventSetup& iSetup )
{
  /// Get the geometry references
  edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
  iSetup.get< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );
  theStackedTracker = StackedTrackerGeomHandle.product();

  /// Get the clustering algorithm 
  iSetup.get< HitMatchingAlgorithmRecord >().get( MatchingAlgoHandle );

  /// Print some information when loaded
  std::cout << std::endl;
  std::cout << "L1TkStubBuilder<" << templateNameFinder<T>() << "> loaded modules:"
            << "\n\tHitMatchingAlgorithm:\t" << MatchingAlgoHandle->AlgorithmName()
            << std::endl;
  std::cout << std::endl;
  std::cout << "W A R N I N G! this producer produces two categories of stubs: accepted and rejected!" << std::endl;
  std::cout << "               ONLY the accepted ones are correctly filled with MC truth etc" << std::endl;
  std::cout << "               the rejected ones have only the clusters correctly filled in data members!" << std::endl;
  std::cout << std::endl;

}

/// End run
template< typename T >
void L1TkStubBuilder< T >::endRun( const edm::Run& run, const edm::EventSetup& iSetup ){}

/// Implement the producer
template< typename T >
void L1TkStubBuilder< T >::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{  
  /// Prepare output
  std::auto_ptr< std::vector< L1TkStub< T > > > L1TkStubsForOutputAccepted( new std::vector< L1TkStub< T > > );
  std::auto_ptr< std::vector< L1TkStub< T > > > L1TkStubsForOutputRejected( new std::vector< L1TkStub< T > > );

  /// Get the Clusters already stored away
  edm::Handle< std::vector< L1TkCluster< T > > > L1TkClusterHandle;
  iEvent.getByLabel( L1TkClustersInputTag, L1TkClusterHandle);   

  /// Map the Clusters according to detector elements
  L1TkClusterMap clusterMap;
  clusterMap.clear();
  typename std::vector< L1TkCluster< T > >::const_iterator inputIter;
  unsigned int j = 0; /// Counter needed to build the edm::Ptr to the L1TkCluster

  for ( inputIter = L1TkClusterHandle->begin();
        inputIter != L1TkClusterHandle->end();
        ++inputIter )
  {
    /// Make the pointer to be put in the map and, later on, in the Stub
    /// as reference to lower-class bricks composing the Stub itself
    edm::Ptr< L1TkCluster< T > > tempCluPtr( L1TkClusterHandle, j++ );
    /// Build the key to the map
    ClusterKey mapkey = std::make_pair( StackedTrackerDetId( inputIter->getDetId() ), inputIter->getStackMember() );

    /// If an entry already exists for this key, just add the cluster
    /// to the vector, otherwise create the entry
    if ( clusterMap.find( mapkey ) == clusterMap.end() )
    {
      /// New entry
      std::vector< edm::Ptr< L1TkCluster< T > > > tempCluVec;
      tempCluVec.clear();
      tempCluVec.push_back( tempCluPtr );
      clusterMap.insert( std::pair< ClusterKey, std::vector< edm::Ptr< L1TkCluster< T > > > > ( mapkey, tempCluVec ) );
    }
    else
    {
      /// Already existing entry
      clusterMap[mapkey].push_back( tempCluPtr );   
    }
  }

  /// Loop over the detector elements
  StackedTrackerGeometry::StackContainerIterator StackedTrackerIterator;
  for ( StackedTrackerIterator = theStackedTracker->stacks().begin();
        StackedTrackerIterator != theStackedTracker->stacks().end();
        ++StackedTrackerIterator )
  {
    StackedTrackerDetUnit* Unit = *StackedTrackerIterator;
    StackedTrackerDetId Id = Unit->Id();
    assert(Unit == theStackedTracker->idToStack(Id));
    
    /// Build the keys to get the Clusters
    ClusterKey inmapkey  = std::make_pair(Id, 0);
    ClusterKey outmapkey = std::make_pair(Id, 1);

    /// Get the vectors of Clusters for the current Pt module
    /// Go on only if the entry in the map is found
    typename L1TkClusterMap::const_iterator innerIter = clusterMap.find(inmapkey);
    typename L1TkClusterMap::const_iterator outerIter = clusterMap.find(outmapkey);

    if ( innerIter == clusterMap.end() || outerIter == clusterMap.end() ) continue;

    std::vector< edm::Ptr< L1TkCluster< T > > > innerClusters = innerIter->second;
    std::vector< edm::Ptr< L1TkCluster< T > > > outerClusters = outerIter->second;

    typename std::vector< edm::Ptr< L1TkCluster< T > > >::iterator innerClusterIter, outerClusterIter;

    /// If there are Clusters in both sensors
    /// you can try and make a Stub
    if ( innerClusters.size() && outerClusters.size() )
    {
      /// Loop over pairs of Clusters
      for ( innerClusterIter = innerClusters.begin();
            innerClusterIter != innerClusters.end();
            ++innerClusterIter )
      {
        for ( outerClusterIter = outerClusters.begin();
              outerClusterIter != outerClusters.end();
              ++outerClusterIter )
        {
          /// Build a temporary Stub
          L1TkStub< T > tempL1TkStub( Id );
          tempL1TkStub.addClusterPtr( *innerClusterIter ); /// innerClusterIter is an iterator pointing to the edm::Ptr
          tempL1TkStub.addClusterPtr( *outerClusterIter );

          /// Check for compatibility
          bool thisConfirmation = false;
          int thisDisplacement = 999999;
          int thisOffset = 0; 

          MatchingAlgoHandle->CheckTwoMemberHitsForCompatibility( thisConfirmation, thisDisplacement, thisOffset, tempL1TkStub );

          /// If the Stub is above threshold
          if ( thisConfirmation )
          {
            if ( iEvent.isRealData() == false )
              tempL1TkStub.checkSimTrack();

            tempL1TkStub.setTriggerDisplacement( thisDisplacement );
            tempL1TkStub.setTriggerOffset( thisOffset );

            /// Put in the output
            L1TkStubsForOutputAccepted->push_back( tempL1TkStub );

          } /// Stub accepted
          else
            L1TkStubsForOutputRejected->push_back( tempL1TkStub );

        } /// End of nested loop
      } /// End of loop over pairs of Clusters
    } /// End of cross check there are Clusters in both sensors
  } /// End of loop over detector elements

  /// Put output in the event
  iEvent.put( L1TkStubsForOutputAccepted, "StubsPass" );
  iEvent.put( L1TkStubsForOutputRejected, "StubsFail" );
}

#endif

