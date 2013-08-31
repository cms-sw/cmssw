/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
///                                      ///
/// Nicola Pozzobon,  UNIPD              ///
///                                      ///
/// 2011, June                           ///
/// ////////////////////////////////////////

#ifndef CLUSTER_BUILDER_H
#define CLUSTER_BUILDER_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/ClusteringAlgorithm.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/ClusteringAlgorithmRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerDetUnit.h"
#include <memory>
#include <map>
#include <vector>

/** ************************ **/
/**                          **/
/**   DECLARATION OF CLASS   **/
/**                          **/
/** ************************ **/

template< typename T >
class L1TkClusterBuilder : public edm::EDProducer
{
  /// NOTE since pattern hit correlation must be performed within a stacked module, one must store
  /// Clusters in a proper way, providing easy access to them in a detector/member-wise way
  public:
    /// Constructors
    explicit L1TkClusterBuilder( const edm::ParameterSet& iConfig );

    /// Destructor
    ~L1TkClusterBuilder();

  private:
    /// Data members
    const StackedTrackerGeometry              *theStackedTrackers;
    edm::ESHandle< ClusteringAlgorithm< T > > ClusteringAlgoHandle; // Handles are needed in ::produce()
    edm::Handle< edm::DetSetVector< PixelDigiSimLink > >   PixelDigiSimLinkHandle;
    edm::Handle< edm::SimTrackContainer >                  SimTrackHandle;
    std::vector< edm::InputTag >                           rawHitInputTags;
    edm::InputTag                                          simTrackInputTag;
    unsigned int                                           ADCThreshold;  

    /// Mandatory methods
    virtual void beginRun( const edm::Run& run, const edm::EventSetup& iSetup );
    virtual void endRun( const edm::Run& run, const edm::EventSetup& iSetup );
    virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup );

    /// Get hits
    void RetrieveRawHits( std::map< DetId, std::vector< T > > &mRawHits, const edm::Event& iEvent );

}; /// Close class

/** ***************************** **/
/**                               **/
/**   IMPLEMENTATION OF METHODS   **/
/**                               **/
/** ***************************** **/

/// Constructors
/// Default is for PixelDigis
template< typename T >
L1TkClusterBuilder< T >::L1TkClusterBuilder( const edm::ParameterSet& iConfig )
{
  rawHitInputTags  = iConfig.getParameter< std::vector< edm::InputTag > >("rawHits");
  simTrackInputTag = iConfig.getParameter< edm::InputTag >("simTrackHits");
  ADCThreshold     = iConfig.getParameter< unsigned int >("ADCThreshold");
  produces< std::vector< L1TkCluster< T > > >();
}

/// Constructors
/// Specialize for PSimHits
template<>
L1TkClusterBuilder< Ref_PSimHit_ >::L1TkClusterBuilder( const edm::ParameterSet& iConfig )
{
  rawHitInputTags  = iConfig.getParameter< std::vector< edm::InputTag > >("rawHits");
  simTrackInputTag = iConfig.getParameter< edm::InputTag >("simTrackHits");
  produces< std::vector< L1TkCluster< Ref_PSimHit_ > > >();
}

/// Destructor
template< typename T >
L1TkClusterBuilder< T >::~L1TkClusterBuilder(){}

/// Begin run
template< typename T >
void L1TkClusterBuilder< T >::beginRun( const edm::Run& run, const edm::EventSetup& iSetup )
{
  /// Get the geometry
  edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
  iSetup.get< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );
  theStackedTrackers = StackedTrackerGeomHandle.product();

  /// Get the clustering algorithm 
  iSetup.get< ClusteringAlgorithmRecord >().get( ClusteringAlgoHandle );

  /// Print some information when loaded
  std::cout << std::endl;
  std::cout << "L1TkClusterBuilder<" << templateNameFinder<T>() << "> loaded modules:"
            << "\n\tClusteringAlgorithm:\t" << ClusteringAlgoHandle->AlgorithmName()
            << std::endl;
  std::cout << std::endl;
}

/// End run
template< typename T >
void L1TkClusterBuilder< T >::endRun( const edm::Run& run, const edm::EventSetup& iSetup ){}

/// Implement the producer
template< typename T >
void L1TkClusterBuilder< T >::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  /// Prepare output
  std::auto_ptr< std::vector< L1TkCluster< T > > > ClustersForOutput( new std::vector< L1TkCluster< T > > );

  /// Get the PixelDigiSimLink
  iEvent.getByLabel( "simSiPixelDigis", PixelDigiSimLinkHandle );

  /// Get the SimTracks
  iEvent.getByLabel( simTrackInputTag, SimTrackHandle );

  std::map< DetId, std::vector< T > > rawHits; /// This is a map containing hits:
                                               /// a vector of type T is mapped wrt
                                               /// the DetId
  this->RetrieveRawHits( rawHits, iEvent );

  /// Loop over the detector elements
  StackedTrackerGeometry::StackContainerIterator StackedTrackerIterator;
  for ( StackedTrackerIterator = theStackedTrackers->stacks().begin();
        StackedTrackerIterator != theStackedTrackers->stacks().end();
        ++StackedTrackerIterator )
  {
    StackedTrackerDetUnit* Unit = *StackedTrackerIterator;
    StackedTrackerDetId Id = Unit->Id();
    assert(Unit == theStackedTrackers->idToStack(Id));

    /// Temp vectors containing the vectors of the
    /// hits used to build each cluster
    std::vector< std::vector< T > > innerHits, outerHits;

    /// Find the hits in each stack member
    typename std::map< DetId, std::vector< T > >::const_iterator innerHitFind = rawHits.find( Unit->stackMember(0) );
    typename std::map< DetId, std::vector< T > >::const_iterator outerHitFind = rawHits.find( Unit->stackMember(1) );

    /// If there are hits, cluster them
    /// It is the ClusteringAlgorithm::Cluster method which
    /// calls the constructor to the Cluster class!
    if ( innerHitFind != rawHits.end() ) ClusteringAlgoHandle->Cluster( innerHits, innerHitFind->second );
    if ( outerHitFind != rawHits.end() ) ClusteringAlgoHandle->Cluster( outerHits, outerHitFind->second );

    /// Create L1TkCluster objects and store them
    for ( unsigned int i = 0; i < innerHits.size(); i++ )
    {
      L1TkCluster< T > temp( innerHits.at(i), Id, 0 );
      /// If MC, check also fake or non/fake
      if ( iEvent.isRealData() == false )
        theStackedTrackers->checkSimTrack(&temp, PixelDigiSimLinkHandle, SimTrackHandle ); 

      ClustersForOutput->push_back( temp );
    }
    for ( unsigned int i = 0; i < outerHits.size(); i++ )
    {
      L1TkCluster< T > temp( outerHits.at(i), Id, 1 );
      if ( iEvent.isRealData() == false )
        theStackedTrackers->checkSimTrack( &temp, PixelDigiSimLinkHandle, SimTrackHandle ); 

      ClustersForOutput->push_back( temp );
    }

  } /// End of loop over detector elements

  /// Put output in the event
  iEvent.put( ClustersForOutput );
}

/// Retrieve hits from the event
/// Specialize template for PixelDigis
template<>
void L1TkClusterBuilder< Ref_PixelDigi_ >::RetrieveRawHits( std::map< DetId, std::vector< Ref_PixelDigi_ > > &mRawHits,
                                                                         const edm::Event& iEvent )
{
  mRawHits.clear();
  /// Loop over the tags used to identify hits in the cfg file
  std::vector< edm::InputTag >::iterator it;
  for ( it = rawHitInputTags.begin();
        it != rawHitInputTags.end();
        ++it )
  {
    /// For each tag, get the corresponding handle
    edm::Handle< edm::DetSetVector< PixelDigi > > HitHandle;
    iEvent.getByLabel( *it, HitHandle );

    edm::DetSetVector<PixelDigi>::const_iterator detsIter;
    edm::DetSet<PixelDigi>::const_iterator       hitsIter;

    /// Loop over detector elements identifying PixelDigis
    for ( detsIter = HitHandle->begin();
          detsIter != HitHandle->end();
          detsIter++ )
    {
      DetId id = detsIter->id;

      /// Is it Pixel?
      if ( id.subdetId()==1 || id.subdetId()==2 )
      {
        /// Loop over Digis in this specific detector element
        for ( hitsIter = detsIter->data.begin();
              hitsIter != detsIter->data.end();
              hitsIter++ )
        {
          if ( hitsIter->adc() >= ADCThreshold )
          {
            /// If the Digi is over threshold,
            /// accept it as a raw hit and put into map
            mRawHits[id].push_back( makeRefTo( HitHandle, id , hitsIter ) );
          } /// End of threshold selection
        } /// End of loop over digis
      } /// End of "is Pixel"
    } /// End of loop over detector elements
  } /// End of loop over tags
}

/// Retrieve hits from the event
/// Specialize template for SimHits
template<>
void L1TkClusterBuilder< Ref_PSimHit_ >::RetrieveRawHits( std::map< DetId, std::vector< Ref_PSimHit_ > > &mRawHits,
                                                                       const edm::Event& iEvent )
{
  mRawHits.clear();
  /// Loop over the tags used to identify hits in the cfg file
  for ( std::vector<edm::InputTag>::iterator it = rawHitInputTags.begin();
        it != rawHitInputTags.end();
        ++it )
  {
    /// For each tag, get the corresponding handle
    edm::Handle< edm::PSimHitContainer > HitHandle;
    iEvent.getByLabel( *it, HitHandle );

    /// Loop over the hits
    for ( unsigned int i=0 ; i != HitHandle->size() ; ++i )
    {
      DetId id = DetId(HitHandle->at(i).detUnitId());
      /// If Pixel, directly map it
      if ( id.subdetId()==1 || id.subdetId()==2 )
        mRawHits[id].push_back( edm::Ref<edm::PSimHitContainer>( HitHandle, i ) );
    } /// End of loop over hits
  } /// End of loop over tags
}

#endif

