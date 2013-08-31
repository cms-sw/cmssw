/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
///                                      ///
/// Nicola Pozzobon,  UNIPD              ///
///                                      ///
/// 2011, September                      ///
/// ////////////////////////////////////////

#ifndef L1TK_TRACK_BUILDER_H
#define L1TK_TRACK_BUILDER_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerDetId.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/TrackingAlgorithm.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/TrackingAlgorithmRecord.h"
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
class L1TkTrackBuilder : public edm::EDProducer
{
  public:

    /// Constructor
    explicit L1TkTrackBuilder( const edm::ParameterSet& iConfig );

    /// Destructor;
    ~L1TkTrackBuilder();

  private:
    /// Data members
    /// Geometry
//    const StackedTrackerGeometry            *theStackedTracker;
    /// Tracking algorithm
    edm::ESHandle< TrackingAlgorithm< T > > TrackingAlgoHandle;
    edm::InputTag                           L1TkStubsInputTag;

    /// Other stuff
    bool enterAssociativeMemoriesWorkflow;

    /// ///////////////// ///
    /// MANDATORY METHODS ///
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
L1TkTrackBuilder< T >::L1TkTrackBuilder( const edm::ParameterSet& iConfig )
{
  produces< std::vector< L1TkTrack< T > > >( "Seeds" );
  produces< std::vector< L1TkTrack< T > > >( "NoDup" );
  L1TkStubsInputTag = iConfig.getParameter< edm::InputTag >("L1TkStubsBricks");
  enterAssociativeMemoriesWorkflow = iConfig.getParameter< bool >("AssociativeMemories");
}

/// Destructor
template< typename T >
L1TkTrackBuilder< T >::~L1TkTrackBuilder() {}

/// Begin run
template< typename T >
void L1TkTrackBuilder< T >::beginRun( const edm::Run& run, const edm::EventSetup& iSetup )
{
  /// Get the tracking algorithm 
  iSetup.get< TrackingAlgorithmRecord >().get( TrackingAlgoHandle );
  /// Print some information when loaded
  std::cout  << "L1TkTrackBuilder<" << templateNameFinder<T>() << "> loaded modules:"
             << "\n\tTrackingAlgorithm:\t" << TrackingAlgoHandle->AlgorithmName()
             << std::endl;
}

/// End run
template< typename T >
void L1TkTrackBuilder< T >::endRun( const edm::Run& run, const edm::EventSetup& iSetup ) {}

/// Implement the producer
template< typename T >
void L1TkTrackBuilder< T >::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  /// Prepare output
  /// The temporary collection is used to store tracks
  /// before removal of duplicates
  std::vector< L1TkTrack< T > > tempTrackCollection;
  tempTrackCollection.clear();
  std::auto_ptr< std::vector< L1TkTrack< T > > > L1TkTracksSeedsForOutput( new std::vector< L1TkTrack< T > > );
  std::auto_ptr< std::vector< L1TkTrack< T > > > L1TkTracksForOutput( new std::vector< L1TkTrack< T > > );
  std::auto_ptr< std::vector< L1TkTrack< T > > > L1TkTracksForOutputPurged( new std::vector< L1TkTrack< T > > );

  /// Get the Stubs already stored away
  edm::Handle< std::vector< L1TkStub< T > > > L1TkStubHandle;
  iEvent.getByLabel( L1TkStubsInputTag, L1TkStubHandle );

  if ( enterAssociativeMemoriesWorkflow )
  {
    /// Enter AM 
    std::cerr << "TEST: AM workflow" << std::endl;
    TrackingAlgoHandle->PatternFinding();
    TrackingAlgoHandle->PatternRecognition();

  } /// End AM workflow
  else
  {
    /// Tracklet-based approach

    /// Create the Seeds and map the Stubs per Sector/Wedge
    std::vector< L1TkTrack< T > > theseSeeds;
    std::map< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< L1TkStub< T > > > > *stubSectorWedgeMap;
    stubSectorWedgeMap = new std::map< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< L1TkStub< T > > > >();
    TrackingAlgoHandle->CreateSeeds( theseSeeds, stubSectorWedgeMap, L1TkStubHandle );

    /// Store the number of sectors
    unsigned int nSectors = TrackingAlgoHandle->ReturnNumberOfSectors();
    unsigned int nWedges = TrackingAlgoHandle->ReturnNumberOfWedges();

    /// Here all the seeds are available and all the stubs are stored
    /// in a sector-wise map: loop over seeds, find the sector, attach stubs
    /// Store the seeds menawhile ...
    for ( unsigned int it = 0; it < theseSeeds.size(); it++ )
    {
      L1TkTrack< T > curSeed = theseSeeds.at(it);

      /// Check SimTrack if needed
      if ( iEvent.isRealData() == false )
        curSeed.checkSimTrack();

      /// Immediately store the seed as it is being modified later!
      L1TkTracksSeedsForOutput->push_back( curSeed );

      /// Find the sector and the stubs to be attached
      unsigned int curSector0 = curSeed.getSector() + nSectors; /// This is to use the %nSectors later
      unsigned int curWedge0 = curSeed.getWedge();

      /// Loop over the sector and its two neighbors
      for ( unsigned int iSector = 0; iSector < 2; iSector++ )
      {
        for ( unsigned int iWedge = 0; iWedge < 2; iWedge++)
        {
          /// Find the correct sector index
          unsigned int curSector = ( curSector0 + iSector -1 )%nSectors;
          int curWedge = curWedge0 + iWedge - 1;
          if ( curWedge < 0 || curWedge >= (int)nWedges )
            continue;

          std::pair< unsigned int, unsigned int > sectorWedge = std::make_pair( curSector, (unsigned int)curWedge );

          /// Skip sector if empty
          if ( stubSectorWedgeMap->find( sectorWedge ) == stubSectorWedgeMap->end() )
            continue;

          std::vector< edm::Ptr< L1TkStub< T > > > stubsToAttach = stubSectorWedgeMap->find( sectorWedge )->second;

          /// Loop over the stubs in the Sector
          for ( unsigned int iv = 0; iv < stubsToAttach.size(); iv++ )
          {
            /// Here we have same-sector-different-SL seed and stubs
            TrackingAlgoHandle->AttachStubToSeed( curSeed, stubsToAttach.at(iv) );
          } /// End of nested loop over stubs in the Sector
        }
      } /// End of loop over the sector and its two neighbors

      /// Here the seed is completed with all its matched stubs
      /// The seed is now a track and it is time to fit it
      TrackingAlgoHandle->FitTrack( curSeed );

      /// Store the fitted track in the output
      L1TkTracksForOutput->push_back( curSeed );

    } /// End of loop over seeds

  } /// End of non-AM

  /// Remove duplicates
  std::vector< bool > toBeDeleted;
  for ( unsigned int i = 0; i < L1TkTracksForOutput->size(); i++ )
  {
    toBeDeleted.push_back( false );
  }

  for ( unsigned int i = 0; i < L1TkTracksForOutput->size(); i++ )
  {
    /// This check is necessary as the bool may be reset in a previous iteration
    if ( toBeDeleted.at(i) )
      continue;

    /// Nested loop to compare tracks with each other
    for ( unsigned int j = i+1 ; j < L1TkTracksForOutput->size(); j++ )
    {
      /// This check is necessary as the bool may be reset in a previous iteration
      if ( toBeDeleted.at(j) )
        continue;

      /// Check if they are the same track
      if ( L1TkTracksForOutput->at(i).isTheSameAs( L1TkTracksForOutput->at(j) ) )
      {
        /// Check they both have > 3 stubs
        if ( L1TkTracksForOutput->at(i).getStubPtrs().size() < 3 ||
             L1TkTracksForOutput->at(j).getStubPtrs().size() < 3 )
          continue;

        /// Compare Chi2
        if ( L1TkTracksForOutput->at(i).getChi2() > L1TkTracksForOutput->at(j).getChi2() )
        {
          toBeDeleted[i] = true;
        }
        else
        {
          toBeDeleted[j] = true;
        }
        continue;
      }
    }

    if ( toBeDeleted.at(i) ) continue; /// Is it really necessary?
  }

  /// Store only the non-deleted tracks
  for ( unsigned int i = 0; i < L1TkTracksForOutput->size(); i++ )
  {
    if ( toBeDeleted.at(i) ) continue;

    L1TkTrack< T > tempL1TkTrack = L1TkTracksForOutput->at(i);

    /// Check SimTrack if needed
    if ( iEvent.isRealData() == false )
      tempL1TkTrack.checkSimTrack();

    L1TkTracksForOutputPurged->push_back( tempL1TkTrack );
  }

  /// Put in the event content
  iEvent.put( L1TkTracksSeedsForOutput, "Seeds" );
  iEvent.put( L1TkTracksForOutputPurged, "NoDup" );
}

#endif

