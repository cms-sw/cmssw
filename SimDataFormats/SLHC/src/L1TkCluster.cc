/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
///                                      ///
/// Nicola Pozzobon, UNIPD               ///
/// Emmanuele Salvati, Cornell           ///
///                                      ///
/// 2011, June                           ///
/// 2011, October                        ///
/// 2013, January                        ///
/// ////////////////////////////////////////

#include "SimDataFormats/SLHC/interface/L1TkCluster.h"

  /** ***************************** **/
  /**                               **/
  /**   IMPLEMENTATION OF METHODS   **/
  /**                               **/
  /** ***************************** **/

  /// Get cluster width
  /// Default template for PixelDigis in *.h
  /// Specialize the template for PSimHits
  template<>
  unsigned int L1TkCluster< edm::Ref< edm::PSimHitContainer > >::findWidth() const
  {
    return theHits.size();
  }

  /// Get hit local coordinates
  /// Default template for PixelDigis in *.h
  /// Specialize the template for PSimHits
  template<>
  MeasurementPoint L1TkCluster< edm::Ref< edm::PSimHitContainer > >::findHitLocalCoordinates( unsigned int hitIdx ) const
  {
    MeasurementPoint mp( 0, 0 ); /// Dummy values
    return mp;
  }
  
  /// Get hit local position
  /// Default template for PixelDigis in *.h
  /// Specialize the template for PSimHits
  template<>
  LocalPoint L1TkCluster< edm::Ref< edm::PSimHitContainer > >::findHitLocalPosition( const StackedTrackerGeometry *theStackedTracker,
                                                                                                  unsigned int hitIdx ) const
  {
    return theHits.at(hitIdx)->localPosition();
  }

  /// Get hit global position
  /// Default template for PixelDigis in *.h
  /// Specialize the template for PSimHits
  template<>
  GlobalPoint L1TkCluster< edm::Ref< edm::PSimHitContainer > >::findHitGlobalPosition( const StackedTrackerGeometry *theStackedTracker,
                                                                                                    unsigned int hitIdx ) const
  {
    const GeomDetUnit* geomDetUnit = theStackedTracker->idToDetUnit( theDetId, theStackMember );
    return geomDetUnit->surface().toGlobal( theHits.at(hitIdx)->localPosition() );
  }

  /// Unweighted average local cluster coordinates
  /// Default template for PixelDigis in *.h
  /// Specialize the template for PSimHits
  template<>
  MeasurementPoint L1TkCluster< edm::Ref< edm::PSimHitContainer > >::findAverageLocalCoordinates() const
  {
    MeasurementPoint mp( 0, 0 ); /// Dummy values
    return mp;
  }

  /// Collect MC truth
  /// Default template for PixelDigis in *.h
  /// Specialize the template for PSimHits
  template<>
  void L1TkCluster< edm::Ref< edm::PSimHitContainer > >::checkSimTrack( const StackedTrackerGeometry *theStackedTracker,
                                                                                     edm::Handle<edm::DetSetVector<PixelDigiSimLink> >  thePixelDigiSimLinkHandle,
                                                                                     edm::Handle<edm::SimTrackContainer>   simTrackHandle )
  {
    /// Loop over all the hits composing the L1TkCluster
    for ( unsigned int i = 0; i < theHits.size(); i++ ) {

      /// Get SimTrack Id and type
      unsigned int curSimTrkId = theHits.at(i)->trackId();

      /// This version of the collection of the SimTrack ID and PDG
      /// may not be fast and optimal, but is safer since the
      /// SimTrack ID is shifted by 1 wrt the index in the vector,
      /// and this may not be so true on a general basis...
      bool foundSimTrack = false;
      for ( unsigned int j = 0; j < simTrackHandle->size() && !foundSimTrack; j++ )
      {
        if ( simTrackHandle->at(j).trackId() == curSimTrkId )
        {
          foundSimTrack = true;
          edm::Ptr< SimTrack > testSimTrack( simTrackHandle, j );
          theSimTracks.push_back( testSimTrack );
        }
      }
      if ( !foundSimTrack )
      {
        edm::Ptr< SimTrack >* testSimTrack = new edm::Ptr< SimTrack >();
        theSimTracks.push_back( *testSimTrack );
      }
    } /// End of Loop over all the hits composing the L1TkCluster
  }

