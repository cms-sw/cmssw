/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// Written by:                          ///
/// Nicola Pozzobon                      ///
/// UNIPD                                ///
/// 2011, June                           ///
///                                      ///
/// Modified by:                         ///
/// Emmanuele Salvati, Nicola Pozzobon   ///
/// UNIPD, Cornell                       ///
/// 2011, October                        ///
/// * added fake/non fake methods        ///
/// ////////////////////////////////////////

#include "SimDataFormats/SLHC/interface/L1TkCluster.h"

namespace cmsUpgrades {

  /** ***************************** **/
  /**                               **/
  /**   IMPLEMENTATION OF METHODS   **/
  /**                               **/
  /** ***************************** **/

  /// ////////////// ///
  /// HELPER METHODS ///
  /// ////////////// ///

  /// Get cluster width
  /// Default template for PixelDigis in *.h
  /// Specialize the template for PSimHits
  template<>
  unsigned int cmsUpgrades::L1TkCluster< edm::Ref< edm::PSimHitContainer > >::getWidth() const
  {
    return theHits.size();
  }

  /// Get hit local coordinates
  /// Default template for PixelDigis in *.h
  /// Specialize the template for PSimHits
  template<>
  MeasurementPoint cmsUpgrades::L1TkCluster< edm::Ref< edm::PSimHitContainer > >::getHitLocalCoordinates( const edm::Ref< edm::PSimHitContainer > &hit ) const
  {
    MeasurementPoint mp( 0, 0 ); /// Dummy values
    return  mp;
  }
  
  /// Get hit local position
  /// Default template for PixelDigis in *.h
  /// Specialize the template for PSimHits
  template<>
  LocalPoint cmsUpgrades::L1TkCluster< edm::Ref< edm::PSimHitContainer > >::getHitLocalPosition( const GeomDetUnit* geom,
                                                                                                 const edm::Ref< edm::PSimHitContainer > &hit ) const
  {
    return hit->localPosition();
  }

  /// Get hit global position
  /// Default template for PixelDigis in *.h
  /// Specialize the template for PSimHits
  template<>
  GlobalPoint cmsUpgrades::L1TkCluster< edm::Ref< edm::PSimHitContainer > >::getHitPosition( const GeomDetUnit* geom,
                                                                                             const edm::Ref< edm::PSimHitContainer > &hit ) const
  {
    return geom->surface().toGlobal( hit->localPosition() ) ;
  }

  /// Unweighted average local cluster coordinates
  /// Default template for PixelDigis in *.h
  /// Specialize the template for PSimHits
  template<>
  MeasurementPoint cmsUpgrades::L1TkCluster< edm::Ref< edm::PSimHitContainer > >::getAverageLocalCoordinates() const
  {
    MeasurementPoint mp( 0, 0 ); /// Dummy values
    return  mp;
  }

  /// Fake or non fake?
  /// Default template for PixelDigis in *.h
  /// Specialize the template for PSimHits
  template<>
  void cmsUpgrades::L1TkCluster< edm::Ref< edm::PSimHitContainer > >::checkSimTrack( const cmsUpgrades::StackedTrackerGeometry *theStackedTracker,
                                                                                     edm::Handle<edm::DetSetVector<PixelDigiSimLink> >  thePixelDigiSimLinkHandle,
                                                                                     edm::Handle<edm::SimTrackContainer>   simTrackHandle )
  {
    /// Define a couple of vectors used
    /// to store the values of the SimTrack Id
    /// and of the associated particle
    std::vector< unsigned int > simTrkIdVec; simTrkIdVec.clear();
    std::vector< int > simTrkPdgVec;         simTrkPdgVec.clear();

    /// Loop over all the hits composing the L1TkCluster
    for ( unsigned int i = 0; i < theHits.size(); i++ ) {

      /// Get SimTrack Id and type
      unsigned int curSimTrkId = theHits.at(i)->trackId();
      int curSimTrkPdg = theHits.at(i)->particleType();
      simTrkIdVec.push_back( curSimTrkId );
      simTrkPdgVec.push_back( curSimTrkPdg );
    } /// End of Loop over all the hits composing the L1TkCluster

    bool tempGenuine = true;
    int tempType = 0;
    unsigned int tempSimTrack = 0;
    if ( simTrkPdgVec.size() != 0 ) {
      tempType = simTrkPdgVec.at(0);
      for ( unsigned int j = 1; j != simTrkPdgVec.size(); j++ ) {
        if ( simTrkIdVec.at(j) > simTrackHandle->size() )
          tempGenuine = false;
        else if ( simTrkPdgVec.at(j) != tempType )
          tempGenuine = false;
        else {
          tempType = simTrkPdgVec.at(j);
          tempSimTrack = simTrkIdVec.at(j);
        }
      } /// End of loop over vector of PDG codes
    }
    else
      tempGenuine = false;

    /// Set the variable
    this->setGenuine( tempGenuine );
    if ( tempGenuine ) {
      this->setType( tempType );
      this->setSimTrackId( tempSimTrack );
    }
    //std::cerr << this->isGenuine() << " " << this->getType() << " " << this->getSimTrackId() << std::endl;
  }


} /// Close namespace



