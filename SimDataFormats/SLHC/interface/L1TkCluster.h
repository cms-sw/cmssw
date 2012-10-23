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
/// 2012, August                         ///
/// * modified to include Endcaps        ///
/// * fixed fake/non fake methods        ///
/// ////////////////////////////////////////

#ifndef STACKED_TRACKER_L1TK_CLUSTER_FORMAT_H
#define STACKED_TRACKER_L1TK_CLUSTER_FORMAT_H

#include "Geometry/CommonTopologies/interface/Topology.h" 
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"

#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerDetId.h"
#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerGeometry.h"

#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

namespace cmsUpgrades{

  /** ************************ **/
  /**                          **/
  /**   DECLARATION OF CLASS   **/
  /**                          **/
  /** ************************ **/

  template< typename T >
  class L1TkCluster {

    public:
      typedef std::vector< T >                        HitCollection;
      typedef typename HitCollection::const_iterator  HitCollectionIterator;

    private:
      /// Data members
      HitCollection       theHits;
      StackedTrackerDetId theDetId;
      unsigned int        theStackMember;
      unsigned int        theSimTrackId;
      bool                theGenuine;
      int                 theType;

    public:
      /// Constructors
      L1TkCluster();
      L1TkCluster( std::vector< T > aHits, StackedTrackerDetId aDetId, unsigned int aStackMember );
      /// Destructor
      ~L1TkCluster();

      /// //////////////////////// ///
      /// METHODS FOR DATA MEMBERS ///
      /// Hits in the Cluster
      HitCollection        getHits() const;
      void                 setHits( HitCollection aHits );
      /// Detector element
      StackedTrackerDetId  getDetId() const;
      void                 setDetId( StackedTrackerDetId aDetId );
      unsigned int         getStackMember() const;
      void                 setStackMember( unsigned int aStackMember );
      unsigned int         getStack() const;
      unsigned int         getLadderPhi() const;
      unsigned int         getLadderZ() const;
      unsigned int         getRingR() const;
      unsigned int         getRingPhi() const;
      /// Fake or not
      bool                 isGenuine() const;
      void                 setGenuine( bool aGenuine );
      int                  getType() const;
      void                 setType( int aType );
      unsigned int         getSimTrackId() const;
      void                 setSimTrackId( unsigned int aSimTrackId );

      /// ////////////// ///
      /// HELPER METHODS ///
      /// Cluster width
      unsigned int     getWidth() const;
      /// Single hit coordinates and position
      MeasurementPoint getHitLocalCoordinates( const T &hit ) const;
      LocalPoint       getHitLocalPosition( const GeomDetUnit* geom, const T &hit ) const;
      GlobalPoint      getHitPosition( const GeomDetUnit* geom, const T &hit ) const;
      /// Average cluster position
      MeasurementPoint getAverageLocalCoordinates() const;
      LocalPoint       getAverageLocalPosition( const cmsUpgrades::StackedTrackerGeometry *theStackedTracker ) const;
      GlobalPoint      getAveragePosition( const cmsUpgrades::StackedTrackerGeometry *theStackedTracker ) const;
      /// Check SimTracks
      void checkSimTrack( const cmsUpgrades::StackedTrackerGeometry *theStackedTracker,
                          edm::Handle<edm::DetSetVector<PixelDigiSimLink> >  thePixelDigiSimLinkHandle,
                          edm::Handle<edm::SimTrackContainer>   simTrackHandle );


      /// /////////////////// ///
      /// INFORMATIVE METHODS ///
      std::string print( unsigned int i=0 ) const;

  }; /// Close class



  /** ***************************** **/
  /**                               **/
  /**   IMPLEMENTATION OF METHODS   **/
  /**                               **/
  /** ***************************** **/

  /// ////////////////////////// ///
  /// CONSTRUCTORS & DESTRUCTORS ///
  /// ////////////////////////// ///

  /// Default Constructor
  /// NOTE: to be used with setSomething(...) methods
  template< typename T >
  cmsUpgrades::L1TkCluster< T >::L1TkCluster()
  {
    /// Set default data members
    theHits.clear();
    theDetId = 0;
    theStackMember = 0;
    theSimTrackId = 0;
    theGenuine = false;
    theType = -999999999;
  }

  /// Another Constructor
  template< typename T >
  cmsUpgrades::L1TkCluster< T >::L1TkCluster( std::vector< T > aHits, StackedTrackerDetId aDetId, unsigned int aStackMember )
  {
    /// Set default data members
    theHits.clear();
    for ( unsigned int j=0; j<aHits.size(); j++ )
      theHits.push_back(aHits.at(j));
    theDetId = aDetId;
    theStackMember = aStackMember;
    theSimTrackId = 0;
    theGenuine = false;
    theType = -999999999;
  }

  /// Destructor
  template< typename T >
  cmsUpgrades::L1TkCluster< T >::~L1TkCluster()
  {
    /// Nothing is done
  }



  /// //////////////////////// ///
  /// METHODS FOR DATA MEMBERS ///
  /// //////////////////////// ///

  /// Hits composing the cluster
  template< typename T >
  std::vector< T > cmsUpgrades::L1TkCluster< T >::getHits() const
  {
    return theHits;    
  }

  template< typename T >
  void cmsUpgrades::L1TkCluster< T >::setHits( HitCollection aHits )
  {
    for ( unsigned int j=0; j<aHits.size(); j++ )
      theHits.push_back( aHits.at(j) );
  }

  /// Detector element information
  template< typename T >
  StackedTrackerDetId cmsUpgrades::L1TkCluster< T >::getDetId() const
  {
    return theDetId;
  }

  template< typename T >
  void cmsUpgrades::L1TkCluster< T >::setDetId( StackedTrackerDetId aDetId )
  {
    theDetId = aDetId;
  }

  template< typename T >
  unsigned int cmsUpgrades::L1TkCluster< T >::getStackMember() const
  {
    return theStackMember;
  }

  template< typename T >
  void cmsUpgrades::L1TkCluster< T >::setStackMember( unsigned int aStackMember )
  {
    theStackMember = aStackMember;
  }

  template< typename T >
  unsigned int cmsUpgrades::L1TkCluster< T >::getStack() const
  {
    if (theDetId.isBarrel())
      return theDetId.iLayer();
    else
      return theDetId.iDisk();
  }

  template< typename T >
  unsigned int cmsUpgrades::L1TkCluster< T >::getLadderPhi() const
  {
    if (theDetId.isEndcap())
      return 999999; //std::cerr << " W A R N I N G ! Attempt to getLadderPhi() from an Endcap!" << std::endl;
    return theDetId.iPhi();
  }

  template< typename T >
  unsigned int cmsUpgrades::L1TkCluster< T >::getLadderZ() const
  {
    if (theDetId.isEndcap())
      return 999999; //std::cerr << " W A R N I N G ! Attempt to getLadderZ() from an Endcap!" << std::endl;
    return theDetId.iZ();
  }

  template< typename T >
  unsigned int cmsUpgrades::L1TkCluster< T >::getRingR() const
  {
    if (theDetId.isBarrel())
      return 999999; //std::cerr << " W A R N I N G ! Attempt to getRingR() from a Barrel!" << std::endl;
    return theDetId.iRing();
  }

  template< typename T >
  unsigned int cmsUpgrades::L1TkCluster< T >::getRingPhi() const
  {
    if (theDetId.isBarrel())
      return 999999; //std::cerr << " W A R N I N G ! Attempt to getRingPhi() from a Barrel!" << std::endl;
    return theDetId.iPhi();
  }

  /// Fake or not fake
  template< typename T >
  bool cmsUpgrades::L1TkCluster< T >::isGenuine() const
  {
    return theGenuine;
  }

  template< typename T >
  void cmsUpgrades::L1TkCluster< T >::setGenuine( bool aGenuine ) {
    theGenuine = aGenuine;
  }

  template< typename T >
  int cmsUpgrades::L1TkCluster< T >::getType() const
  {
    return theType;
  }

  template< typename T >
  void cmsUpgrades::L1TkCluster< T >::setType( int aType ) {
    theType = aType;
  }

  template< typename T >
  unsigned int cmsUpgrades::L1TkCluster< T >::getSimTrackId() const
  {
    return theSimTrackId;
  }

  template< typename T >
  void cmsUpgrades::L1TkCluster< T >::setSimTrackId( unsigned int aSimTrackId ) {
    theSimTrackId = aSimTrackId;
  }


  /// ////////////// ///
  /// HELPER METHODS ///
  /// ////////////// ///

  /// Cluster width
  /// Specialize the template for PSimHits in *cc
  template<>
  unsigned int cmsUpgrades::L1TkCluster< edm::Ref<edm::PSimHitContainer> >::getWidth() const;

  /// Cluster width
  /// Default template for PixelDigis
  template< typename T >
  unsigned int cmsUpgrades::L1TkCluster< T >::getWidth() const
  {
    int colMin = 99999999;
    int colMax = 0;
    /// For broadside Clusters this is equivalent to theHits.size()
    /// but for 2d or neighbor Clusters this is only the actual size in RPhi
    for ( unsigned int i=0; i<theHits.size(); i++ ) {
      if ( theHits.at(i)->column() < colMin ) colMin = theHits.at(i)->column();
      if ( theHits.at(i)->column() > colMax ) colMax = theHits.at(i)->column();
    }

    return abs( colMax - colMin + 1 ); /// This takes care of 1-Pixel clusters
  }

  /// Get hit local coordinates
  /// Specialize the template for PSimHits in *cc
  template<>
  MeasurementPoint cmsUpgrades::L1TkCluster< edm::Ref< edm::PSimHitContainer > >::getHitLocalCoordinates( const edm::Ref< edm::PSimHitContainer > &hit ) const;

  /// Get hit local coordinates
  /// Default template for PixelDigis
  template< typename T >
  MeasurementPoint cmsUpgrades::L1TkCluster< T >::getHitLocalCoordinates( const T &hit ) const
  {
    /// NOTE in this case, DO NOT add 0.5
    /// to get the center of the pixel
    MeasurementPoint mp( hit->row(), hit->column() );
    return  mp;
  }

  /// Get hit local position
  /// Specialize the template for PSimHits in *cc
  template<>
  LocalPoint cmsUpgrades::L1TkCluster< edm::Ref< edm::PSimHitContainer > >::getHitLocalPosition( const GeomDetUnit* geom,
                                                                                                 const edm::Ref< edm::PSimHitContainer > &hit ) const;

  /// Get hit local position
  /// Default template for PixelDigis
  template< typename T >
  LocalPoint cmsUpgrades::L1TkCluster< T >::getHitLocalPosition( const GeomDetUnit* geom,
                                                                 const T &hit ) const
  {
    /// Add 0.5 to get the center of the pixel
    MeasurementPoint mp( hit->row() + 0.5, hit->column() + 0.5 );
    return geom->topology().localPosition( mp );
  }

  /// Get hit global position
  /// Specialize the template for PSimHits in *cc
  template<>
  GlobalPoint cmsUpgrades::L1TkCluster< edm::Ref< edm::PSimHitContainer > >::getHitPosition( const GeomDetUnit* geom,
                                                                                             const edm::Ref< edm::PSimHitContainer > &hit ) const;

  /// Get hit global position
  /// Default template for PixelDigis
  template< typename T >
  GlobalPoint cmsUpgrades::L1TkCluster< T >::getHitPosition( const GeomDetUnit* geom,
                                                             const T &hit ) const
  {
    /// Add 0.5 to get the center of the pixel
    MeasurementPoint mp( hit->row() + 0.5, hit->column() + 0.5 );
    return geom->surface().toGlobal( geom->topology().localPosition( mp ) ) ;
  }

  /// Unweighted average local cluster coordinates
  /// Specialize the template for PSimHits in *cc
  template<>
  MeasurementPoint cmsUpgrades::L1TkCluster< edm::Ref< edm::PSimHitContainer > >::getAverageLocalCoordinates() const;

  /// Unweighted average local cluster coordinates
  /// Default template for PixelDigis
  template< typename T >
  MeasurementPoint cmsUpgrades::L1TkCluster< T >::getAverageLocalCoordinates() const
  {
    double averageCol = 0.0;
    double averageRow = 0.0;

    /// Loop over the hits and calculate the average coordinates
    if ( theHits.size() != 0 ) {
      for ( HitCollectionIterator hits_itr = theHits.begin();
            hits_itr != theHits.end();
            hits_itr++ ) {
        averageCol += (*hits_itr)->column();
        averageRow += (*hits_itr)->row();
      }
      averageCol /= theHits.size();
      averageRow /= theHits.size();
    }
    return MeasurementPoint( averageRow, averageCol );
  }

  /// Unweighted average local cluster position
  template< typename T >
  LocalPoint cmsUpgrades::L1TkCluster< T >::getAverageLocalPosition( const cmsUpgrades::StackedTrackerGeometry *theStackedTracker ) const
  {
    double averageX = 0.0;
    double averageY = 0.0;

    /// Loop over the hits and calculate the average coordinates
    if ( theHits.size() != 0 ) {
      for ( HitCollectionIterator hits_itr = theHits.begin();
            hits_itr != theHits.end();
            hits_itr++ ) {
        const GeomDetUnit* det = theStackedTracker->idToDetUnit( this->getDetId() , theStackMember );
        LocalPoint thisHitPosition = getHitLocalPosition( det, *hits_itr );
        averageX += thisHitPosition.x();
        averageY += thisHitPosition.y();
      }
      averageX /= theHits.size();
      averageY /= theHits.size();
    }
    return LocalPoint( averageX, averageY );
  }

  /// Unweighted average cluster position
  template< typename T >
  GlobalPoint cmsUpgrades::L1TkCluster< T >::getAveragePosition( const cmsUpgrades::StackedTrackerGeometry *theStackedTracker ) const
  {
    double averageX = 0.0;
    double averageY = 0.0;
    double averageZ = 0.0;

    /// Loop over the hits and calculate the average coordinates
    if ( theHits.size() != 0 ) {
      for ( HitCollectionIterator hits_itr = theHits.begin();
            hits_itr != theHits.end();
            hits_itr++ ) {
        const GeomDetUnit* det = theStackedTracker->idToDetUnit( this->getDetId() , theStackMember );
        GlobalPoint thisHitPosition = getHitPosition( det, *hits_itr );
        averageX += thisHitPosition.x();
        averageY += thisHitPosition.y();
        averageZ += thisHitPosition.z();
      }
      averageX /= theHits.size();
      averageY /= theHits.size();
      averageZ /= theHits.size();
    }
    return GlobalPoint( averageX, averageY, averageZ );
  }

  /// WARNING
  /// Weighted methods are not
  /// on the current wishlist

  /// Check SimTracks
  /// Specify Template for PSimHits in *.cc file
  template<>
  void cmsUpgrades::L1TkCluster< edm::Ref< edm::PSimHitContainer > >::checkSimTrack( const cmsUpgrades::StackedTrackerGeometry *theStackedTracker,
                                                                                     edm::Handle<edm::DetSetVector<PixelDigiSimLink> >  thePixelDigiSimLinkHandle,
                                                                                     edm::Handle<edm::SimTrackContainer>   simTrackHandle );

  /// Default template for PixelDigis
  template< typename T >
  void cmsUpgrades::L1TkCluster< T >::checkSimTrack( const cmsUpgrades::StackedTrackerGeometry *theStackedTracker,
                                                     edm::Handle<edm::DetSetVector<PixelDigiSimLink> >  thePixelDigiSimLinkHandle,
                                                     edm::Handle<edm::SimTrackContainer>   simTrackHandle )
  {
    /// Discrimination between MC and data in Builder!

    /// Define a couple of vectors used
    /// to store the values of the SimTrack Id
    /// and of the associated particle
    std::vector< unsigned int > simTrkIdVec; simTrkIdVec.clear();
    std::vector< int > simTrkPdgVec;         simTrkPdgVec.clear();

    /// Get the PixelDigiSimLink
    const DetId detId = theStackedTracker->idToDet( theDetId, theStackMember )->geographicalId();
    edm::DetSet<PixelDigiSimLink> thisDigiSimLink = (*thePixelDigiSimLinkHandle)[detId.rawId()];
    edm::DetSet<PixelDigiSimLink>::const_iterator iterSimLink;
    /// Loop over all the hits composing the L1TkCluster
    for ( unsigned int i = 0; i < theHits.size(); i++ ) {

      /// Loop over PixelDigiSimLink
      for ( iterSimLink = thisDigiSimLink.data.begin();
            iterSimLink != thisDigiSimLink.data.end();
            iterSimLink++ ) {

        /// Threshold (redundant, already applied within L1TkClusterBuilder)
        //if ( theHit.adc() <= 30 ) continue;
        /// Find the link and, if there's not, skip
        if ( (int)iterSimLink->channel() != theHits.at(i)->channel() ) continue;

        /// Get SimTrack Id and type
        unsigned int curSimTrkId = iterSimLink->SimTrackId();

        /// This version of the collection of the SimTrack ID and PDG
        /// may not be fast and optimal, but is safer since the
        /// SimTrack ID is shifted by 1 wrt the index in the vector,
        /// and this may not be so true on a general basis...
        int curSimTrkPdg = -9999999;
        bool fastExit = false;
        for ( unsigned int j = 0; j < simTrackHandle->size() && !fastExit; j++ ){
          if ( simTrackHandle->at(j).trackId() ==  curSimTrkId ) {
            fastExit = true;
            curSimTrkPdg = simTrackHandle->at(j).type();
          }
        }
        if (!fastExit) continue; /// This prevents to fill the vector if the SimTrack is not found

        simTrkIdVec.push_back( curSimTrkId );
        simTrkPdgVec.push_back( curSimTrkPdg );
      }
    } /// End of Loop over all the hits composing the L1TkCluster

    bool tempGenuine = true;
    int tempType = 0;
    unsigned int tempSimTrack = 0;
    if ( simTrkPdgVec.size() != 0 ) {
      tempType = simTrkPdgVec.at(0);
      tempSimTrack = simTrkIdVec.at(0);
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
  }




  /// /////////////////// ///
  /// INFORMATIVE METHODS ///
  /// /////////////////// ///

  template< typename T >
  std::string cmsUpgrades::L1TkCluster< T >::print( unsigned int i ) const {
    std::string padding("");
    for ( unsigned int j=0; j!=i; ++j )padding+="\t";
    std::stringstream output;
    output<<padding<<"L1TkCluster:\n";
    padding+='\t';
    output << padding << "StackedTrackerDetId: " << theDetId << '\n';
    output << padding << "member: " << theStackMember << ", cluster size: " << theHits.size() << '\n';
    return output.str();
  }

  template< typename T >
  std::ostream& operator << (std::ostream& os, const cmsUpgrades::L1TkCluster< T >& aL1TkCluster) {
    return (os<<aL1TkCluster.print() );
  }

} /// Close namespace

#endif




