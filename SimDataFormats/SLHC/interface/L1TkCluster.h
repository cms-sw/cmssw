/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
///                                      ///
/// Nicola Pozzobon, UNIPD               ///
/// Emmanuele Salvati, Cornell           ///
///                                      ///
/// 2011, June                           ///
/// 2011, October                        ///
/// 2012, August                         ///
/// 2013, January                        ///
/// ////////////////////////////////////////

#ifndef STACKED_TRACKER_L1TK_CLUSTER_FORMAT_H
#define STACKED_TRACKER_L1TK_CLUSTER_FORMAT_H

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"

#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"


  /** ************************ **/
  /**                          **/
  /**   DECLARATION OF CLASS   **/
  /**                          **/
  /** ************************ **/

  template< typename T >
  class L1TkCluster
  {
    public:
      /// Constructors
      L1TkCluster();
      L1TkCluster( std::vector< T > aHits, DetId aDetId, unsigned int aStackMember );

      /// Destructor
      ~L1TkCluster();

      /// Data members:   getABC( ... )
      /// Helper methods: findABC( ... )

      /// Hits in the Cluster
      std::vector< T > getHits() const;
      void             setHits( std::vector< T > aHits );

      /// Detector element
      DetId        getDetId() const;
      void         setDetId( DetId aDetId );
      unsigned int getStackMember() const;
      void         setStackMember( unsigned int aStackMember );

      /// MC Truth
      std::vector< edm::Ptr< SimTrack > > getSimTrackPtrs() const;
      std::vector< uint32_t >             getEventIds() const;
      void                                addSimTrack( const edm::Ptr< SimTrack > &trk) { theSimTracks.push_back(trk);}
      void                                addEventId( const uint32_t anId ) { theEventIds.push_back(anId); }
      bool                                isGenuine() const;
      bool                                isCombinatoric() const;
      bool                                isUnknown() const;
      int                                 findType() const;
      unsigned int                        findSimTrackId() const;

      /// Cluster width
      unsigned int findWidth() const;

      /// Single hit coordinates and position
      MeasurementPoint findHitLocalCoordinates( unsigned int hitIdx ) const;

      /// Average cluster coordinates and position
      MeasurementPoint findAverageLocalCoordinates() const;

      /// Information
      std::string print( unsigned int i=0 ) const;

    private:
      /// Data members
      std::vector< T >                    theHits;
      DetId                               theDetId;
      unsigned int                        theStackMember;
      std::vector< edm::Ptr< SimTrack > > theSimTracks;
      std::vector< uint32_t >             theEventIds;

  }; /// Close class

  /** ***************************** **/
  /**                               **/
  /**   IMPLEMENTATION OF METHODS   **/
  /**                               **/
  /** ***************************** **/

  /// Default Constructor
  /// NOTE: to be used with setSomething(...) methods
  template< typename T >
  L1TkCluster< T >::L1TkCluster()
  {
    /// Set default data members
    theHits.clear();
    theDetId = 0;
    theStackMember = 0;
    theSimTracks.clear();
  }

  /// Another Constructor
  template< typename T >
  L1TkCluster< T >::L1TkCluster( std::vector< T > aHits, DetId aDetId, unsigned int aStackMember )
  {
    /// Set default data members
    theHits.clear();
    for ( unsigned int j = 0; j < aHits.size(); j++ )
      theHits.push_back( aHits.at(j) );
    theDetId = aDetId;
    theStackMember = aStackMember;
    theSimTracks.clear();
  }

  /// Destructor
  template< typename T >
  L1TkCluster< T >::~L1TkCluster(){}

  /// Hits composing the cluster
  template< typename T >
  std::vector< T > L1TkCluster< T >::getHits() const { return theHits; }

  template< typename T >
  void L1TkCluster< T >::setHits( std::vector< T > aHits )
  {
    for ( unsigned int j=0; j<aHits.size(); j++ )
      theHits.push_back( aHits.at(j) );
  }

  /// Detector element information
  template< typename T >
  DetId L1TkCluster< T >::getDetId() const { return theDetId; }

  template< typename T >
  void L1TkCluster< T >::setDetId( DetId aDetId ) { theDetId = aDetId; }

  template< typename T >
  unsigned int L1TkCluster< T >::getStackMember() const { return theStackMember; }

  template< typename T >
  void L1TkCluster< T >::setStackMember( unsigned int aStackMember ) { theStackMember = aStackMember; }

  /// MC truth information
  template< typename T >
  std::vector< edm::Ptr< SimTrack > > L1TkCluster< T >::getSimTrackPtrs() const { return theSimTracks; }

  template< typename T >
  std::vector< uint32_t > L1TkCluster< T >::getEventIds() const { return theEventIds; }

  template< typename T >
  bool L1TkCluster< T >::isGenuine() const
  {
    /// Check how many event ID's
    std::vector< uint32_t > tempVecEvId = theEventIds;
    tempVecEvId.erase( std::unique( tempVecEvId.begin(), tempVecEvId.end() ), tempVecEvId.end() );
    if ( tempVecEvId.size() > 1 )
      return false;

    /// If all SimTracks are from the same event/BX ...
    /// GENUINE means that ALL hits could be associated to a
    /// SimTrack stored in the corresponding collection, AND
    /// all of these SimTracks are actually the same
    int prevTrack = -99999; // SimTrackId storage
    if ( theSimTracks.size() == 0 ) return false;
    for ( unsigned int k = 0; k < theSimTracks.size(); k++ )
    {
      edm::Ptr< SimTrack > curSimTrackPtr = theSimTracks.at(k);
      if ( curSimTrackPtr.isNull() )
        /// Unknown SimTrack means false
        return false;
      else
      {
        if ( theSimTracks.size() > 1 )
        {
          if ( prevTrack < 0 )
          {
            prevTrack = curSimTrackPtr->trackId();
          }

          if ( prevTrack != (int)curSimTrackPtr->trackId() )
            /// Two different known SimTracks means false
            return false;
 
          prevTrack = curSimTrackPtr->trackId();
        }
      }
    }
    /// If not false, then it is true
    return true;
  }

  template< typename T >
  bool L1TkCluster< T >::isCombinatoric() const
  {
    /// Check how many event ID's
    std::vector< uint32_t > tempVecEvId = theEventIds;
    tempVecEvId.erase( std::unique( tempVecEvId.begin(), tempVecEvId.end() ), tempVecEvId.end() );
    if ( tempVecEvId.size() > 1 )
      return true;

    /// If all SimTracks are from the same event/BX ...
    /// COMBINATORIC means that different SimTracks contribute
    /// to the cluster, which means that both a mixture of NULL
    /// pointers and good ones are present, or that all are
    /// good but there are more SimTracks associated to the cluster
    int prevTrack = -99999; // SimTrackId storage
    unsigned int numberNulls = 0; // Number of non-found SimTracks
    unsigned int numberGoods = 0; // Number of found SimTracks
    for ( unsigned int k = 0; k < theSimTracks.size(); k++ )
    {
      edm::Ptr< SimTrack > curSimTrackPtr = theSimTracks.at(k);
      if ( curSimTrackPtr.isNull() )
        numberNulls++;
      else
        numberGoods++;

      if ( numberNulls > 0 && numberGoods > 0 )
        /// Mixture of known and unknown SimTracks means true
        return true;

      if ( curSimTrackPtr.isNull() == false )
      {
        if ( theSimTracks.size() > 1 )
        {
          if ( prevTrack < 0 )
            prevTrack = curSimTrackPtr->trackId();

          if ( prevTrack != (int)curSimTrackPtr->trackId() )
            /// Two different known SimTracks means true
            return true;

          prevTrack = curSimTrackPtr->trackId();
        }
      }
    }

    if ( numberNulls > 0 && numberGoods == 0 )
      /// Only unknown SimTracks means unknown, hence false
      return false;

    /// If not true, then it is false
    /// This includes if ( theSimTracks.size() == 0 )
    return false;
  }

  template< typename T >
  bool L1TkCluster< T >::isUnknown() const
  {
    /// Check how many event ID's
    std::vector< uint32_t > tempVecEvId = theEventIds;
    tempVecEvId.erase( std::unique( tempVecEvId.begin(), tempVecEvId.end() ), tempVecEvId.end() );
    if ( tempVecEvId.size() > 1 )
      return false;

    /// If all SimTracks are from the same event/BX ...
    /// UNKNOWN means that all SimTracks pointers are NULL
    for ( unsigned int k = 0; k < theSimTracks.size(); k++ )
    {
      edm::Ptr< SimTrack > curSimTrackPtr = theSimTracks.at(k);
      if ( curSimTrackPtr.isNull() == false )
        /// A known SimTrack means false
        return false;
    }

    /// If not false, then it is true
    /// This includes if ( theSimTracks.size() == 0 )
    return true;
  }

  template< typename T >
  int L1TkCluster< T >::findType() const
  {
    if ( this->isGenuine() && theSimTracks.size()>0 )
      return theSimTracks.at(0)->type();
    return 999999999;
  }

  template< typename T >
  unsigned int L1TkCluster< T >::findSimTrackId() const
  {
    if ( this->isGenuine() && theSimTracks.size()>0 )
      return theSimTracks.at(0)->trackId();
    return 0;
  }

  /// Cluster width
  /// Specialize the template for PSimHits in *cc
  template<>
  unsigned int L1TkCluster< edm::Ref<edm::PSimHitContainer> >::findWidth() const;

  /// Cluster width
  /// Default template for PixelDigis
  template< typename T >
  unsigned int L1TkCluster< T >::findWidth() const
  {
    int rowMin = 99999999;
    int rowMax = 0;
    /// For broadside Clusters this is equivalent to theHits.size()
    /// but for 2d or neighbor Clusters this is only the actual size in RPhi
    for ( unsigned int i=0; i<theHits.size(); i++ )
    {
      if ( theHits.at(i)->row() < rowMin )
        rowMin = theHits.at(i)->row();
      if ( theHits.at(i)->row() > rowMax )
        rowMax = theHits.at(i)->row();
    }
    return abs( rowMax - rowMin + 1 ); /// This takes care of 1-Pixel clusters
  }

  /// Get hit local coordinates
  /// Specialize the template for PSimHits in *cc
  template<>
  MeasurementPoint L1TkCluster< edm::Ref< edm::PSimHitContainer > >::findHitLocalCoordinates( unsigned int hitIdx ) const;

  /// Get hit local coordinates
  /// Default template for PixelDigis
  template< typename T >
  MeasurementPoint L1TkCluster< T >::findHitLocalCoordinates( unsigned int hitIdx ) const
  {
    /// NOTE in this case, DO NOT add 0.5
    /// to get the center of the pixel
    MeasurementPoint mp( theHits.at(hitIdx)->row(), theHits.at(hitIdx)->column() );
    return mp;
  }


  /// Unweighted average local cluster coordinates
  /// Specialize the template for PSimHits in *cc
  template<>
  MeasurementPoint L1TkCluster< edm::Ref< edm::PSimHitContainer > >::findAverageLocalCoordinates() const;

  /// Unweighted average local cluster coordinates
  /// Default template for PixelDigis
  template< typename T >
  MeasurementPoint L1TkCluster< T >::findAverageLocalCoordinates() const
  {
    double averageCol = 0.0;
    double averageRow = 0.0;

    /// Loop over the hits and calculate the average coordinates
    if ( theHits.size() != 0 )
    {
      typename std::vector< T >::const_iterator hitIter;
      for ( hitIter = theHits.begin();
            hitIter != theHits.end();
            hitIter++ )
      {
        averageCol += (*hitIter)->column();
        averageRow += (*hitIter)->row();
      }
      averageCol /= theHits.size();
      averageRow /= theHits.size();
    }
    return MeasurementPoint( averageRow, averageCol );
  }


  /// Information
  template< typename T >
  std::string L1TkCluster< T >::print( unsigned int i ) const
  {
    std::string padding("");
    for ( unsigned int j=0; j!=i; ++j )
      padding+="\t";
    std::stringstream output;
    output<<padding<<"L1TkCluster:\n";
    padding+='\t';
    output << padding << "DetId: " << theDetId.rawId() << '\n';
    output << padding << "member: " << theStackMember << ", cluster size: " << theHits.size() << '\n';
    return output.str();
  }

  template< typename T >
  std::ostream& operator << (std::ostream& os, const L1TkCluster< T >& aL1TkCluster) { return ( os<<aL1TkCluster.print() ); }



#endif

