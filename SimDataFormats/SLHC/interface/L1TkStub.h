/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// Written by:                          ///
/// Andrew W. Rose                       ///
/// 2008                                 ///
///                                      ///
/// Modified by:                         ///
/// Nicola Pozzobon                      ///
/// UNIPD                                ///
/// 2010, June; 2011, June; 2012, Oct    ///
///                                      ///
/// Added features:                      ///
/// Higher threshold flag in 3_3_6 has   ///
/// been replaced by rough Pt            ///
/// calculation.                         ///
/// LocalStub and GlobalStub unified,    ///
/// Global information available through ///
/// a flag in configuration file.        ///
/// Introduced a switch on Barrel/Endcap ///
/// for the rough Pt calculation.        ///
/// ////////////////////////////////////////

#ifndef STACKED_TRACKER_L1TK_STUB_FORMAT_H
#define STACKED_TRACKER_L1TK_STUB_FORMAT_H

#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerGeometry.h"
#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerDetId.h"
#include "SLHCUpgradeSimulations/Utilities/interface/constants.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/Topology.h" 

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/SLHC/interface/L1TkCluster.h"

namespace cmsUpgrades{

  /** ************************ **/
  /**                          **/
  /**   DECLARATION OF CLASS   **/
  /**                          **/
  /** ************************ **/

  template< typename T >
  class L1TkStub {

    public:
      typedef L1TkCluster< T >                                        L1TkClusterType;
      typedef std::vector< L1TkClusterType >                          L1TkClusterCollection;
      typedef edm::Ptr< L1TkClusterType >                             L1TkClusterPtrType;
      typedef std::vector< L1TkClusterPtrType >                       L1TkClusterPtrCollection;
      typedef typename L1TkClusterPtrCollection::const_iterator       L1TkClusterPtrCollectionIterator;

    private:
      /// Data members
      double                   theRoughPt;
      StackedTrackerDetId      theDetId;
      L1TkClusterPtrCollection theClusters;
      GlobalPoint              thePosition;
      GlobalVector             theDirection; /// NOTE this is not normalized to 1 !!!
      unsigned int             theSimTrackId;
      bool                     theGenuine;
      int                      theType;

    public:
      /// Constructors
      L1TkStub();
      L1TkStub( StackedTrackerDetId aDetId );
      /// Destructor
      ~L1TkStub();

      /// //////////////////////// ///
      /// METHODS FOR DATA MEMBERS ///
      /// Roughly measured Pt
      double                    getRoughPt() const;
      void                      setRoughPt( double aRoughPt );
      /// Hits composing the Stub
      /// NOT hits composing Clusters composing the Stub
      L1TkClusterCollection     getClusters() const;
      const L1TkClusterType&    getCluster( unsigned int hitIdentifier ) const;
      const L1TkClusterPtrType& getClusterRef( unsigned int hitIdentifier ) const; /// Use it to check that objects are really the same!
      void                      addCluster( L1TkClusterPtrType aL1TkCluster );
      /// NOTE they are added and stored as edm::Ptr< Cluster > but
      /// returned as just Cluster for backward compatibility with
      /// HitMatching Algorithms
      /// Detector element
      StackedTrackerDetId       getDetId() const;
      void                      setDetId( StackedTrackerDetId aDetId );
      unsigned int              getStack() const;
      unsigned int              getLadderPhi() const;
      unsigned int              getLadderZ() const;
      unsigned int              getRingR() const;
      unsigned int              getRingPhi() const;
      /// Position and direction
      GlobalPoint               getPosition() const;
      GlobalVector              getDirection() const;
      void                      setPosition( GlobalPoint aPosition );
      void                      setDirection( GlobalVector aDirection );
      /// Fake or not
      bool                      isGenuine() const;
      void                      setGenuine( bool aGenuine );
      int                       getType() const;
      void                      setType( int aType );
      unsigned int              getSimTrackId() const;
      void                      setSimTrackId( unsigned int aSimTrackId );

      /// ////////////// ///
      /// HELPER METHODS ///
      /// Fake or not
      void checkSimTrack();

      /// Fit Stub as in Builder
      /// To be used for out-of-Builder Stubs
      void fitStub( double aMagneticFieldStrength, const cmsUpgrades::StackedTrackerGeometry *theStackedTracker );

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
  template< typename T >
  cmsUpgrades::L1TkStub< T >::L1TkStub()
  {
    /// Set default data members
    theRoughPt = 0;
    theDetId = 0;
    theClusters.clear();
    thePosition = GlobalPoint(0,0,0);
    theDirection = GlobalVector(0,0,0);
    theSimTrackId = 0;
    theGenuine = false;
    theType = -999999999;
  }

  /// Another Constructor
  template< typename T >
  cmsUpgrades::L1TkStub< T >::L1TkStub( StackedTrackerDetId aDetId )
  {
    /// Set default data members
    theRoughPt = 0;
    theDetId = aDetId;
    theClusters.clear();
    thePosition = GlobalPoint(0,0,0);
    theDirection = GlobalVector(0,0,0);
    theSimTrackId = 0;
    theGenuine = false;
    theType = -999999999;
  }

  /// Destructor
  template< typename T >
  cmsUpgrades::L1TkStub< T >::~L1TkStub()
  {
    /// Nothing is done
  }



  /// //////////////////////// ///
  /// METHODS FOR DATA MEMBERS ///
  /// //////////////////////// ///

  /// Roughly measured Pt
  template< typename T >
  double cmsUpgrades::L1TkStub< T >::getRoughPt() const {
    return theRoughPt;
  }

  template< typename T >
  void cmsUpgrades::L1TkStub< T >::setRoughPt ( double aRoughPt ) {
    theRoughPt = aRoughPt;
  }

  /// Get the Clusters composing the Stub
  /// NOT as Pointers!!!
  template< typename T >
  std::vector< cmsUpgrades::L1TkCluster< T > > cmsUpgrades::L1TkStub< T >::getClusters( ) const
  {
    std::vector< L1TkCluster< T > > tempClusters;
    tempClusters.clear();
    for ( unsigned int i=0; i< theClusters.size(); i++ )
      tempClusters.push_back( *(theClusters.at(i)) );
    return tempClusters;
  }

  /// Get a Cluster
  template< typename T >
  const cmsUpgrades::L1TkCluster< T >& cmsUpgrades::L1TkStub< T >::getCluster( unsigned int hitIdentifier ) const
  {
    for ( L1TkClusterPtrCollectionIterator i = theClusters.begin(); i!= theClusters.end(); ++i ) {
      if ( (*i)->getStackMember() == hitIdentifier ) return **i;
    }
    return L1TkCluster< T >();
  }

  /// Get the Pointer to a Cluster
  template< typename T >
  const edm::Ptr< cmsUpgrades::L1TkCluster< T > >& cmsUpgrades::L1TkStub< T >::getClusterRef( unsigned int hitIdentifier ) const
  {
    for ( L1TkClusterPtrCollectionIterator i = theClusters.begin(); i!= theClusters.end(); ++i ) {
      if ( (*i)->getStackMember() == hitIdentifier ) return *i;
    }
    return edm::Ptr< L1TkCluster< T > >();
  }

  /// Add the Clusters to the candidate Stub
  template< typename T >
  void cmsUpgrades::L1TkStub< T >::addCluster( edm::Ptr< L1TkCluster< T > > aL1TkCluster )
  {
    /// NOTE: this must be used ONLY as it is used
    /// within the L1TkStubBuilder!
    /// So, pushing back in the right order!!
    theClusters.push_back( aL1TkCluster );
  }

  /// Detector element
  template< typename T >
  StackedTrackerDetId cmsUpgrades::L1TkStub< T >::getDetId() const
  {
    return theDetId;
  }

  template< typename T >
  void cmsUpgrades::L1TkStub< T >::setDetId( StackedTrackerDetId aDetId )
  {
    theDetId = aDetId;    
  }

  template< typename T >
  unsigned int cmsUpgrades::L1TkStub< T >::getStack() const
  {
    if (theDetId.isBarrel())
      return theDetId.iLayer();
    else
      return theDetId.iDisk();
  }

  template< typename T >
  unsigned int cmsUpgrades::L1TkStub< T >::getLadderPhi() const
  {
    if (theDetId.isEndcap())
      return 999999; //std::cerr << " W A R N I N G ! Attempt to getLadderPhi() from an Endcap!" << std::endl;
    return theDetId.iPhi();
  }

  template< typename T >
  unsigned int cmsUpgrades::L1TkStub< T >::getLadderZ() const
  {
    if (theDetId.isEndcap())
      return 999999; //std::cerr << " W A R N I N G ! Attempt to getLadderZ() from an Endcap!" << std::endl;
    return theDetId.iZ();
  }

  template< typename T >
  unsigned int cmsUpgrades::L1TkStub< T >::getRingR() const
  {
    if (theDetId.isBarrel())
      return 999999; //std::cerr << " W A R N I N G ! Attempt to getRingR() from a Barrel!" << std::endl;
    return theDetId.iRing();
  }

  template< typename T >
  unsigned int cmsUpgrades::L1TkStub< T >::getRingPhi() const
  {
    if (theDetId.isBarrel())
      return 999999; //std::cerr << " W A R N I N G ! Attempt to getRingPhi() from a Barrel!" << std::endl;
    return theDetId.iPhi();
  }

  /// Position and direction of the Stub
  template< typename T >
  GlobalPoint cmsUpgrades::L1TkStub< T >::getPosition() const
  {
    return thePosition;
  }

  template< typename T >
  GlobalVector cmsUpgrades::L1TkStub< T >::getDirection() const
  {
    return theDirection;
  }

  template< typename T >
  void cmsUpgrades::L1TkStub< T >::setPosition( GlobalPoint aPosition )
  {
    thePosition = aPosition;
  }

  template< typename T >
  void cmsUpgrades::L1TkStub< T >::setDirection( GlobalVector aDirection )
  {
    theDirection = aDirection;
  }

  /// Fake or not fake
  template< typename T >
  bool cmsUpgrades::L1TkStub< T >::isGenuine() const
  {
    return theGenuine;
  }

  template< typename T >
  void cmsUpgrades::L1TkStub< T >::setGenuine( bool aGenuine ) {
    theGenuine = aGenuine;
  }

  template< typename T >
  int cmsUpgrades::L1TkStub< T >::getType() const
  {
    return theType;
  }

  template< typename T >
  void cmsUpgrades::L1TkStub< T >::setType( int aType ) {
    theType = aType;
  }

  template< typename T >
  unsigned int cmsUpgrades::L1TkStub< T >::getSimTrackId() const
  {
    return theSimTrackId;
  }

  template< typename T >
  void cmsUpgrades::L1TkStub< T >::setSimTrackId( unsigned int aSimTrackId ) {
    theSimTrackId = aSimTrackId;
  }



  /// ////////////// ///
  /// HELPER METHODS ///
  /// ////////////// ///

  /// Check SimTracks
  template< typename T >
  void cmsUpgrades::L1TkStub< T >::checkSimTrack()
  {
    /// L1TkCluster number should be at least 2,
    /// but only 2 for standard configuration
    bool tempGenuine = theClusters.at(0)->isGenuine();
    unsigned int tempSimTrack = theClusters.at(0)->getSimTrackId();
    int tempType = theClusters.at(0)->getType();

    /// Loop over L1TkClusters
    for ( unsigned int i = 1; i < theClusters.size(); i++ ) {
      if ( tempGenuine == false ) continue;
      if ( tempSimTrack !=  theClusters.at(i)->getSimTrackId() ) {
        tempGenuine = false;
        continue;
      }
      else {
        tempGenuine = theClusters.at(i)->isGenuine();
        tempSimTrack = theClusters.at(i)->getSimTrackId();
        /// We update only the bool flag and the unsigned int used
        /// to check within each loop... If tempGenuine at the end is
        /// true, tempSimTrack will automatically contain the Id of
        /// the associated SimTrack and tempType the corresponding
        /// PDG Code... maybe the tempSimTrack update is redundant
        /// and could be easliy removed...
      }
    } /// End of Loop over L1TkClusters

    this->setGenuine( tempGenuine );
    if ( tempGenuine ) {
      this->setType( tempType );
      this->setSimTrackId( tempSimTrack );
    }
  }


  /// Fit Stub as in Builder
  /// To be used for out-of-Builder Stubs
  template< typename T >
  void cmsUpgrades::L1TkStub< T >::fitStub( double aMagneticFieldStrength, const cmsUpgrades::StackedTrackerGeometry *aStackedTracker )
  {
    /// Get the magnetic field
    /// Usually it is done like the following three lines
    //iSetup.get<IdealMagneticFieldRecord>().get(magnet);
    //magnet_ = magnet.product();
    //mMagneticFieldStrength = magnet_->inTesla(GlobalPoint(0,0,0)).z();
    /// Calculate factor for rough Pt estimate
    /// B rounded to 4.0 or 3.8
    /// This is B * C / 2 * appropriate power of 10
    /// So it's B * 0.0015
    double mPtFactor = (floor(aMagneticFieldStrength*10.0 + 0.5))/10.0*0.0015;

    /// Get average position of Clusters composing the Stub
    GlobalPoint innerHitPosition = this->getCluster(0).getAveragePosition( aStackedTracker);
    GlobalPoint outerHitPosition = this->getCluster(1).getAveragePosition( aStackedTracker);
    /// Get useful quantities
    double outerPointRadius = outerHitPosition.perp();
    double innerPointRadius = innerHitPosition.perp();
    double outerPointPhi = outerHitPosition.phi();
    double innerPointPhi = innerHitPosition.phi();
    double deltaRadius = outerPointRadius - innerPointRadius;

    /// Here a switch on Barrel/Endcap is introduced
    if (theDetId.isBarrel()) {
      /// Calculate angular displacement from hit phi locations
      /// and renormalize it, if needed
      double deltaPhi = outerPointPhi - innerPointPhi;
      if (deltaPhi < 0) deltaPhi = -deltaPhi;
      if (deltaPhi > cmsUpgrades::KGMS_PI) deltaPhi = 2*cmsUpgrades::KGMS_PI - deltaPhi;

      /// Set the rough Pt
      this->setRoughPt( deltaRadius * mPtFactor / deltaPhi );
    }
    else if (theDetId.isEndcap()) {
      /// Test approximated formula for Endcap stubs
      /// Check always to be consistent with HitMatchingAlgorithm_window2012.h
      double roughPt = innerPointRadius * innerPointRadius * mPtFactor / fabs(this->getCluster(0).getAverageLocalPosition( aStackedTracker ).x()) ;
      roughPt += outerPointRadius * outerPointRadius * mPtFactor / fabs(this->getCluster(1).getAverageLocalPosition( aStackedTracker ).x()) ;
      roughPt = roughPt / 2.;

      /// Set the rough Pt
      this->setRoughPt( roughPt );
    }
  }



  /// /////////////////// ///
  /// INFORMATIVE METHODS ///
  /// /////////////////// ///

  template< typename T >
  std::string cmsUpgrades::L1TkStub< T >::print( unsigned int i ) const {
    std::string padding("");
    for ( unsigned int j=0; j!=i; ++j )padding+="\t";
    std::stringstream output;
    output<<padding<<"L1TkStub:\n";
    padding+='\t';
    output << padding << "StackedTrackerDetId: " << theDetId << '\n';
    unsigned int iClu = 0;
    for ( L1TkClusterPtrCollectionIterator i = theClusters.begin(); i!= theClusters.end(); ++i )
      output << padding << "cluster: " << iClu++ << ", member: " << (*i)->getStackMember() << ", cluster size: " << (*i)->getHits().size() << '\n';
    return output.str();
  }

  template< typename T >
  std::ostream& operator << (std::ostream& os, const cmsUpgrades::L1TkStub< T >& aL1TkStub) {
    return (os<<aL1TkStub.print() );
  }

} /// Close namespace

#endif



