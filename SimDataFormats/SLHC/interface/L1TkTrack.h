/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// Written by:                          ///
/// Andrew W. Rose                       ///
/// 2008                                 ///
///                                      ///
/// Changed by:                          ///
/// Nicola Pozzobon                      ///
/// UNIPD                                ///
/// 2010, June; 2011, June               ///
///                                      ///
/// Added features:                      ///
/// Higher threshold flag in 3_3_6 has   ///
/// been replaced by rough Pt            ///
/// calculation.                         ///
/// LocalStub and GlobalStub unified,    ///
/// Global information available through ///
/// a flag in configuration file         ///
/// ////////////////////////////////////////

#ifndef STACKED_TRACKER_L1TK_TRACK_FORMAT_H
#define STACKED_TRACKER_L1TK_TRACK_FORMAT_H

#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerGeometry.h"
#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerDetId.h"
#include "SLHCUpgradeSimulations/Utilities/interface/constants.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/Topology.h" 

//#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "SimDataFormats/SLHC/interface/L1TkStub.h"
#include "SimDataFormats/SLHC/interface/L1TkTracklet.h"

//#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/CircleFit.h"
//#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/LineFit.h"

namespace cmsUpgrades{

  /** ************************ **/
  /**                          **/
  /**   DECLARATION OF CLASS   **/
  /**                          **/
  /** ************************ **/

  template< typename T >
  class L1TkTrack {

    public:
      typedef L1TkStub< T >                                         L1TkStubType;
      typedef edm::Ptr< L1TkStubType >                              L1TkStubPtrType;
      typedef std::vector< L1TkStubType >                           L1TkStubCollection;
      typedef std::vector< L1TkStubPtrType >                        L1TkStubPtrCollection;
      typedef cmsUpgrades::L1TkTracklet< T >                        L1TkTrackletType;
      typedef edm::Ptr< L1TkTrackletType >                          L1TkTrackletPtrType;
      typedef std::set< std::pair<unsigned int , L1TkStubPtrType> > L1TkTrackletMap;
      typedef typename L1TkStubPtrCollection::const_iterator        L1TkStubPtrCollectionIterator;

    private:
      /// Data members
      L1TkStubPtrCollection   theBrickStubs;    /// The Stubs
      L1TkTrackletPtrType     theSeedTracklet;  /// The Seed
      unsigned int            theSimTrackId;
      bool                    theGenuine;
      int                     theType;
      /// From fit
      GlobalPoint             theVertex;
      GlobalVector            theMomentum;
      double                  theCharge; /// WARNING maybe should be changed to short int
      double                  theRadius; /// WARNING do we need it?
      GlobalPoint             theAxis;   /// WARNING do we need it?
      double                  theChi2RPhi;
      double                  theChi2ZPhi;
      bool                    theUseSeedVertex; /// Used also Vertex from Seed in Fit?


    public:
      /// Constructors
      L1TkTrack();
      L1TkTrack( L1TkStubPtrCollection aBrickStubs, L1TkTrackletPtrType aSeedTracklet );
      /// Destructor
      ~L1TkTrack();

      /// //////////////////////// ///
      /// METHODS FOR DATA MEMBERS ///
      /// Track components
      L1TkTrackletType       getSeedTracklet() const;
      L1TkStubCollection     getStubs() const;
      L1TkStubType           getStub( unsigned int layerIdentifier ) const;
      L1TkStubPtrType        getStubRef( unsigned int layerIdentifier) const;
      //void                   addStub( L1TkStubPtrType aL1TkStub );
      unsigned int           getSeedDoubleStack() const;
      /// Vertex from Fit
      void                   setVertex( GlobalPoint aVertex );
      GlobalPoint            getVertex() const;
      /// Momentum from Fit
      void                   setMomentum( GlobalVector aMomentum );
      GlobalVector           getMomentum() const;
      /// Charge from Fit
      void                   setCharge( double aCharge ); /// WARNING maybe better to change it into a short int!
      double                 getCharge() const;
      /// Trajectory radius from fit
      void                   setRadius( double aRadius ); /// WARNING additional information with sign?
      double                 getRadius() const;
      /// Trajectory axis from Fit
      void                   setAxis( double xAxis, double yAxis );
      GlobalPoint            getAxis() const;
      /// Chi2 from Fit
      void                   setChi2RPhi( double aChi2RPhi );
      double                 getChi2RPhi() const;
      void                   setChi2ZPhi( double aChi2ZPhi );
      double                 getChi2ZPhi() const;
      double                 getChi2Tot() const;
      /// Is Seed Vertex used in the Fit?
      void                   setUseSeedVertex( bool aUseSeedVertex );
      bool                   getUseSeedVertex() const;
      /// Fake or not
      bool                   isGenuine() const;
      void                   setGenuine( bool aGenuine );
      int                    getType() const;
      void                   setType( int aType );
      unsigned int           getSimTrackId() const;
      void                   setSimTrackId( unsigned int aSimTrackId );


      /// ////////////// ///
      /// HELPER METHODS ///
      /// Fake or not
      void checkSimTrack();

      /// Tricky Fit as suggested by Pierluigi, employing
      /// a tracklet-style approach for triplets within
      /// the chain of stubs composing the track
      void fitTrack( double aMagneticFieldStrength, bool useAlsoVtx, bool aDoHelixFit );



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
  cmsUpgrades::L1TkTrack< T >::L1TkTrack()
  {
    theBrickStubs.clear();
    theSeedTracklet = edm::Ptr< cmsUpgrades::L1TkTracklet< T > >();
    theVertex   = GlobalPoint(0.0,0.0,0.0);
    theMomentum = GlobalVector(0.0,0.0,0.0);
    theCharge   = 0;
    theRadius   = -99999.9;
    theAxis     = GlobalPoint(0.0,0.0,0.0);
    theChi2RPhi = -999.9;
    theChi2ZPhi = -999.9;
    theSimTrackId = 0;
    theGenuine = false;
    theType = -999999999;

  }

  /// Another Constructor
  template< typename T >
  cmsUpgrades::L1TkTrack< T >::L1TkTrack( std::vector< edm::Ptr< L1TkStub< T > > > aBrickStubs, edm::Ptr< L1TkTracklet< T > > aSeedTracklet ) 
  {
    /// Default
    theSimTrackId = 0;
    theGenuine = false;
    theType = -999999999;
    /// From input
    theBrickStubs = aBrickStubs;
    theSeedTracklet = aSeedTracklet;
    /// Default, to be used for Fit results
    theVertex   = GlobalPoint(0.0,0.0,0.0);
    theMomentum = GlobalVector(0.0,0.0,0.0);
    theCharge   = 0;
    theRadius   = -99999.9;
    theAxis     = GlobalPoint(0.0,0.0,0.0);
    theChi2RPhi = -999.9;
    theChi2ZPhi = -999.9;
  }

  /// Destructor
  template< typename T >
  cmsUpgrades::L1TkTrack< T >::~L1TkTrack()
  {
    /// Nothing is done
  }



  /// //////////////////////// ///
  /// METHODS FOR DATA MEMBERS ///
  /// //////////////////////// ///

  /// Get Seed Tracklet as an object, not a pointer
  template< typename T >
  cmsUpgrades::L1TkTracklet< T > cmsUpgrades::L1TkTrack< T >::getSeedTracklet() const
  {
    return *theSeedTracklet;
  }

  /// Get all the Stubs composing the Track
  template< typename T >
  std::vector< cmsUpgrades::L1TkStub< T > > cmsUpgrades::L1TkTrack< T >::getStubs() const
  {
    std::vector< cmsUpgrades::L1TkStub< T > > tempColl;
    tempColl.clear();
    for ( unsigned int i = 0; i < theBrickStubs.size(); i++ ) {
      tempColl.push_back( *(theBrickStubs.at(i)) );
    }
    return tempColl;
  }

  /// Get the Stub in a chosen trigger layer, if any, as an object
  template< typename T >
  cmsUpgrades::L1TkStub< T > cmsUpgrades::L1TkTrack< T >::getStub( unsigned int layerIdentifier ) const
  {
    /// Check each Stub and look for match with chosen layer
    for ( unsigned int i = 0; i < theBrickStubs.size(); i++ ) {
      if ( theBrickStubs.at(i)->getStack() == layerIdentifier ) return *(theBrickStubs.at(i));
    }
    /// Default return for no match
    return L1TkStub< T >();
  }

  /// Get the Stub in a chosen trigger layer, if any, as a pointer
  template< typename T >
  edm::Ptr< cmsUpgrades::L1TkStub< T > > cmsUpgrades::L1TkTrack< T >::getStubRef( unsigned int layerIdentifier ) const
  {
    /// Check each Stub and look for match with chosen layer
    for ( unsigned int i = 0; i < theBrickStubs.size(); i++ ) {
      if ( theBrickStubs.at(i)->getStack() == layerIdentifier ) return theBrickStubs.at(i);
    }
    /// Default return for no match
    return edm::Ptr< L1TkStub< T > >();
  }

  /// Seed Double Stack
  template< typename T>
  unsigned int cmsUpgrades::L1TkTrack< T >::getSeedDoubleStack() const
  {
    return theSeedTracklet->getDoubleStack();
  }

  /// Fit Vertex
  template< typename T >
  void cmsUpgrades::L1TkTrack< T >::setVertex( GlobalPoint aVertex )
  {
    theVertex = aVertex;
  }

  template< typename T >
  GlobalPoint cmsUpgrades::L1TkTrack< T >::getVertex() const
  {
    return theVertex;
  }

  /// Fit Momentum
  template< typename T >
  void cmsUpgrades::L1TkTrack< T >::setMomentum( GlobalVector aMomentum )
  {
    theMomentum = aMomentum;
  }

  template< typename T >
  GlobalVector cmsUpgrades::L1TkTrack< T >::getMomentum() const
  {
    return theMomentum;
  }

  /// Fit Charge
  template< typename T >
  void cmsUpgrades::L1TkTrack< T >::setCharge( double aCharge )
  {
    theCharge = aCharge;
  }

  template< typename T >
  double cmsUpgrades::L1TkTrack< T >::getCharge() const
  {
    return theCharge;
  }

  /// Fit Radius
  template< typename T >
  void cmsUpgrades::L1TkTrack< T >::setRadius( double aRadius )
  {
    theRadius = aRadius;
  }

  template< typename T >
  double cmsUpgrades::L1TkTrack< T >::getRadius() const
  {
    return theRadius;
  }

  /// Fit Axis
  template< typename T >
  void cmsUpgrades::L1TkTrack< T >::setAxis( double xAxis, double yAxis )
  {
    theAxis = GlobalPoint( xAxis, yAxis, 0.0 );
  }

  template< typename T >
  GlobalPoint cmsUpgrades::L1TkTrack< T >::getAxis() const
  {
    return theAxis;
  }

  /// Fit Chi2
  template< typename T >
  void cmsUpgrades::L1TkTrack< T >::setChi2RPhi( double aChi2RPhi )
  {
    theChi2RPhi = aChi2RPhi;
  }

  template< typename T >
  double cmsUpgrades::L1TkTrack< T >::getChi2RPhi() const
  {
    return theChi2RPhi;
  }

  template< typename T >
  void cmsUpgrades::L1TkTrack< T >::setChi2ZPhi( double aChi2ZPhi )
  {
    theChi2ZPhi = aChi2ZPhi;
  }

  template< typename T >
  double cmsUpgrades::L1TkTrack< T >::getChi2ZPhi() const
  {
    return theChi2ZPhi;
  }

  template< typename T >
  double cmsUpgrades::L1TkTrack< T >::getChi2Tot() const
  {
    return this->getChi2RPhi() + this->getChi2ZPhi();
  }

  /// Is Seed Vertex used in the Fit?
  template< typename T >
  void cmsUpgrades::L1TkTrack< T >::setUseSeedVertex( bool aUseSeedVertex )
  {
    theUseSeedVertex = aUseSeedVertex;
  }

  template< typename T >
  bool cmsUpgrades::L1TkTrack< T >::getUseSeedVertex() const
  {
    return theUseSeedVertex;
  }

  /// Fake or not fake
  template< typename T >
  bool cmsUpgrades::L1TkTrack< T >::isGenuine() const
  {
    return theGenuine;
  }

  template< typename T >
  void cmsUpgrades::L1TkTrack< T >::setGenuine( bool aGenuine ) {
    theGenuine = aGenuine;
  }

  template< typename T >
  int cmsUpgrades::L1TkTrack< T >::getType() const
  {
    return theType;
  }

  template< typename T >
  void cmsUpgrades::L1TkTrack< T >::setType( int aType ) {
    theType = aType;
  }

  template< typename T >
  unsigned int cmsUpgrades::L1TkTrack< T >::getSimTrackId() const
  {
    return theSimTrackId;
  }

  template< typename T >
  void cmsUpgrades::L1TkTrack< T >::setSimTrackId( unsigned int aSimTrackId ) {
    theSimTrackId = aSimTrackId;
  }


  /// ////////////// ///
  /// HELPER METHODS ///
  /// ////////////// ///

  /// Check SimTracks
  template< typename T >
  void cmsUpgrades::L1TkTrack< T >::checkSimTrack()
  {
    /// L1TkCluster number should be at least 2,
    /// but only 2 for standard configuration
    bool tempGenuine = theBrickStubs.at(0)->isGenuine();
    unsigned int tempSimTrack = theBrickStubs.at(0)->getSimTrackId();
    int tempType = theBrickStubs.at(0)->getType();

    /// Loop over L1TkStubs
    for ( unsigned int i = 1; i < theBrickStubs.size(); i++ ) {

      if ( tempGenuine == false ) continue;
      if ( tempSimTrack !=  theBrickStubs.at(i)->getSimTrackId() ) {
        tempGenuine = false;
        continue;
      }
      else {
        tempGenuine = theBrickStubs.at(i)->isGenuine();
        tempSimTrack = theBrickStubs.at(i)->getSimTrackId();
        /// We update only the bool flag and the unsigned int used
        /// to check within each loop... If tempGenuine at the end is
        /// true, tempSimTrack will automatically contain the Id of
        /// the associated SimTrack and tempType the corresponding
        /// PDG Code... maybe the tempSimTrack update is redundant
        /// and could be easliy removed...
      }
    } /// End of Loop over L1TkStubs

    this->setGenuine( tempGenuine );
    if ( tempGenuine ) {
      this->setType( tempType );
      this->setSimTrackId( tempSimTrack );
    }
  }


  /// Fit
  template< typename T >
  void cmsUpgrades::L1TkTrack< T >::fitTrack( double aMagneticFieldStrength, bool useAlsoVtx, bool aDoHelixFit )
  {
    /// Step 00
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

    /// Step 0
    /// Get Stubs chain and Vertex      
    std::vector< cmsUpgrades::L1TkStub< T > > brickStubs = this->getStubs();
    cmsUpgrades::L1TkTracklet< T >            seedTracklet = this->getSeedTracklet();
    /// This automatically sets 00 or beamspot according to L1TkTracklet type
    GlobalPoint seedVertexXY = GlobalPoint( seedTracklet.getVertex().x(), seedTracklet.getVertex().y(), 0.0 );

    /// If the seed vertex is requested for the fit, add it to stubs
    if ( useAlsoVtx ) {
      /// Prepare dummy stub with vertex position
      std::vector< cmsUpgrades::L1TkStub< T > > auxStubs;
      auxStubs.clear();
      cmsUpgrades::L1TkStub< T > dummyStub = cmsUpgrades::L1TkStub< T >( 0 );
      dummyStub.setPosition( seedTracklet.getVertex() );
      dummyStub.setDirection( GlobalVector(0,0,0) );
      auxStubs.push_back( dummyStub );
      /// Put together also other stubs
      for ( unsigned int j = 0; j < brickStubs.size(); j++ ) auxStubs.push_back( brickStubs.at(j) );
      /// Overwrite
      brickStubs = auxStubs;
    }

    /// Step 1
    /// Find charge using only stubs, regardless of useAlsoVtx option!
    unsigned int iMin = 0;
    if ( useAlsoVtx ) iMin = 1;
    /// Check L1TkTracklet for further information
    double outerPointPhi = brickStubs.at( brickStubs.size()-1 ).getPosition().phi();
    double innerPointPhi = brickStubs.at( iMin ).getPosition().phi();
    double deltaPhi = outerPointPhi - innerPointPhi;
    if ( fabs(deltaPhi) >= cmsUpgrades::KGMS_PI) {
      if ( deltaPhi>0 ) deltaPhi = deltaPhi - 2*cmsUpgrades::KGMS_PI;
      else deltaPhi = 2*cmsUpgrades::KGMS_PI - fabs(deltaPhi);
    }
    double deltaPhiC = deltaPhi; /// This is for charge
    deltaPhi = fabs(deltaPhi);
    double fCharge = -deltaPhiC / deltaPhi;
    this->setCharge( fCharge );

    /// Step 2
    /// Average for Momentum and Axis
    std::vector< double > outputFitPt; outputFitPt.clear();
    std::vector< double > outputFitPz; outputFitPz.clear();
    std::vector< double > outputFitX;  outputFitX.clear();
    std::vector< double > outputFitY;  outputFitY.clear();
    /// Now loop over Triplets
    unsigned int totalTriplets = 0;
    for ( unsigned int a1 = 0; a1 < brickStubs.size(); a1++ ) {
      for ( unsigned int a2 = a1+1; a2 < brickStubs.size(); a2++ ) {
        for ( unsigned int a3 = a2+1; a3 < brickStubs.size(); a3++ ) {
          totalTriplets++;
          /// Read Stubs in a "L1TkTracklet-wise" way
          GlobalPoint vtxPos = brickStubs.at(a1).getPosition();
          GlobalPoint innPos = brickStubs.at(a2).getPosition();
          GlobalPoint outPos = brickStubs.at(a3).getPosition();
          /// Correct for position of a1
          innPos = GlobalPoint( innPos.x()-vtxPos.x(), innPos.y()-vtxPos.y(), innPos.z() );
          outPos = GlobalPoint( outPos.x()-vtxPos.x(), outPos.y()-vtxPos.y(), outPos.z() );
          double outRad = outPos.perp();
          double innRad = innPos.perp();
          deltaPhi = outPos.phi() - innPos.phi(); /// NOTE overwrite already declared deltaPhi
          if ( fabs(deltaPhi) >= cmsUpgrades::KGMS_PI ) {
            if ( deltaPhi>0 ) deltaPhi = deltaPhi - 2*cmsUpgrades::KGMS_PI;
            else deltaPhi = 2*cmsUpgrades::KGMS_PI - fabs(deltaPhi);
          }
          deltaPhi = fabs(deltaPhi);
          double x2 = outRad * outRad + innRad * innRad - 2 * innRad * outRad * cos(deltaPhi);
          double twoRadius = sqrt(x2) / sin(fabs(deltaPhi));
          double roughPt = mPtFactor * twoRadius;
          double roughPz;
          /// Switch fit type
          if ( !aDoHelixFit ) roughPz = roughPt * (outPos.z()-innPos.z()) / (outRad-innRad);
          else {
            double phioi = acos(1 - 2*x2/(twoRadius*twoRadius));
            if ( fabs(phioi) >= cmsUpgrades::KGMS_PI ) {
              if ( phioi>0 ) phioi = phioi - 2*cmsUpgrades::KGMS_PI;
              else phioi = 2*cmsUpgrades::KGMS_PI - fabs(phioi);
            }
            if ( phioi == 0 ) return;
            roughPz = 2 * mPtFactor * (outPos.z()-innPos.z()) / fabs(phioi);
          }
          /// Store Momenta for average
          outputFitPt.push_back( roughPt );
          outputFitPz.push_back( roughPz );
          /// Find angle from a1 pointing to Axis
          double vertexangle = acos( outRad/twoRadius );
          vertexangle = outPos.phi() - fCharge * vertexangle;
          /// Helix axis
          outputFitX.push_back( 0.5 * twoRadius * cos(vertexangle) + vtxPos.x() );
          outputFitY.push_back( 0.5 * twoRadius * sin(vertexangle) + vtxPos.y() );
        } /// End of loop over third element
      } /// End of loop over second element
    } /// End of loop over first element
    /// Compute averages and store them
    double tempOutputX = 0;
    double tempOutputY = 0;
    double tempOutputPt = 0;
    double tempOutputPz = 0;
    for ( unsigned int q = 0; q < totalTriplets; q++ ) {
      tempOutputX += outputFitX.at(q);
      tempOutputY += outputFitY.at(q);
      tempOutputPt += outputFitPt.at(q);
      tempOutputPz += outputFitPz.at(q);        
    }

    /// Step 3
    /// Get Helix Axis and correct wrt Seed VTX
    GlobalPoint fAxis = GlobalPoint( tempOutputX/totalTriplets, tempOutputY/totalTriplets, 0.0 );
    GlobalPoint fAxisCorr = GlobalPoint( fAxis.x() - seedVertexXY.x(), fAxis.y() - seedVertexXY.y(), 0.0 );
    this->setAxis( tempOutputX/totalTriplets, tempOutputY/totalTriplets );

    /// Step 4
    /// Momentum, starting from azimuth at vertex
    double fPhiV = atan2( fCharge*fAxisCorr.x(), -fCharge*fAxisCorr.y() );
    double fPt = tempOutputPt/totalTriplets;
    double fPz = tempOutputPz/totalTriplets;
    double fRadius  = 0.5*fPt/mPtFactor;
    GlobalVector fMomentum = GlobalVector( cos(fPhiV)*fPt, sin(fPhiV)*fPt, fPz );
    this->setMomentum( fMomentum );

    /// Step 5
    /// Average for Vertex (Closest Approach)
    double rMinAppr = fAxisCorr.perp() - fRadius;
    double xMinAppr = rMinAppr*cos( fAxisCorr.phi() ) + seedVertexXY.x();
    double yMinAppr = rMinAppr*sin( fAxisCorr.phi() ) + seedVertexXY.y();
    GlobalPoint tempVertex = GlobalPoint( xMinAppr, yMinAppr, 0.0 );
    double propFactorHel = 0;
    double offsetHel = 0;
    /// Average for Vtx z
    std::vector< double > outputFitZ; outputFitZ.clear();
    /// Now loop over Doublets
    /// Cannot put into the same loop as before because
    /// here we need radius and therefore Pt to have
    /// the radius, and the radius is needed to find the
    /// closest approach distance, is it clear?
    unsigned int totalDoublets = 0;
    for ( unsigned int a1 = iMin; a1 < brickStubs.size(); a1++) { /// iMin already set according to useAlsoVtx or not
      for ( unsigned int a2 = a1+1; a2 < brickStubs.size(); a2++) {
        totalDoublets++;
        /// Read Stubs in a "L1TkTracklet-wise" way
        GlobalPoint innPos = brickStubs.at(a1).getPosition();
        GlobalPoint outPos = brickStubs.at(a2).getPosition();
        /// Calculate z = z0 + c*phiStar
        GlobalPoint innPosStar = GlobalPoint( innPos.x() - fAxis.x(), innPos.y() - fAxis.y(), innPos.z() - fAxis.z() );
        GlobalPoint outPosStar = GlobalPoint( outPos.x() - fAxis.x(), outPos.y() - fAxis.y(), outPos.z() - fAxis.z() );
        double deltaPhiStar = outPosStar.phi() - innPosStar.phi();
        if ( fabs(deltaPhiStar) >= cmsUpgrades::KGMS_PI ) {
          if ( outPosStar.phi() < 0 ) deltaPhiStar += cmsUpgrades::KGMS_PI;
          else deltaPhiStar -= cmsUpgrades::KGMS_PI;
          if ( innPosStar.phi() < 0 ) deltaPhiStar -= cmsUpgrades::KGMS_PI;
          else deltaPhiStar += cmsUpgrades::KGMS_PI;
        }
        if ( deltaPhiStar == 0 ) std::cerr<<"BIG PROBLEM IN DELTAPHI DENOMINATOR"<<std::endl;
        else {
          propFactorHel += ( outPosStar.z() - innPosStar.z() )/deltaPhiStar;
          offsetHel += innPosStar.z() - innPosStar.phi()*( outPosStar.z() - innPosStar.z() )/deltaPhiStar;
        } /// End of calculate z = z0 + c*phiStar
        innPos = GlobalPoint( innPos.x() - tempVertex.x(), innPos.y() - tempVertex.y(), innPos.z() );
        outPos = GlobalPoint( outPos.x() - tempVertex.x(), outPos.y() - tempVertex.y(), outPos.z() );
        double outRad = outPos.perp();
        double innRad = innPos.perp();
        deltaPhi = outPos.phi() - innPos.phi(); /// NOTE overwrite already declared deltaPhi
        if ( fabs(deltaPhi) >= cmsUpgrades::KGMS_PI ) {
          if ( deltaPhi>0 ) deltaPhi = deltaPhi - 2*cmsUpgrades::KGMS_PI;
          else deltaPhi = 2*cmsUpgrades::KGMS_PI - fabs(deltaPhi);
        }
        deltaPhi = fabs(deltaPhi);
        double x2 = outRad * outRad + innRad * innRad - 2 * innRad * outRad * cos(deltaPhi);
        double twoRadius = sqrt(x2) / sin(fabs(deltaPhi));
        double zProj;
        /// Switch fit type
        if ( !aDoHelixFit ) zProj = outPos.z() - ( outRad * (outPos.z() - innPos.z()) / (outRad - innRad) );
        else {
          double phioi = acos(1 - 2*x2/(twoRadius*twoRadius));
          double phiiv = acos(1 - 2*innRad*innRad/(twoRadius*twoRadius));
          if ( fabs(phioi) >= cmsUpgrades::KGMS_PI ) {
            if ( phioi>0 ) phioi = phioi - 2*cmsUpgrades::KGMS_PI;
            else phioi = 2*cmsUpgrades::KGMS_PI - fabs(phioi);
          }
          if ( fabs(phiiv) >= cmsUpgrades::KGMS_PI ) {
            if ( phiiv>0 ) phiiv = phiiv - 2*cmsUpgrades::KGMS_PI;
            else phiiv = 2*cmsUpgrades::KGMS_PI - fabs(phiiv);
          }
          if ( phioi == 0 ) return;
          /// Vertex
          zProj = innPos.z() - (outPos.z()-innPos.z())*phiiv/phioi;
        }
        outputFitZ.push_back( zProj );
      } /// End of loop over second element
    } /// End of loop over first element
    /// Compute averages and store them
    double tempOutputZ = 0;
    for ( unsigned int q = 0; q < totalDoublets; q++ ) tempOutputZ += outputFitZ.at(q);
    double zMinAppr = tempOutputZ/totalDoublets;

    /// Step 6
    /// Vertex
    GlobalPoint fVertex = GlobalPoint( xMinAppr, yMinAppr, zMinAppr );
    this->setVertex( fVertex );


    /// Step 7
    /// Calculate Chi2
    propFactorHel = propFactorHel/totalDoublets;
    offsetHel = offsetHel/totalDoublets;
    double fChi2RPhi = 0;
    double fChi2ZPhi = 0;
    double tempStep;
    /// Calculate for the Seed VTX if needed
    if (useAlsoVtx) {
      GlobalPoint posPoint = seedTracklet.getVertex();
      GlobalPoint posPointCorr = GlobalPoint( posPoint.x() - fAxis.x(), posPoint.y() - fAxis.y(), posPoint.z() );
      /// Add X: x_meas - x_fit(phi*_meas)
      tempStep = posPoint.x() - fAxis.x() - fRadius * cos( posPointCorr.phi() );
      fChi2RPhi += tempStep*tempStep;
      /// Add Y: y_meas - y_fit(phi*_meas)
      tempStep = posPoint.y() - fAxis.y() - fRadius * sin( posPointCorr.phi() );
      fChi2RPhi += tempStep*tempStep;
      /// b = propFactorH
      /// a = offsetH
      /// z = b*phi - a
      /// Add Z: z_meas - z_fit(phi*_meas)
      tempStep = posPoint.z() - offsetHel - propFactorHel * posPointCorr.phi();
      fChi2ZPhi += tempStep*tempStep;
    }
    /// Calculate for all other Stubs
    for ( unsigned int a = iMin; a < brickStubs.size(); a++ ) {
      GlobalPoint posPoint = brickStubs.at(a).getPosition();
      GlobalPoint posPointCorr = GlobalPoint( posPoint.x() - fAxis.x(), posPoint.y() - fAxis.y(), posPoint.z() );

      tempStep = posPoint.x() - fAxis.x() - fRadius * cos( posPointCorr.phi() );
      fChi2RPhi += tempStep*tempStep;
      tempStep = posPoint.y() - fAxis.y() - fRadius * sin( posPointCorr.phi() );
      fChi2RPhi += tempStep*tempStep;

      tempStep = posPoint.z() - offsetHel - propFactorHel * posPointCorr.phi();
      fChi2ZPhi += tempStep*tempStep;
    }
    this->setChi2RPhi( fChi2RPhi );
    this->setChi2ZPhi( fChi2ZPhi );
  }



  /// /////////////////// ///
  /// INFORMATIVE METHODS ///
  /// /////////////////// ///

  template< typename T >
  std::string cmsUpgrades::L1TkTrack< T >::print( unsigned int i ) const {
    std::string padding("");
    for ( unsigned int j=0; j!=i; ++j )padding+="\t";
    std::stringstream output;
    output<<padding<<"L1TkTrack:\n";
    padding+='\t';
    output << padding << "SeedDoubleStack: " << this->getSeedDoubleStack() << '\n';
    output << padding << "Length of Chain: " << theBrickStubs.size() << '\n';
    unsigned int iStub = 0;
    for ( L1TkStubPtrCollectionIterator i = theBrickStubs.begin(); i!= theBrickStubs.end(); ++i )
      output << padding << "stub: " << iStub++ << ", stack: " << (*i)->getStack() << ", rough Pt: " << (*i)->getRoughPt() << '\n';
    return output.str();
  }

  template< typename T >
  std::ostream& operator << (std::ostream& os, const cmsUpgrades::L1TkTrack< T >& aL1TkTrack) {
    return (os<<aL1TkTrack.print() );
  }

} /// Close namespace

#endif


