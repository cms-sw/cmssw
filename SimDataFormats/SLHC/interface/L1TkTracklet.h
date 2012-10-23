/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// Written by:                          ///
/// Andrew W. Rose                       ///
/// 2008                                 ///
///                                      ///
/// Changed by:                          ///
/// Nicola Pozzobon                      ///
/// UNIPD                                ///
/// 2010, June, August, October          ///
///                                      ///
/// Added features:                      ///
/// FNAL design flag                     ///
/// Check fakeness and SimTrack Id       ///
/// Beamspot option for the vertex       ///
/// Added momentum fit with TRUE helix   ///
///                                      ///
/// Changed by:                          ///
/// Nicola Pozzobon                      ///
/// UNIPD                                ///
/// 2011, July                           ///
///                                      ///
/// Avoid duplicate operations: fit      ///
/// always in the builder, for default   ///
/// option (approximated fit, vtx = 00)  ///
/// plus additional optional strategies  ///
/// (helix fit, vtx = beamspot)          ///
/// FNAL design from a flag in cfi file  ///
/// different output collections of the  ///
/// builder for different fit options    ///
/// ////////////////////////////////////////


#ifndef STACKED_TRACKER_L1TK_TRACKLET_H
#define STACKED_TRACKER_L1TK_TRACKLET_H

#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerDetId.h"
#include "SLHCUpgradeSimulations/Utilities/interface/constants.h"

#include "SimDataFormats/SLHC/interface/L1TkStub.h"
#include "DataFormats/Common/interface/Ptr.h"

#include "FWCore/Framework/interface/EventSetup.h"
/// WARNING do we really need them?
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"

namespace cmsUpgrades{

  /** ************************ **/
  /**                          **/
  /**   DECLARATION OF CLASS   **/
  /**                          **/
  /** ************************ **/

  template< typename T >
  class L1TkTracklet {

    public:
      typedef L1TkStub< T >                                            L1TkStubType;
      typedef edm::Ptr< L1TkStubType >                                 L1TkStubPtrType;
      typedef std::set< std::pair< unsigned int , L1TkStubPtrType > >  L1TkTrackletMap;
      typedef typename L1TkTrackletMap::const_iterator                 L1TkTrackletMapIterator;    

    private:
      /// Data members
      L1TkTrackletMap theStubs;
      GlobalPoint     theVertex;
      GlobalVector    theMomentum;
      double          theCharge; /// WARNING maybe should be changed to short int
      unsigned int    theSimTrackId;
      bool            theGenuine;
      int             theType;

    public:
      /// Constructors
      L1TkTracklet();
      /// Destructor
      ~L1TkTracklet(); /// WARNING once it was virtual. WHY????

      /// //////////////////////// ///
      /// METHODS FOR DATA MEMBERS ///
      /// Detector element
      unsigned int           getDoubleStack() const;
      bool                   isLongBarrelHermetic() const;
      /// Stubs composing the Tracklet
      void                   addStub( unsigned int aStubIdentifier, const L1TkStubPtrType &aStub );
      L1TkStubPtrType        getStubRef( unsigned int aStubIdentifier ) const;
      const L1TkTrackletMap& getStubRefs() const;
      /// Vertex from Fit
      void                   setVertex( GlobalPoint aVertex );
      GlobalPoint            getVertex() const;
      /// Momentum from Fit
      void                   setMomentum( GlobalVector aMomentum );
      GlobalVector           getMomentum() const;
      /// Charge from Fit
      void                   setCharge( double aCharge ); /// WARNING maybe better to change it into a short int!
      double                 getCharge() const;
      /// Fake or not
      bool                   isGenuine() const;
      void                   setGenuine( bool aGenuine );
      int                    getType() const;
      void                   setType( int aType );
      unsigned int           getSimTrackId() const;
      void                   setSimTrackId( unsigned int aSimTrackId );

      /// Hermetic according to FNAL design
      /// Methods are removed: just access
      /// to L1TkStub::getLadderPhi() and
      /// L1TkStub::getStack() and you can
      /// decide if it is FNAL-design compliant
      /// or not! Look at the Builder for the flag
      /// used for production which selects
      /// or rejects candidates according to
      /// FNAL design and ladder/stack

      /// Beamspot flag
      /// Methods are removed: beamspot is
      /// used only for the fit, not to 
      /// accept/reject a L1TkTracklet since
      /// previous studies showed the result
      /// is not influenced by the choice.
      /// Flag is moved within product name
      /// of the chosen collection

      /// ////////////// ///
      /// HELPER METHODS ///
      /// Fake or not
      void checkSimTrack();

      /// Fit Stub as in Builder
      /// To be used for out-of-Builder Stubs
      void fitTracklet( double aMagneticFieldStrength, GlobalPoint aVertex, bool aDoHelixFit );

      /// Get Momentum at the position of Stubs
      GlobalVector getMomentum( unsigned int aStubIdentifier ) const;

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
  cmsUpgrades::L1TkTracklet< T >::L1TkTracklet()
  {
    /// Set default data members
    theStubs.clear();
    theVertex   = GlobalPoint(0.0,0.0,0.0);
    theMomentum = GlobalVector(0.0,0.0,0.0);
    theCharge   = 0;
    theSimTrackId = 0;
    theGenuine = false;
    theType = -999999999;
  }

  /// Another Constructor
  /// WARNING Maybe it is useful to
  /// add another constructor of the kind
  /// L1TkTracklet( some of the data members )

  /// Destructor
  template< typename T >
  cmsUpgrades::L1TkTracklet< T >::~L1TkTracklet()
  {
    /// Nothing is done
  }



  /// //////////////////////// ///
  /// METHODS FOR DATA MEMBERS ///
  /// //////////////////////// ///

  /// Detector element
  template< typename T >
  unsigned int cmsUpgrades::L1TkTracklet< T >::getDoubleStack() const
  {
    unsigned int innerStack = this->getStubRef( 0 )->getStack();
    unsigned int outerStack = this->getStubRef( 1 )->getStack();
    if ( innerStack/2 != (outerStack-1)/2 ) return 9999;
    else return innerStack/2;
  }

  /// Is it Hermetic according to Long-Barrel
  /// Tracker layout? same ladder and no DoubleStack crossing?
  template< typename T >
  bool cmsUpgrades::L1TkTracklet< T >::isLongBarrelHermetic() const
  {
    unsigned int innerStack = this->getStubRef( 0 )->getStack();
    unsigned int outerStack = this->getStubRef( 1 )->getStack();
    unsigned int innerPhi = this->getStubRef( 0 )->getLadderPhi();
    unsigned int outerPhi = this->getStubRef( 1 )->getLadderPhi();

    /// Inner Stack Idx even, outer one odd
    /// Same iPhi in both trigger modules
    return ( innerStack%2 == 0 &&
             outerStack%2 == 1 &&
             innerPhi == outerPhi );
  }

  /// Add a Stub
  template< typename T >
  void cmsUpgrades::L1TkTracklet< T >::addStub( unsigned int aStubIdentifier,
                                                const edm::Ptr< cmsUpgrades::L1TkStub< T > > &aStub )
  {
      theStubs.insert( std::make_pair( aStubIdentifier , aStub ) ); 
  }

  /// Get a Stub as a Pointer
  template< typename T >
  edm::Ptr< cmsUpgrades::L1TkStub< T > > cmsUpgrades::L1TkTracklet< T >::getStubRef( unsigned int aStubIdentifier ) const
  {
    for ( L1TkTrackletMapIterator i = theStubs.begin(); i != theStubs.end(); ++i ) {
      if ( i->first == aStubIdentifier ) return i->second;
    }
    return edm::Ptr< L1TkStub< T > >();
  }

  /// Get all Stubs as Pointers
  template< typename T >
  const std::set< std::pair< unsigned int , edm::Ptr< cmsUpgrades::L1TkStub< T > > > >& cmsUpgrades::L1TkTracklet< T >::getStubRefs() const
  {
    return theStubs;
  }

  /// Fit Vertex
  template< typename T >
  void cmsUpgrades::L1TkTracklet< T >::setVertex( GlobalPoint aVertex )
  {
    theVertex = aVertex;
  }

  template< typename T >
  GlobalPoint cmsUpgrades::L1TkTracklet< T >::getVertex() const
  {
    return theVertex;
  }

  /// Fit Momentum
  template< typename T >
  void cmsUpgrades::L1TkTracklet< T >::setMomentum( GlobalVector aMomentum )
  {
    theMomentum = aMomentum;
  }

  template< typename T >
  GlobalVector cmsUpgrades::L1TkTracklet< T >::getMomentum() const
  {
    return theMomentum;
  }

  /// Fit Charge
  template< typename T >
  void cmsUpgrades::L1TkTracklet< T >::setCharge( double aCharge )
  {
    theCharge = aCharge;
  }

  template< typename T >
  double cmsUpgrades::L1TkTracklet< T >::getCharge() const
  {
    return theCharge;
  }

  /// Fake or not fake
  template< typename T >
  bool cmsUpgrades::L1TkTracklet< T >::isGenuine() const
  {
    return theGenuine;
  }

  template< typename T >
  void cmsUpgrades::L1TkTracklet< T >::setGenuine( bool aGenuine ) {
    theGenuine = aGenuine;
  }

  template< typename T >
  int cmsUpgrades::L1TkTracklet< T >::getType() const
  {
    return theType;
  }

  template< typename T >
  void cmsUpgrades::L1TkTracklet< T >::setType( int aType ) {
    theType = aType;
  }

  template< typename T >
  unsigned int cmsUpgrades::L1TkTracklet< T >::getSimTrackId() const
  {
    return theSimTrackId;
  }

  template< typename T >
  void cmsUpgrades::L1TkTracklet< T >::setSimTrackId( unsigned int aSimTrackId ) {
    theSimTrackId = aSimTrackId;
  }



  /// ////////////// ///
  /// HELPER METHODS ///
  /// ////////////// ///

  /// Check SimTracks
  template< typename T >
  void cmsUpgrades::L1TkTracklet< T >::checkSimTrack()
  {
    /// L1TkCluster number should be at least 2,
    /// but only 2 for standard configuration
    bool tempGenuine = this->getStubRef(0)->isGenuine();
    unsigned int tempSimTrack = this->getStubRef(0)->getSimTrackId();
    int tempType = this->getStubRef(0)->getType();

    /// Loop over L1TkStubs
    for ( L1TkTrackletMapIterator i = theStubs.begin(); i != theStubs.end(); ++i ) {

      /// Loop starts from the second one
      if ( i->first == 0 ) continue;

      if ( tempGenuine == false ) continue;
      if ( tempSimTrack !=  i->second->getSimTrackId() ) {
        tempGenuine = false;
        continue;
      }
      else {
        tempGenuine = i->second->isGenuine();
        tempSimTrack = i->second->getSimTrackId();
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

  /// Get Momentum at the position of Stubs
  template< typename T >
  GlobalVector L1TkTracklet< T >::getMomentum( unsigned int aStubIdentifier ) const
  {
    /// Get position of the Stubs and Vertex
    GlobalPoint innerStubPosition = this->getStubRef(0)->getPosition();
    GlobalPoint outerStubPosition = this->getStubRef(1)->getPosition();
    GlobalPoint vertexPosition    = this->getVertex(); /// Already includes Vtx00/VtxBS and StdFit/HelFit options
    
    /// Correct for Vertex position
    innerStubPosition = GlobalPoint( innerStubPosition.x() - vertexPosition.x(),
                                     innerStubPosition.y() - vertexPosition.y(),
                                     (double)innerStubPosition.z() );
    outerStubPosition = GlobalPoint( outerStubPosition.x() - vertexPosition.x(),
                                     outerStubPosition.y() - vertexPosition.y(),
                                     (double)outerStubPosition.z() );

    /// Get distances etc...
    double outerPointRadius = outerStubPosition.perp();
    double innerPointRadius = innerStubPosition.perp();
    double chosenStubRadius;
    if ( aStubIdentifier==0 ) chosenStubRadius = innerPointRadius;
    else if ( aStubIdentifier==1 ) chosenStubRadius = outerPointRadius;
    else return GlobalVector(0.0,0.0,0.0);

    double outerPointPhi = outerStubPosition.phi();
    double innerPointPhi = innerStubPosition.phi();
    double deltaPhi = outerPointPhi - innerPointPhi;
    if (deltaPhi < 0) deltaPhi = -deltaPhi;
    if (deltaPhi > cmsUpgrades::KGMS_PI) deltaPhi = 2*cmsUpgrades::KGMS_PI - deltaPhi;

    /// Circle radius
    double x2 = outerPointRadius * outerPointRadius +
                innerPointRadius * innerPointRadius -
                2 * innerPointRadius * outerPointRadius * cos(deltaPhi); /// Cosine theorem
    double twoRadius = sqrt(x2) / sin(fabs(deltaPhi));

    /// Get Momentum at Vertex
    GlobalVector vtxMomentum = this->getMomentum(); /// Already includes Vtx00/VtxBS and StdFit/HelFit options
    double vertexphi = vtxMomentum.phi();
    vertexphi = vertexphi - this->getCharge() * 2 * asin( chosenStubRadius/twoRadius );

    return GlobalVector( vtxMomentum.perp()*cos(vertexphi),
                         vtxMomentum.perp()*sin(vertexphi),
                         vtxMomentum.z() );

  }

  /// Fit Tracklet as in Builder
  /// To be used for out-of-Builder Tracklets
  template< typename T >
  void cmsUpgrades::L1TkTracklet< T >::fitTracklet( double aMagneticFieldStrength, GlobalPoint aVertex, bool aDoHelixFit )
  {
    /// Calculate factor for rough Pt estimate
    /// B rounded to 4.0 or 3.8
    /// This is B * C / 2 * appropriate power of 10
    /// So it's B * 0.0015
    double mPtFactor = (floor(aMagneticFieldStrength*10.0 + 0.5))/10.0*0.0015;   
    
    /// Get the Stubs    
    edm::Ptr< cmsUpgrades::L1TkStub< T > > innerStub, outerStub;
    innerStub = this->getStubRef(0);
    outerStub = this->getStubRef(1);
    /// Get average position of Stubs composing the Tracklet
    GlobalPoint innerStubPosition = (*innerStub).getPosition();
    GlobalPoint outerStubPosition = (*outerStub).getPosition();

    /// Correct, if needed, for supplied vertex
    innerStubPosition = GlobalPoint( innerStubPosition.x() - aVertex.x(),
                                     innerStubPosition.y() - aVertex.y(),
                                     (double)innerStubPosition.z() );
    outerStubPosition = GlobalPoint( outerStubPosition.x() - aVertex.x(),
                                     outerStubPosition.y() - aVertex.y(),
                                     (double)outerStubPosition.z() );

    /// Get useful quantities
    double outerPointRadius = outerStubPosition.perp();
    double innerPointRadius = innerStubPosition.perp();
    double outerPointPhi = outerStubPosition.phi();
    double innerPointPhi = innerStubPosition.phi();
    double deltaRadius = outerPointRadius - innerPointRadius;
    /// Calculate angular displacement from hit phi locations
    /// and renormalize it, if needed
    double deltaPhi = outerPointPhi - innerPointPhi;
    //if (deltaPhi < 0) deltaPhi = -deltaPhi;
    //if (deltaPhi > cmsUpgrades::KGMS_PI) deltaPhi = 2*cmsUpgrades::KGMS_PI - deltaPhi;
    if ( fabs(deltaPhi) >= cmsUpgrades::KGMS_PI ) {
      if ( deltaPhi>0 ) deltaPhi = deltaPhi - 2*cmsUpgrades::KGMS_PI;
      else deltaPhi = 2*cmsUpgrades::KGMS_PI - fabs(deltaPhi);
    }
    double deltaPhiC = deltaPhi; /// This is for charge
    deltaPhi = fabs(deltaPhi);
    /// This is time for fit!
    double charge = -deltaPhiC / deltaPhi;
    this->setCharge( charge );    
    
    /// Do things intependent on Std/Hel
    /// Momentum
    double x2 = outerPointRadius * outerPointRadius +
                innerPointRadius * innerPointRadius -
                2 * innerPointRadius * outerPointRadius * cos(deltaPhi); /// Cosine theorem
    double twoRadius = sqrt(x2) / sin(fabs(deltaPhi));
    double roughPt = mPtFactor * twoRadius;
    double vertexphi = acos(outerPointRadius/twoRadius);
    vertexphi = outerPointPhi - charge*vertexphi;
    vertexphi = vertexphi + charge*0.5*cmsUpgrades::KGMS_PI;
    if ( vertexphi > cmsUpgrades::KGMS_PI ) vertexphi -= 2*cmsUpgrades::KGMS_PI;
    else if ( vertexphi <= -cmsUpgrades::KGMS_PI ) vertexphi += 2*cmsUpgrades::KGMS_PI;

    /// For longitudinal projection
    double zProj, roughPz;
      
    /// Switch fit type
    if ( !aDoHelixFit ) {
      /// Std fit
      /// Vertex
      zProj = outerStubPosition.z() - ( outerPointRadius * (outerStubPosition.z()-innerStubPosition.z()) / deltaRadius );
      /// Momentum
      roughPz = roughPt*(outerStubPosition.z()-innerStubPosition.z())/deltaRadius;
    }
    else {
      /// Hel fit
      double phioi = acos(1 - 2*x2/(twoRadius*twoRadius));
      double phiiv = acos(1 - innerPointRadius*innerPointRadius*2/(twoRadius*twoRadius));
      /// Find advancement!
      if ( fabs(phioi) >= cmsUpgrades::KGMS_PI ) {
        if ( phioi>0 ) phioi = phioi - 2*cmsUpgrades::KGMS_PI;
        else phioi = 2*cmsUpgrades::KGMS_PI - fabs(phioi);
      }
      if ( fabs(phiiv) >= cmsUpgrades::KGMS_PI ) {
        if ( phiiv>0 ) phiiv = phiiv - 2*cmsUpgrades::KGMS_PI;
        else phiiv = 2*cmsUpgrades::KGMS_PI - fabs(phiiv);
      }
      if (phioi==0) return;
      /// Vertex
      zProj = (*innerStub).getPosition().z() - ((*outerStub).getPosition().z()-(*innerStub).getPosition().z())*phiiv/phioi;
                                                                          /// This is fine enough as using innerBeamSpot or
                                                                          /// outerBeamSpot and then recorrecting back to the CMS
                                                                          /// coordinate frame would bring to the same result
      /// Momentum
      roughPz = 2*mPtFactor*(outerStubPosition.z()-innerStubPosition.z())/fabs(phioi);
    }

    /// Store Vtx and Momentum
    this->setVertex( GlobalPoint(aVertex.x(), aVertex.y(), zProj) );
    this->setMomentum( GlobalVector( roughPt*cos(vertexphi), roughPt*sin(vertexphi), roughPz ) );
  }



  /// /////////////////// ///
  /// INFORMATIVE METHODS ///
  /// /////////////////// ///

  template< typename T >
  std::string cmsUpgrades::L1TkTracklet< T >::print( unsigned int i ) const {
    std::string padding("");
    for(unsigned int j=0;j!=i;++j)padding+="\t";
    std::stringstream output;
    output<<padding<<"L1TkTracklet:\n";
    padding+='\t';
    output << padding << "Projected Vertex: " << theVertex << '\n';
    output << padding << "Two Point Pt: " << theMomentum.perp() << ", Pz: " << theMomentum.z() << '\n';
    output << padding << "Direction phi: " << theMomentum.phi() << ", eta: " << theMomentum.eta() << '\n';
    for( L1TkTrackletMapIterator it=theStubs.begin() ; it!=theStubs.end() ; ++it )
      output << it->second->print(i+1) << '\n';
    return output.str();
  }

  template< typename T >
  std::ostream& operator << (std::ostream& os, const cmsUpgrades::L1TkTracklet< T >& aL1TkTracklet) {
    return (os<<aL1TkTracklet.print() );
  }

} /// Close namespace

#endif


