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
/// Possibility to have a flag telling   ///
/// if the Tracklet is ok with the       ///
/// constraints imposed by hermetic      ///
/// design 'isHermetic()'.               ///
/// Possibility to have Fake Tracklet    ///
/// flag in Simulations 'isFake()' and   ///
/// Trk ID too 'trackID()'. A Tracklet   ///
/// is flagged as Fake in the            ///
/// TrackletBuilder in two cases which   ///
/// are distinguished from each other:   ///
/// 1) at least one Stub is Fake (see    ///
/// GlobalStub.h)                        ///
/// 2) both Stubs are genuine but coming ///
/// from different SimTracks             ///
/// More details in the TrackletBuilder. ///
/// Possibility to have a Tracklet VTX   ///
/// which is different from the default  ///
/// one (0,0,z). This is obtained in the ///
/// TrackletBuilder which tries to build ///
/// 2 Tracklets from each pair of Stubs, ///
/// one with (0,0,z), one with the other ///
/// option which is supposed to be the   ///
/// "true" one: 'isBeamSpot00()'         ///
/// Added momentum fit with TRUE helix   ///
/// ////////////////////////////////////////

#ifndef TRACKLET_H
#define TRACKLET_H

#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerDetId.h"
#include "SimDataFormats/SLHC/interface/GlobalStub.h"
//#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"

//for the helper methods
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"

namespace cmsUpgrades{

template< typename T >
class Tracklet {

     typedef GlobalStub< T >                     GlobalStubType;
    typedef edm::Ptr< GlobalStubType >  GlobalStubPtrType;


public:
    typedef  std::set< std::pair<unsigned int , GlobalStubPtrType> >    TrackletMap;
    typedef typename TrackletMap::const_iterator                        TrackletMapIterator;

    Tracklet()
    {
      vtx00 = true;
      hermeticity = false;
      fakeness = -9999;
      trackid = -9999;
      theStubs.clear();
      theVertex22X=GlobalPoint(0.0,0.0,0.0); /// Vertex from OLD 2_2_X Fit
      theVertex=GlobalPoint(0.0,0.0,0.0);    /// Vertex from NEW Fit
      theAxis=GlobalPoint(0.0,0.0,0.0);      /// Helix Axis from NEW Fit
      NullStub=GlobalStubPtrType();
    }


    virtual ~Tracklet(){}

    void addHit(  unsigned int aStubIdentifier,
            const GlobalStubPtrType &aHit )
    {
        theStubs.insert( std::make_pair( aStubIdentifier , aHit ) ); 
    }

    /// New Fit VTX
    void addVertex( const GlobalPoint & aVertex )
    {
      theVertex=aVertex;
    }
    const GlobalPoint& vertex() const
    {
      return theVertex;
    }

    /// Old Fit VTX
    void addVertex22X( const GlobalPoint & aVertex )
    {
      theVertex22X=aVertex;
    }
    const GlobalPoint& vertex22X() const
    {
      return theVertex22X;
    }

    /// New Fit Helix Axis
    void addAxis( const GlobalPoint & aAxis )
    {
      theAxis=aAxis;
    }
    const GlobalPoint& axis() const
    {
      return theAxis;
    }

    /// Hermetic Phi-coverage of Double Stack ladders
    /// FNAL design
    void setHermetic( bool a ) {
      hermeticity = a;
    }
    bool isHermetic() const {
      return hermeticity;
    }

    /// Fake or Good in MC
    void setFakeness( int a ) {
      fakeness = a;
    }
    int isFake() const {
      return fakeness; /// 0 good, 1 fakeStub, 2 fakeTracklet
    }

    /// SimTrack ID for Good Tracklets in MC
    void setTrackID( int a ) {
      trackid = a;
    }
    int trackID() const {
      return trackid;
    }

    /// Beamline VTX
    void setBeamSpot00( bool a ) {
      vtx00 = a;
    }
    bool isBeamSpot00() const {
      return vtx00;
    }

    /// Already present in 2_2_X
    const GlobalStubPtrType &stub( unsigned int aStubIdentifier ) const
    {
      for (TrackletMapIterator i = theStubs.begin(); i != theStubs.end(); ++i){
        if ( i->first == aStubIdentifier ) return i->second;
      }
      return NullStub;
    }

    /// Already present in 2_2_X
    const TrackletMap& stubs() const
    {
      return theStubs;
    }

    /// useful methods
    /// OLD Fit in 2_2_X, still available for comparison
    FastHelix HelixFit(const edm::EventSetup& iSetup) const
    {
      return  FastHelix(  theStubs.rbegin()->second->position(),
                          theStubs.begin()->second->position(),
                          theVertex22X,
                          iSetup );
    }
  
    FreeTrajectoryState VertexTrajectoryState(const edm::EventSetup& iSetup) const
    {
      AlgebraicSymMatrix errorMatrix(5,1);
      CurvilinearTrajectoryError initialError(errorMatrix*100.);
      return  FreeTrajectoryState(this->HelixFit(iSetup).stateAtVertex().parameters(),initialError);
    }

    double deltaPhiNorm() const
    {
       GlobalPoint inner = theStubs.begin()->second->position();
       GlobalPoint outer = theStubs.rbegin()->second->position();
       if (this->isBeamSpot00()==false) {
         inner = GlobalPoint(inner.x()-this->vertex().x(), inner.y()-this->vertex().y(), inner.z() );
         outer = GlobalPoint(outer.x()-this->vertex().x(), outer.y()-this->vertex().y(), outer.z() );
       }
       double deltaPhiTracklet = outer.phi() - inner.phi();
       double pigreco = 4.0*atan(1.0);
       if ( fabs(deltaPhiTracklet) >= pigreco) {
        if ( deltaPhiTracklet>0 ) deltaPhiTracklet = deltaPhiTracklet - 2*pigreco;
        else deltaPhiTracklet = 2*pigreco - fabs(deltaPhiTracklet);
       }
       return deltaPhiTracklet;
    }

    double twoPointPt() const
    {
      GlobalPoint inner = theStubs.begin()->second->position();
      GlobalPoint outer = theStubs.rbegin()->second->position();
      if (this->isBeamSpot00()==false) {
        inner = GlobalPoint(inner.x()-this->vertex().x(), inner.y()-this->vertex().y(), inner.z() );
        outer = GlobalPoint(outer.x()-this->vertex().x(), outer.y()-this->vertex().y(), outer.z() );
      }
      double phidiff = outer.phi() - inner.phi();
      double r1 = inner.perp()/100;
      double r2 = outer.perp()/100;
      double x2 = r1*r1 + r2*r2 - 2*r1*r2*cos(phidiff);
      return 0.6*sqrt(x2)/sin(fabs(phidiff));
      //return 0.0;
    }

    double twoPointPz() const {
      GlobalPoint inner = theStubs.begin()->second->position();
      GlobalPoint outer = theStubs.rbegin()->second->position();
      /// No need of distinction between beamspot and 00
      /// Find circumference center using Cramer!
      double xo = outer.x();
      double yo = outer.y();
      double xi = inner.x();
      double yi = inner.y();
      double xc = this->axis().x();
      double yc = this->axis().y();
      /// Find angles wrt center
      double phio = atan2( yo-yc, xo-xc );
      double phii = atan2( yi-yc, xi-xc );
      double phioi = phio - phii;
      double pigreco = 4.0*atan(1.0);
      if ( fabs(phioi) >= pigreco) {
        if ( phioi>0 ) phioi = phioi - 2*pigreco;
        else phioi = 2*pigreco - fabs(phioi);
      }
      double pt_over_rad = 300*4*1e-5;
      return (outer.z()-inner.z())*pt_over_rad/fabs(phioi);
    }
    
    double twoPointEta() const {
      /// No need of distinction between beamspot and 00
      /// already embedded in called methods
      return -log( tan ( 0.5*atan2( this->twoPointPt(), this->twoPointPz() ) ) );
    }
    
    double twoPointPhi() const {
      double trkQ = -this->deltaPhiNorm() / fabs( this->deltaPhiNorm() );
      double xv = this->vertex().x();
      double yv = this->vertex().y();
      double xc = this->axis().x();
      double yc = this->axis().y();
      /// Find angles wrt center
      return atan2( trkQ*(xc-xv), -trkQ*(yc-yv) );
    }

    std::string print(unsigned int i = 0 ) const {
                        std::string padding("");
                        for(unsigned int j=0;j!=i;++j)padding+="\t";

      std::stringstream output;
      output<<padding<<"Tracklet:\n";
      padding+='\t';
      output << padding << "Projected Vertex: " << theVertex << '\n';
      output << padding << "Two Point Pt: " << this->twoPointPt() << '\n';

      for( TrackletMapIterator it=theStubs.begin() ; it!=theStubs.end() ; ++it )
        output << it->second->print(i+1) << '\n';

      return output.str();
    }



  private:
  //which hits formed the tracklet
    bool hermeticity;
    int fakeness;
    int trackid;
    TrackletMap theStubs;
    GlobalPoint theVertex;
    GlobalPoint theVertex22X;
    GlobalPoint theAxis;
    bool vtx00;
//    TrajectoryMap theTrajectories;

    GlobalStubPtrType NullStub;

};

}

template<  typename T  >
std::ostream& operator << (std::ostream& os, const cmsUpgrades::Tracklet< T >& aTracklet) {
  return (os<<aTracklet.print() );
//  return os;
}

#endif

