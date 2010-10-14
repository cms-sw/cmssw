
/// ////////////////////////////////////////
/// Written by:                          ///
/// Nicola Pozzobon                      ///
/// UNIPD                                ///
/// 2010, September                      ///
/// ////////////////////////////////////////

#ifndef STACKED_TRACKER_L1_TRACK_FORMAT_H
#define STACKED_TRACKER_L1_TRACK_FORMAT_H

#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerDetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "SimDataFormats/SLHC/interface/LocalStub.h"
#include "SimDataFormats/SLHC/interface/GlobalStub.h"

#include "SLHCUpgradeSimulations/L1Trigger/interface/CircleFit.h"
#include "SLHCUpgradeSimulations/L1Trigger/interface/LineFit.h"

//#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"

namespace cmsUpgrades{

template< typename T >

class L1Track {

  typedef GlobalStub< T >             GlobalStubType;
  typedef edm::Ptr< GlobalStubType >  GlobalStubPtrType;
	typedef LocalStub< T > 							LocalStubType;
	typedef edm::Ptr< LocalStubType >	  LocalStubPtrType;
	
	public:

		L1Track()
		{
      numstubs = 0;  // Constructor
      seed = -999;   // Constructor
      stubs.clear(); // Constructor
      chi2rphi = -999.9;
      chi2zphi = -999.9;
      charge = 0;
      pt = -999.9;
      x0 = 0;
      y0 = 0;
      rad = -999.9;
      xvtx = 0;
      yvtx = 0;
      zvtx = 0;
      eta = 0;
      phi = 0;
      isfake = 0;
      simtrackid = -999;
      bfield = 4.0; // Constructor
      xbeam = 0.0;  // Constructor
      ybeam = 0.0;  // Constructor
      c_light = 299.792458;
      isbeamspot00 = true;
      probwindow = -99.0;
	 	}

/// another constructor with seed, field, etc...

    L1Track( std::vector< GlobalStubType > thestubs, int theseed, double thebfield, double theXbeam, double theYbeam, double pwin ) {
      stubs = thestubs;
      seed = theseed;
      numstubs = thestubs.size();
      bfield = thebfield;
      xbeam = theXbeam;
      ybeam = theYbeam;
      c_light = 299.792458;
      
      probwindow = pwin;

      isfake = 0;

      simtrackid = -999;
      bool isGood = true;
      //if (stubs.size() < 3) isfake = 99;
      //else {
        for (unsigned int k=0; k<stubs.size(); k++){
          GlobalStubType stub = stubs.at(k);
          if (stub.isFake()) { /// Exit from loop
            isfake = 1;
            k = stubs.size();
            continue;
          }
          /// Here only if > 3 Stubs and no fake Stub yet
          bool good = (isfake==0);
          isGood = isGood && good;
          if (isGood) {
            if (stub.trackID() > -9. && simtrackid<0) simtrackid = stub.trackID();
            if (simtrackid != stub.trackID()) isGood = false;
          }
        }
        if (!isGood) isfake = 2;
        if (isfake!=0) simtrackid = -999;
        
      //}
    }

    std::vector< GlobalStubType > getStubs() const {
      return stubs;    
    }

    int numberStubs() const {
      return numstubs;
    }

    double probWindow() const{
      return probwindow;
    }

    int whichSeed() const {
      return seed;
    }

    int trkCharge() const {
      return charge;
    }

    double fitChi2RPhi() const {
      return chi2rphi;
    }

    double fitChi2ZPhi() const {
      return chi2zphi;
    }

    double Pt() const {
      return pt;
    }

    double fitCircX0() const {
      return x0;
    }

    double fitCircY0() const {
      return y0;
    }

    double fitRadius() const {
      return rad;
    }

    GlobalPoint fitVertex() const {
      return GlobalPoint (xvtx, yvtx, zvtx);
    }

    double fitEta() const {
      return eta;
    }

    double fitPhi() const {
      return phi;
    }

    int isFake() const {
      return isfake;
    }

    int simTrkId() const {
      return simtrackid;
    }

    void SetBeamSpot00 (bool p) {
      isbeamspot00 = p;
    }

    bool isBeamSpot00() const {
      return isbeamspot00;
    }

    void fitL1Track() {
      /// Find innermost and outermost Stubs
      double rmin = 9999999.9;
      double rmax = 0;
      int imin = 0;
      int imax = 0;
      std::vector< GlobalPoint > stubspoints;
      stubspoints.clear();
      for (unsigned int i=0; i<stubs.size(); i++) {
        GlobalPoint p0 = stubs.at(i).position();
        /// Correct for beamspot, if needed.
        GlobalPoint p( p0.x()-xbeam, p0.y()-ybeam, p0.z() );
        stubspoints.push_back(p);
        if (p0.perp() < rmin) {
          rmin = p0.perp();
          imin = i;
        }
        if (p0.perp() > rmax) {
          rmax = p0.perp();
          imax = i;
        }
      }

      /// Circle fit on r-phi
      CircleFit cf( stubspoints );
  
      std::vector<double> c1;
      //if (bisec) c1 = cf.perpendicularBisectorFit();
      //else
        c1 = cf.modifiedLeastSquaresFit();

      /// Set Radius, center of circumference and chi2
      rad = c1[0];
      x0 = c1[1] + xbeam; /// re-correct
      y0 = c1[2] + ybeam;
      chi2rphi = c1[3];

      /// Find minimum approach to beamspot
      double phiCenBS = atan2( c1[2], c1[1] );
      /// if > 0, beamspot outside, else inside
      double rMinAppr = sqrt( c1[1]*c1[1] + c1[2]*c1[2] ) - rad;
      /// sign of radius adjusts for relative coordinates
      double xMinAppr = rMinAppr*cos(phiCenBS);
      double yMinAppr = rMinAppr*sin(phiCenBS);

      /// Set Pt
      /// assumes *1.0 at the end, unit charge!
      pt = c_light * 1e-5 * bfield * rad;

      /// Set charge
      double deltaPhi = stubspoints.at(imax).phi() - stubspoints.at(imin).phi();
      double pigreco = 4.0*atan(1.0);
      if ( fabs(deltaPhi) >= pigreco) {
        if ( deltaPhi>0 ) deltaPhi = deltaPhi - 2*pigreco;
        else deltaPhi = 2*pigreco - fabs(deltaPhi);
      }
      /// +1 or -1 !!!
      charge = - deltaPhi / fabs(deltaPhi);

      /// Set track phi at vertex
      double phiTrkMomMinAppr = atan2( charge*c1[1], -charge*c1[2] );
      phi = phiTrkMomMinAppr;

      /// Prepare points in the helix axis reference
      std::vector< GlobalPoint > stubspoints_h;
      stubspoints_h.clear();
      for (unsigned int i=0; i<stubs.size(); i++) {
        GlobalPoint p0 = stubs.at(i).position();
        /// Correct for beamspot, if needed.
        GlobalPoint p( p0.x()-x0, p0.y()-y0, p0.z() );
        stubspoints_h.push_back(p);
      }
 
      /// Linear fit on z-phi
      LineFit lf( stubspoints_h );
      std::vector<double> l1 = lf.modifiedLeastSquaresFit();

      /// Find angle wrt helix axis
      double phiCenBSCompl = atan2( -c1[2], -c1[1] );
      double angcoeff = l1[0];
      double phi0 = l1[1];
      /// Now we have phi = a + b * z
      /// b = l[0]
      /// a = l[1]
      /// We can find VTX z, first of all.
      /// z = (phi - a) / b
      double zMinAppr = (phiCenBSCompl - phi0) / angcoeff;
      xvtx = xMinAppr + xbeam;
      yvtx = yMinAppr + ybeam;
      zvtx = zMinAppr;

      /// NOTE that angles are measured always in the same way
      /// we must not take a positive charge as a negative one
      /// going backwards, so this is why we need the "-charge*"
      /// factor at the beginning. if the particle is negative,
      /// pz = pt/(rad * angcoeff), if it is positive, the correct
      /// sign is restored
      /// Corresonding one in Tracklet fit is already fine.
      double pz = - charge * pt/(rad * angcoeff);

      /// Set eta and chi2
      eta = -log( tan ( 0.5*atan2( pt, pz ) ) );
      chi2zphi = l1[2];
      
      /// End of fit
    
    }

		~L1Track(){}

	private:
      int numstubs;                        /// Number of Stubs
      int seed;                            /// Seed Super Layer
      std::vector< GlobalStubType > stubs; /// The Stubs
      double chi2rphi;  /// Chi2 of the fit in rphi
      double chi2zphi;    /// Chi2 of the fit in rz
      double charge;    /// +1 or -1
      double pt;        /// pt from circle fit
      double x0;        /// x of center of circumference
      double y0;        /// y of center of circumference
      double rad;       /// radius of circumference
      double xvtx;      /// x of point of min approach to beamline used for the fit
      double yvtx;      /// y of point of min approach to beamline used for the fit
      double zvtx;      /// z of point of min approach to beamline used for the fit
      double eta;       /// direction of the line used for the rz fit
      double phi;       /// direction of tangent in rphi to the point of min approach
      int isfake;       /// Genuinity:
                        /// =0 good, =1 fake Stub,
                        /// =2 good stubs from different tracks,
                        /// =99 less than 3 Stubs in the Track
      int simtrackid;   /// TrackID in Simulations
      bool isbeamspot00;   /// Assumed VTX for seed Tracklet
      double bfield;    /// magnetic field
      double xbeam;
      double ybeam;
      double probwindow;
      /// The speed of light in mm/ns (!) without clhep (yeaaahhh!)
      double c_light;
};


}

template<	typename T	>
std::ostream& operator << (std::ostream& os, const cmsUpgrades::L1Track< T >& aL1Track) {
	return (os<<aL1Track.print() );
//	return os;
}

#endif


