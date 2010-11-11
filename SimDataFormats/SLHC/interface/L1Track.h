
/// ////////////////////////////////////////
/// Written by:                          ///
/// Nicola Pozzobon                      ///
/// UNIPD                                ///
/// 2010, November                       ///
/// Major Changes: more fit options, fit ///
/// removed from builder and uses less   ///
/// data members than before             ///
/// ////////////////////////////////////////

#ifndef STACKED_TRACKER_L1_TRACK_FORMAT_H
#define STACKED_TRACKER_L1_TRACK_FORMAT_H

#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerDetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "SimDataFormats/SLHC/interface/LocalStub.h"
#include "SimDataFormats/SLHC/interface/GlobalStub.h"

#include "SimDataFormats/SLHC/interface/L1TrackFit.h"

#include "SLHCUpgradeSimulations/L1Trigger/interface/CircleFit.h"
#include "SLHCUpgradeSimulations/L1Trigger/interface/LineFit.h"

//#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"

namespace cmsUpgrades{

template< typename T >

class L1Track {

  typedef GlobalStub< T >             GlobalStubType;
  typedef edm::Ptr< GlobalStubType >  GlobalStubPtrType;
  typedef LocalStub< T >               LocalStubType;
  typedef edm::Ptr< LocalStubType >    LocalStubPtrType;
  typedef cmsUpgrades::Tracklet< T >       TrackletType;
  typedef edm::Ptr< TrackletType >         TrackletPtrType;
  typedef std::set< std::pair<unsigned int , GlobalStubPtrType> > TrackletMap;
  
  public:

    L1Track()
    {
      storestubs.clear(); // Constructor
      chi2rphi = -999.9;
      chi2zphi = -999.9;
      isfake = 0;
      simtrackid = -999;
      bfield = 4.0; // Constructor
      c_light = 299.792458;
      probwindow = -99.0;
     }

/// another constructor with seed, field, etc...

    L1Track( std::vector< GlobalStubType > thestubs, TrackletType theseed, double thebfield, double pwin ) {
      storestubs = thestubs;
      storeseed = theseed;
      bfield = thebfield;

      c_light = 299.792458;

      probwindow = pwin;

      isfake = 0;
      simtrackid = -999;
      bool isGood = true;
      for (unsigned int k=0; k<storestubs.size(); k++){
        GlobalStubType stub = storestubs.at(k);
        if (stub.isFake()) { /// Exit from loop
          isfake = 1;
          k = storestubs.size();
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
    }

    std::vector< GlobalStubType > getStubs() const {
      return storestubs;    
    }

    int numberStubs() const {
      return this->getStubs().size();
    }

    double probWindow() const{
      return probwindow;
    }

    TrackletType getSeed() const {
      return storeseed;
    }

    int whichSeed() const {
      TrackletMap theStubs = this->getSeed().stubs();
      GlobalStubPtrType innerStub = theStubs.begin()->second;
      GlobalStubPtrType outerStub = theStubs.rbegin()->second;
      /// Renormalize layer number from 5-14 to 0-9
      cmsUpgrades::StackedTrackerDetId innerId = innerStub->Id();
      cmsUpgrades::StackedTrackerDetId outerId = outerStub->Id();
      int normLayerInner = innerId.layer();
      int normLayerOuter = outerId.layer();
      /// Get the Double Stack index
      int dSidx = normLayerInner/2;
      if (dSidx != (normLayerOuter-1)/2 ) std::cerr << "HOUSTON WE GOT A PROBLEM!" << std::endl;
      return dSidx;
    }

    int isFake() const {
      return isfake;
    }

    int simTrkId() const {
      return simtrackid;
    }

    bool isBeamSpot00() const {
      return this->getSeed().isBeamSpot00();
    }

    /// //////////////////////////////
    /// IDEA FOR TRICKY FIT         //
    /// TAKE ALL TRIPLETS OF STUBS  //
    /// BUILD A TRACKLET WITH THEM  //
    /// FIT THE TRACKLET            //
    /// AVERAGE VALUE OVER TRIPLETS //
    /// //////////////////////////////

    /// ///////////////////////////////////////////////
    /// Put EVERYTHING in a single output structure ///
    /// so that you can call the fit only once      ///
    /// ///////////////////////////////////////////////
    L1TrackFit trickFitL1Track(bool useAlsoVtx) const {
      /// Step 0
      /// Get Stubs chain and Vertex      
      std::vector< GlobalStubType > stubs = this->getStubs();
      /// NOTE
      /// QUESTO È PER QUANDO RIESCO A METTERE IL VECTOR DI SEEDS?
      TrackletType seed = this->getSeed();
      double xbeam = seed.vertex().x(); /// This automatically sets 00 or beamspot according to Tracklet type
      double ybeam = seed.vertex().y();
      /// If the VTX is requested for the fit, add to Stubs
      if (useAlsoVtx) {
        std::vector< GlobalStubType > aux_stubs; aux_stubs.clear();
        GlobalStubType dummy = GlobalStubType(0, seed.vertex(), GlobalVector(0,0,0) );
        aux_stubs.push_back(dummy);
        for (unsigned int j=0; j<stubs.size(); j++) aux_stubs.push_back(stubs.at(j));
        stubs = aux_stubs;
      }

      /// Constants
      double pt_over_rad = 300*4*1e-5; /// 4 T + 300000 km/s approximation
      double pigreco = 4.0*atan(1.0);

      /// Step 1
      /// Find charge
      int imin0 = 0;
      if (useAlsoVtx) imin0 += 1;
      GlobalPoint innermost = stubs.at(imin0).position();
      GlobalPoint outermost = stubs.at(stubs.size()-1).position();
      innermost = GlobalPoint(innermost.x()-xbeam, innermost.y()-ybeam, innermost.x());
      outermost = GlobalPoint(outermost.x()-xbeam, outermost.y()-ybeam, outermost.x());
      double deltaPhi = outermost.phi() - innermost.phi();
      if ( fabs(deltaPhi) >= pigreco) {
        if ( deltaPhi>0 ) deltaPhi = deltaPhi - 2*pigreco;
        else deltaPhi = 2*pigreco - fabs(deltaPhi);
      }
      /// +1 or -1 !!!
      double fit_q = -deltaPhi / fabs(deltaPhi);

      /// Momentum
      std::vector<double> outputFitPt;
      std::vector<double> outputFitPz;
      outputFitPt.clear();
      outputFitPz.clear();
      /// Helix Axis
      std::vector<double> outputFitX;
      std::vector<double> outputFitY;
      outputFitX.clear();
      outputFitY.clear();

      /// Now loop over Triplets
      int totalTriplets=0;
      for (unsigned int a1=0; a1<stubs.size(); a1++) {
        for (unsigned int a2=a1+1; a2<stubs.size(); a2++) {
          for (unsigned int a3=a2+1; a3<stubs.size(); a3++) {
            totalTriplets++;
            /// Read Stubs in a "Tracklet-wise" way
            GlobalPoint vertex = stubs.at(a1).position();
            GlobalPoint inner = stubs.at(a2).position();
            GlobalPoint outer = stubs.at(a3).position();
            /// Correct for position of a1
            inner = GlobalPoint(inner.x()-vertex.x(), inner.y()-vertex.y(), inner.z() );
            outer = GlobalPoint(outer.x()-vertex.x(), outer.y()-vertex.y(), outer.z() );
            /// Angle between a2 and a3 with vertex in a1
            double phidiff = outer.phi() - inner.phi();
            double r1 = inner.perp()/100; /// Check out later ($)
            double r2 = outer.perp()/100;
            double x2 = r1*r1 + r2*r2 - 2*r1*r2*cos(phidiff);
            /// Circle radius
            double radius = 0.5*sqrt(x2)/sin(fabs(phidiff));
            /// Angle between a2 and a3 with vertex in Axis
            double phioi = acos(1 - x2/(2*radius*radius));
            /// Renormalize angles
            if ( fabs(phioi) >= pigreco) {
              if ( phioi>0 ) phioi = phioi - 2*pigreco;
              else phioi = 2*pigreco - fabs(phioi);
            }
            /// Momentum
            outputFitPt.push_back(0.6*sqrt(x2)/sin(fabs(phidiff))); /// That 0.6 cannot be separated from those /100 up there ($)
            outputFitPz.push_back( (outer.z()-inner.z())*pt_over_rad/fabs(phioi) );
            /// Find angle from a1 pointing to Axis
            double vertexangle = acos(0.5*r2/radius);
            vertexangle = outer.phi() - fit_q*vertexangle;
            /// Helix axis
            outputFitX.push_back( 100*radius * cos(vertexangle) + vertex.x() ); /// 100* adjusting unit of measure ($)
            outputFitY.push_back( 100*radius * sin(vertexangle) + vertex.y() );
          } /// End of loop over third element
        } /// End of loop over second element
      } /// End of loop over first element
      double tempOutputX = 0;
      double tempOutputY = 0;
      double tempOutputPt = 0;
      double tempOutputPz = 0;
      for (int q=0; q<totalTriplets; q++) {
        tempOutputX += outputFitX.at(q);
        tempOutputY += outputFitY.at(q);
        tempOutputPt += outputFitPt.at(q);
        tempOutputPz += outputFitPz.at(q);        
      }

      /// Get Helix Axis and correct wrt Seed VTX
      GlobalPoint fit_axis = GlobalPoint(tempOutputX/(double)totalTriplets,
                                         tempOutputY/(double)totalTriplets,
                                         0);
      GlobalPoint axisCorr = GlobalPoint(fit_axis.x()-xbeam,
                                         fit_axis.y()-ybeam,
                                         0);
      /// Calculate Track Phi
      double phiC = atan2( fit_q*axisCorr.x(), -fit_q*axisCorr.y() );

      /// Step 2
      /// Momentum
      /// UNWEIGHTED AVERAGE
      double fit_pt = tempOutputPt/(double)totalTriplets;
      double fit_pz = tempOutputPz/(double)totalTriplets;
      double fit_r  = fit_pt/pt_over_rad;
      GlobalVector fit_momentum = GlobalVector( cos(phiC)*fit_pt,
                                                sin(phiC)*fit_pt,
                                                fit_pz );

      /// Vertex (Closest Approach)
      double rMinAppr = axisCorr.perp() - fit_r;
      double phiCenBS = axisCorr.phi();
      double xMinAppr = rMinAppr*cos(phiCenBS) + xbeam;
      double yMinAppr = rMinAppr*sin(phiCenBS) + ybeam;    

      /// Now loop over Doublets
      /// Cannot put into the same loop as before because
      /// here we need radius and therefore Pt to have
      /// the radius, and the radius is needed to find the
      /// closest approach distance, is it clear?
      int totalDoublets=0;
      std::vector<double> outputFitZ;
      outputFitZ.clear();
      for (unsigned int a1=0; a1<stubs.size(); a1++) {
        for (unsigned int a2=a1+1; a2<stubs.size(); a2++) {
          totalDoublets++;
          GlobalPoint vertex = GlobalPoint(xMinAppr,yMinAppr,0.0);
          GlobalPoint inner = stubs.at(a1).position();
          GlobalPoint outer = stubs.at(a2).position();
          inner = GlobalPoint(inner.x()-vertex.x(), inner.y()-vertex.y(), inner.z() );
          outer = GlobalPoint(outer.x()-vertex.x(), outer.y()-vertex.y(), outer.z() );
          double phidiff = outer.phi() - inner.phi();
          double r1 = inner.perp();
          double r2 = outer.perp();
          double x2 = r1*r1 + r2*r2 - 2*r1*r2*cos(phidiff);
          double radius = 0.5*sqrt(x2)/sin(fabs(phidiff));
          double phioi = acos(1 - x2/(2*radius*radius));
          double phiiv = acos(1 - r1*r1/(2*radius*radius));
          ///
          if ( fabs(phioi) >= pigreco) {
            if ( phioi>0 ) phioi = phioi - 2*pigreco;
            else phioi = 2*pigreco - fabs(phioi);
          }    
          if ( fabs(phiiv) >= pigreco) {
            if ( phiiv>0 ) phiiv = phiiv - 2*pigreco;
            else phiiv = 2*pigreco - fabs(phiiv);
          }
          outputFitZ.push_back( inner.z() - (outer.z()-inner.z())*phiiv/phioi );
        } /// End of loop over second element
      } /// End of loop over first element

      double tempOutputZ = 0;
      for (int q=0; q<totalDoublets; q++) tempOutputZ += outputFitZ.at(q);
      double zMinAppr = tempOutputZ/(double)totalDoublets;

      /// Step 3
      /// Vertex
      /// UNWEIGHTED AVERAGE
      GlobalPoint fit_vertex = GlobalPoint(xMinAppr, yMinAppr, zMinAppr);

      /// Step 4
      /// Return output
      /// Charge
      /// Radius
      /// Momentum - perp(), z(), eta(), phi()
      /// Vertex - x(), y(), z()
      /// Axis - x(), y(), z()==0
      return L1TrackFit(fit_q, fit_r, fit_momentum, fit_vertex, fit_axis);
    }

    /// Least Squares one
    L1TrackFit fitL1Track(bool useAlsoVtx) const {
      /// Get Stubs chain and Vertex      
      std::vector< GlobalStubType > stubs = this->getStubs();
      /// NOTE
      /// QUESTO È PER QUANDO RIESCO A METTERE IL VECTOR DI SEEDS?
      TrackletType seed = this->getSeed();
      double xbeam = seed.vertex().x(); /// This automatically sets 00 or beamspot according to Tracklet type
      double ybeam = seed.vertex().y();

      /// Find innermost and outermost Stubs
      double rmin = 9999999.9;
      double rmax = 0;
      int imin = 0;
      int imax = 0;

      /// Get Stub coordinates and prepare them for the fit
      std::vector< GlobalPoint > stubspoints;
      stubspoints.clear();
      /// Add VTX if requested
      if (useAlsoVtx) {
        GlobalPoint p0 = seed.vertex();
        /// Correct for beamspot, if needed.
        GlobalPoint p( p0.x()-xbeam, p0.y()-ybeam, p0.z() );
        stubspoints.push_back(p);
      }
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
      int imin0 = imin;
      if (useAlsoVtx) imin0 += 1;

      /// Circle fit on r-phi
      CircleFit cf( stubspoints );
      std::vector<double> c1;
        c1 = cf.modifiedLeastSquaresFit();

      /// Set Radius, center of circumference and chi2
      double fit_radius = c1[0];
      GlobalPoint fit_axis = GlobalPoint(c1[1] + xbeam, c1[2] + ybeam, 0 ); /// re-correct
      //chi2rphi = c1[3];
      /// Find minimum approach to beamspot
      double phiCenBS = atan2( c1[2], c1[1] );
      /// if > 0, beamspot outside, else inside
      double rMinAppr = sqrt( c1[1]*c1[1] + c1[2]*c1[2] ) - fit_radius;
      /// sign of radius adjusts for relative coordinates
      double xMinAppr = rMinAppr*cos(phiCenBS);
      double yMinAppr = rMinAppr*sin(phiCenBS);
      /// Set Pt
      /// assumes *1.0 at the end, unit charge!
      double fit_pt = c_light * 1e-5 * bfield * fit_radius;

      /// Step 1
      /// Find charge
      double deltaPhi = stubspoints.at(imax).phi() - stubspoints.at(imin0).phi();
      double pigreco = 4.0*atan(1.0);
      if ( fabs(deltaPhi) >= pigreco) {
        if ( deltaPhi>0 ) deltaPhi = deltaPhi - 2*pigreco;
        else deltaPhi = 2*pigreco - fabs(deltaPhi);
      }
      /// +1 or -1 !!!
      double fit_charge = - deltaPhi / fabs(deltaPhi);

      /// Set track phi at vertex
      double phiTrkMomMinAppr = atan2( fit_charge*c1[1], -fit_charge*c1[2] );

      /// Prepare points in the helix axis reference
      std::vector< GlobalPoint > stubspoints_h;
      stubspoints_h.clear();
      for (unsigned int i=0; i<stubs.size(); i++) {
        GlobalPoint p0 = stubs.at(i).position();
        /// Correct for beamspot, if needed.
        GlobalPoint p( p0.x()-fit_axis.x(), p0.y()-fit_axis.y(), p0.z() );
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
      /// NOTE that angles are measured always in the same way
      /// we must not take a positive charge as a negative one
      /// going backwards, so this is why we need the "-charge*"
      /// factor at the beginning. if the particle is negative,
      /// pz = pt/(rad * angcoeff), if it is positive, the correct
      /// sign is restored
      /// Corresonding one in Tracklet fit is already fine.
      double fit_pz = - fit_charge * fit_pt/(fit_radius * angcoeff);

      /// Step 2
      /// Momentum
      GlobalVector fit_momentum = GlobalVector(fit_pt*cos(phiTrkMomMinAppr), fit_pt*sin(phiTrkMomMinAppr), fit_pz);
      /// Step 3
      /// Vertex
      GlobalPoint fit_vertex = GlobalPoint(xMinAppr + xbeam, yMinAppr + ybeam, zMinAppr);

      /// Set eta and chi2
      //eta = -log( tan ( 0.5*atan2( pt, pz ) ) );
      //chi2zphi = l1[2];
      
      /// Step 4
      /// Return output
      /// Charge
      /// Radius
      /// Momentum - perp(), z(), eta(), phi()
      /// Vertex - x(), y(), z()
      /// Axis - x(), y(), z()==0
      return L1TrackFit(fit_charge, fit_radius, fit_momentum, fit_vertex, fit_axis);
    }



    ~L1Track(){}

  private:
      std::vector< GlobalStubType > storestubs; /// The Stubs
      TrackletType storeseed;                   /// The Seed  
      double chi2rphi;  /// Chi2 of the fit in rphi
      double chi2zphi;  /// Chi2 of the fit in rz
      int isfake;       /// Genuinity:
                        /// =0 good, =1 fake Stub,
                        /// =2 good stubs from different tracks,
                        /// =99 less than 3 Stubs in the Track
      int simtrackid;   /// TrackID in Simulations
      double bfield;    /// magnetic field
      double probwindow;
      /// The speed of light in mm/ns (!) without clhep (yeaaahhh!)
      double c_light;
};


}

template<  typename T  >
std::ostream& operator << (std::ostream& os, const cmsUpgrades::L1Track< T >& aL1Track) {
  return (os<<aL1Track.print() );
//  return os;
}

#endif



