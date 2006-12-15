#ifndef BeamSpotProducer_BeamSpot_h
#define BeamSpotProducer_BeamSpot_h

/**_________________________________________________________________
   class:   BeamSpot.h
   package: RecoVertex/BeamSpotProducer
   
 A reconstructed beam spot providing position, widths, slopes,
 and errors.

 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BeamSpot.h,v 1.0 2006/09/19 17:13:31 yumiceva Exp $

________________________________________________________________**/

#include <Rtypes.h>
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/Math/interface/Point3D.h"
#include <iostream>
#include <string>


class BeamSpot {
  public:
    /// point in the space
    typedef math::XYZPoint Point;
	enum { dimension = 7 };
    typedef math::Error<dimension>::type CovarianceMatrix;
	enum { resdim = 2 };
	typedef math::Error<resdim>::type ResCovMatrix;
	
    /// default constructor
    BeamSpot() { }
    /// constructor from values
    BeamSpot( const Point &point,
			  double sigmaZ,
			  double dxdz,
			  double dydz,
			  double BeamWidth,
			  const CovarianceMatrix &error){
		position_ = point;
		sigmaZ_ = sigmaZ;
		dxdz_ = dxdz;
		dydz_ = dydz;
		BeamWidth_ = BeamWidth;
		error_ = error;
	};
    
    /// position 
    const Point & position() const { return position_; }
    /// x coordinate 
    double x0() const { return position_.X(); }
    /// y coordinate 
    double y0() const { return position_.Y(); }
    /// z coordinate 
    double z0() const { return position_.Z(); }
	/// sigma z
	double sigmaZ() const { return sigmaZ_; }
	/// beam width
	double BeamWidth() const { return BeamWidth_; }
    /// error on x
    double x0Error() const { return sqrt( error_(0,0) ); }
    /// error on y
    double y0Error() const { return sqrt( error_(1,1) ); }
    /// error on z
    double z0Error() const { return sqrt( error_(2,2) ); }
    /// error on sigma z
    double sigmaZ0Error() const { return sqrt ( error_(3,3) ); }

    double dxdzError() const { return sqrt ( error_(4,4) ); }

    double dydzError() const { return sqrt ( error_(5,5) ); }

	/// error on beam width
	double BeamWidthError() const { return sqrt ( error_(6,6) );}
	/// (i,j)-th element of error matrix
	double covariance( int i, int j) const {
		return error_(i,j);
	}
	/// return SMatrix
    CovarianceMatrix covariance() const { return error_; }
    /// dxdz slope 
    double dxdz() const { return dxdz_; }
    /// dydz slope 
    double dydz() const { return dydz_; }

	/// print information
	void Print(std::string message="") {

		std::cout << "---------------------------------------------------\n"
				  << "     Beam Spot: " << message << " Fitter\n\n"
				  << "   X0 = " << x0() << " +/- " << x0Error() << " [cm]\n"
				  << "   Y0 = " << y0() << " +/- " << y0Error() << " [cm]\n"
				  << "   Z0 = " << z0() << " +/- " << z0Error() << " [cm]\n"
				  << " Sigma Z0 = " << sigmaZ() << " +/- " << sigmaZ0Error() << " [cm]\n" 
				  << " dxdz = " << dxdz() << " +/- " << dxdzError() << " [radians]\n"
				  << " dydz = " << dydz() << " +/- " << dydzError() << " [radians]\n"
			          << " Beam Width = " << BeamWidth() << " +/- " << BeamWidthError() << " [cm]\n"
				  << "---------------------------------------------------\n\n";
	};

  private:
    /// position
    Point position_;
	/// errors
    CovarianceMatrix error_;

	Double32_t sigmaZ_;
	Double32_t BeamWidth_;
	Double32_t dxdz_;
	Double32_t dydz_;
	
	
  };

#endif
