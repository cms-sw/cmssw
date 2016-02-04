#ifndef BeamSpotProducer_BSTrkParameters_h
#define BeamSpotProducer_BSTrkParameters_h

/**_________________________________________________________________
   class:   BSTrkParameters.h
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BSTrkParameters.h,v 1.3 2009/09/14 17:52:27 yumiceva Exp $

________________________________________________________________**/


class BSTrkParameters {

  public:

	// constructor
	BSTrkParameters() {}
	// constructor from values
	//BSTrkParameters( double z0, double sigz0,
	//				 double d0, double sigd0,
	//				 double phi0, double pt) {
	//	fz0 = z0;
	//	fsigz0 = sigz0;
	//	fd0 = d0;
	//	fsigd0 = sigd0;
	//	fphi0 = phi0;
	//	fpt = pt;
	//	
	//};
	
	BSTrkParameters( double z0, double sigz0,
					 double d0, double sigd0,
					 double phi0, double pt,
					 double d0phi_d0=0.,double d0phi_chi2=0.) {
		fz0 = z0;
		fsigz0 = sigz0;
		fd0 = d0;
		fsigd0 = sigd0;
		fphi0 = phi0;
		fpt = pt;
		fd0phi_d0   = d0phi_d0;
		fd0phi_chi2 = d0phi_chi2;
		fvx = 0.;
		fvy = 0.;
	};

    //
	double z0() const { return fz0; }
	double sigz0() const { return fsigz0; }
	double d0() const { return fd0; }
	double sigd0() const { return fsigd0; }
	double phi0() const { return fphi0; }
	double pt() const { return fpt; }
	double d0phi_chi2() const { return fd0phi_chi2; }
	double d0phi_d0() const { return fd0phi_d0; }
	double vx() const { return fvx; }
	double vy() const { return fvy; }
	void setVx( double vx ) { fvx = vx; }
	void setVy( double vy ) { fvy = vy; }
	
  private:
	double fz0;
	double fsigz0;
	double fd0;
	double fsigd0;
	double fphi0;
	double fpt;
	double fd0phi_chi2;
	double fd0phi_d0;
	double fvx;
	double fvy;
};

#endif
