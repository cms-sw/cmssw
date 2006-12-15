#ifndef BeamSpotProducer_BSTrkParameters_h
#define BeamSpotProducer_BSTrkParameters_h

/**_________________________________________________________________
   class:   BSTrkParameters.h
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BSTrkParameters.h,v 1.0 2006/09/19 17:13:31 yumiceva Exp $

________________________________________________________________**/


class BSTrkParameters {

  public:

	// constructor
	BSTrkParameters() {}
	// constructor from values
	BSTrkParameters( double z0, double sigz0,
					 double d0, double sigd0,
					 double phi0, double pt) {
		fz0 = z0;
		fsigz0 = sigz0;
		fd0 = d0;
		fsigd0 = sigd0;
		fphi0 = phi0;
		fpt = pt;
		
	};

	//
	double z0() const { return fz0; }
	double sigz0() const { return fsigz0; }
	double d0() const { return fd0; }
	double sigd0() const { return fsigd0; }
	double phi0() const { return fphi0; }
	double pt() const { return fpt; }
	
  private:
	double fz0;
	double fsigz0;
	double fd0;
	double fsigd0;
	double fphi0;
	double fpt;
};

#endif
