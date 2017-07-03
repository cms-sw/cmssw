#ifndef BeamSpotProducer_BSpdfsFcn_h
#define BeamSpotProducer_BSpdfsFcn_h

/**_________________________________________________________________
   class:   BSpdfsFcn.h
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)


________________________________________________________________**/

#include "Minuit2/FCNBase.h"

#include "RecoVertex/BeamSpotProducer/interface/BSTrkParameters.h"

#include <iostream>
#include <string>

class BSpdfsFcn : public ROOT::Minuit2::FCNBase {
	
  public:
	// cache the current data
	void SetData(const std::vector < BSTrkParameters > &a_BSvector){
		
		fBSvector = a_BSvector;
		
	};
	// define pdfs to use
	void SetPDFs(std::string usepdfs) {
		fusepdfs = usepdfs;
	}

	double operator() (const std::vector<double>&) const override;
	double Up() const override {return 1.;}
	
  private:

	double PDFGauss_d(double z, double d, double sigmad,
					  double phi, const std::vector<double>& parms) const;
	double PDFGauss_d_resolution(double z, double d,
								 double phi, double pt, const std::vector<double>& parms) const;

	double PDFGauss_z(double z, double sigmaz, const std::vector<double>& parms) const;

	std::string fusepdfs;
	std::vector < BSTrkParameters > fBSvector;

	static const int fPar_X0      = 0;  // 
	static const int fPar_Y0      = 1;  //
	static const int fPar_Z0      = 2;  // position of luminosity peak in z
	static const int fPar_SigmaZ  = 3;  // sigma in z of the beam
	static const int fPar_dxdz    = 4;  // 
	static const int fPar_dydz    = 5;  //
	static const int fPar_SigmaBeam = 6;  // 
	static const int fPar_c0      = 7;  //
	static const int fPar_c1      = 8;  //
	
	//static const int fPar_Z0      = 0;  // index of position of luminosity peak in z
	//static const int fPar_SigmaZ  = 1;  // index of sigma in z of the beam
	//static const int fPar_X0      = 2;  // 
	//static const int fPar_Y0      = 3;  //
	//static const int fPar_dxdz    = 4;  // 
	//static const int fPar_dydz    = 5;  //
	//static const int fPar_SigmaBeam = 6;  // 
	//static const int fPar_c0      = 7;  //
	//static const int fPar_c1      = 8;  //
	
};

#endif
