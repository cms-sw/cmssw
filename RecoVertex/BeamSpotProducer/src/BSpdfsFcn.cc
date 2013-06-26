
/**_________________________________________________________________
   class:   BSpdfsFcn.cc
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BSpdfsFcn.cc,v 1.1 2006/12/15 20:00:37 yumiceva Exp $

________________________________________________________________**/

#include "RecoVertex/BeamSpotProducer/interface/BSpdfsFcn.h"
#include "TMath.h"

#include <cmath>
#include <vector>

//______________________________________________________________________
double BSpdfsFcn::PDFGauss_d(double z, double d, double sigmad,
							 double phi, const std::vector<double>& parms) const {

  //---------------------------------------------------------------------------
  //  PDF for d0 distribution. This PDF is a simple gaussian in the
  //  beam reference frame.
  //---------------------------------------------------------------------------
	double fsqrt2pi = sqrt(2.* TMath::Pi());
	
	double sig = sqrt(parms[fPar_SigmaBeam]*parms[fPar_SigmaBeam] +
					  sigmad*sigmad);

	double dprime = d - ( ( parms[fPar_X0] + z*parms[fPar_dxdz] )*sin(phi)
		- ( parms[fPar_Y0] + z*parms[fPar_dydz] )*cos(phi) );
	
	double result = (exp(-(dprime*dprime)/(2.0*sig*sig)))/(sig*fsqrt2pi);
		
	return result;

}

//______________________________________________________________________
double BSpdfsFcn::PDFGauss_d_resolution(double z, double d, double phi, double pt, const std::vector<double>& parms) const {

  //---------------------------------------------------------------------------
  //  PDF for d0 distribution. This PDF is a simple gaussian in the
  //  beam reference frame. The IP resolution is parametrize by a linear
  //  function as a function of 1/pt.	
  //---------------------------------------------------------------------------
	double fsqrt2pi = sqrt(2.* TMath::Pi());

	double sigmad = parms[fPar_c0] + parms[fPar_c1]/pt;
	
	double sig = sqrt(parms[fPar_SigmaBeam]*parms[fPar_SigmaBeam] +
					  sigmad*sigmad);

	double dprime = d - ( ( parms[fPar_X0] + z*parms[fPar_dxdz] )*sin(phi)
		- ( parms[fPar_Y0] + z*parms[fPar_dydz] )*cos(phi) );
	
	double result = (exp(-(dprime*dprime)/(2.0*sig*sig)))/(sig*fsqrt2pi);
		
	return result;

}

//______________________________________________________________________
double BSpdfsFcn::PDFGauss_z(double z, double sigmaz, const std::vector<double>& parms) const {

  //---------------------------------------------------------------------------
  //  PDF for z-vertex distribution. This distribution
  // is parametrized by a simple normalized gaussian distribution.
  //---------------------------------------------------------------------------
	double fsqrt2pi = sqrt(2.* TMath::Pi());

	double sig = sqrt(sigmaz*sigmaz+parms[fPar_SigmaZ]*parms[fPar_SigmaZ]);
	//double sig = sqrt(sigmaz*sigmaz+parms[1]*parms[1]);
	double result = (exp(-((z-parms[fPar_Z0])*(z-parms[fPar_Z0]))/(2.0*sig*sig)))/(sig*fsqrt2pi);
	//double result = (exp(-((z-parms[0])*(z-parms[0]))/(2.0*sig*sig)))/(sig*fsqrt2pi);
        
	return result;

}




//______________________________________________________________________
double BSpdfsFcn::operator() (const std::vector<double>& params) const {
		
	double f = 0.0;

	//std::cout << "fusepdfs=" << fusepdfs << " params.size="<<params.size() << std::endl;
	
	std::vector<BSTrkParameters>::const_iterator iparam = fBSvector.begin();

	double pdf = 0;
	
	for( iparam = fBSvector.begin(); iparam != fBSvector.end(); ++iparam) {

		
		if (fusepdfs == "PDFGauss_z") {
			pdf = PDFGauss_z( iparam->z0(), iparam->sigz0(),params);
		}
		else if (fusepdfs == "PDFGauss_d") {
			pdf = PDFGauss_d( iparam->z0(), iparam->d0(),
							  iparam->sigd0(), iparam->phi0(),params);
		}
		else if (fusepdfs == "PDFGauss_d_resolution") {
			pdf = PDFGauss_d_resolution( iparam->z0(), iparam->d0(),
							  iparam->phi0(), iparam->pt(),params);
		}
		else if (fusepdfs == "PDFGauss_d*PDFGauss_z") {
			//std::cout << "pdf= " << pdf << std::endl;
			pdf = PDFGauss_d( iparam->z0(), iparam->d0(),
							  iparam->sigd0(), iparam->phi0(),params)*
				PDFGauss_z( iparam->z0(), iparam->sigz0(),params);
		}
		else if (fusepdfs == "PDFGauss_d_resolution*PDFGauss_z") {
			pdf = PDFGauss_d_resolution( iparam->z0(), iparam->d0(),
										 iparam->phi0(), iparam->pt(),params)*
				PDFGauss_z( iparam->z0(), iparam->sigz0(),params);
		}
		
		f = log(pdf) + f;
    }

	f= -2.0*f;
	return f;

}
