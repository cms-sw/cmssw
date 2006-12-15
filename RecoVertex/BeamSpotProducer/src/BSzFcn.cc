
/**_________________________________________________________________
   class:   BSzFcn.cc
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BSzFcn.cc,v 1.0 2006/09/19 17:13:31 yumiceva Exp $

________________________________________________________________**/

#include "RecoVertex/BeamSpotProducer/interface/BSzFcn.h"
#include "TMath.h"

#include <cmath>
#include <vector>

//______________________________________________________________________
double BSzFcn::Gauss_z(double z, double sigmaz, const std::vector<double>& parms) const {

  //---------------------------------------------------------------------------
  //  This is a simple function to parameterize the z-vertex distribution. This 
  // is parametrized by a simple normalized gaussian distribution. 
  //---------------------------------------------------------------------------
	double fsqrt2pi = sqrt(2.* TMath::Pi());
	
	double sig = sqrt(sigmaz*sigmaz+parms[fPar_SigmaZ]*parms[fPar_SigmaZ]);
	double result = (exp(-((z-parms[fPar_Z0])*(z-parms[fPar_Z0]))/(2.0*sig*sig)))/(sig*fsqrt2pi);
	//std::cout << "BSzFcn:: result= "<< result << " z="<< z
	//		  << " sigma=" << sigmaz << " parms0="<< parms[fPar_Z0]
	//		  << " parms1="<< parms[fPar_SigmaZ] << std::endl;
	
	return result;

}

//______________________________________________________________________
double BSzFcn::operator() (const std::vector<double>& params) const {

	//std::cout << "params.size="<< params.size() << std::endl;
	//std::cout << "params[0]= " << params[0] << " params[1]="<< params[1] << std::endl;
	
	double f = 0.0;

	std::vector<BSTrkParameters>::const_iterator iparam = fBSvector.begin();
		
	for( iparam = fBSvector.begin(); iparam != fBSvector.end(); ++iparam) {
		
		f = log(Gauss_z( iparam->z0(), iparam->sigz0(),params))+f;
    }

	f= -2.0*f;
	return f;
}
