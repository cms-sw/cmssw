#ifndef BeamSpotProducer_BSzFcn_h
#define BeamSpotProducer_BSzFcn_h

/**_________________________________________________________________
   class:   BSzFcn.h
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BSzFcn.h,v 1.0 2006/09/19 17:13:31 yumiceva Exp $

________________________________________________________________**/

#include "Minuit2/FCNBase.h"

#include "RecoVertex/BeamSpotProducer/interface/BSTrkParameters.h"

#include <iostream>

using namespace ROOT::Minuit2;


class BSzFcn : public FCNBase {
	
  public:
	// cache the current data
	void SetData(std::vector < BSTrkParameters > a_BSvector){
		//std::cout << "SetData called"<<std::endl;
		fBSvector = a_BSvector;
		//std::cout << "Size: "<<  fBSvector.size() << std::endl;
	};

	virtual double operator() (const std::vector<double>&) const;
	virtual double Up() const {return 1.;}
	
  private:
	double Gauss_z(double z, double sigma, const std::vector<double>& parms) const;
	std::vector < BSTrkParameters > fBSvector;
	static const int fPar_Z0      = 0;  // index of position of luminosity peak in z
	static const int fPar_SigmaZ  = 1;  // index of sigma in z of the beam
};

#endif
