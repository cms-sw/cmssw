#ifndef BeamSpotProducer_BSFitter_h
#define BeamSpotProducer_BSFitter_h

/**_________________________________________________________________
   class:   BSFitter.h
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BSFitter.h,v 1.0 2006/09/19 17:13:31 yumiceva Exp $

________________________________________________________________**/


// CMS
#include "RecoVertex/BeamSpotProducer/interface/BSpdfsFcn.h"
#include "RecoVertex/BeamSpotProducer/interface/BSTrkParameters.h"
#include "RecoVertex/BeamSpotProducer/interface/BeamSpot.h"

// ROT
#include "TMath.h"
#include "Minuit2/VariableMetricMinimizer.h"

// C++ standard
#include <vector>
#include <string>

using namespace ROOT::Minuit2;

class BSFitter {
  public:

	//typedef std::vector <BSTrkParameters> BSTrkCollection;
	
	BSFitter();
	BSFitter( std::vector< BSTrkParameters > BSvector);
	
	virtual ~BSFitter();

	void SetFitType(std::string type) {
		ffit_type = type;
	}

	void SetFitVariable(std::string name) {
		ffit_variable = name;
	}

	BeamSpot Fit();
	
	BeamSpot Fit(double *inipar);
	
	// gaussian function
	//Double_t Gauss_z(Double_t z, Double_t sigma, Double_t *parms);
	// log function of gaussian		
	//void z_fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *params, Int_t iflag);
    // Fit Z distribution with a gaussian
	BeamSpot Fit_z(std::string type, double *inipar);

	BeamSpot Fit_z_chi2(double *inipar);
	BeamSpot Fit_z_likelihood(double *inipar);
	
	// Fit only d0-phi distribution with a chi2
	BeamSpot Fit_d0phi();

	BeamSpot Fit_d_likelihood(double *inipar);
	BeamSpot Fit_d_z_likelihood(double *inipar);
	BeamSpot Fit_dres_z_likelihood(double *inipar);
	
	double GetMinimum() {
		return ff_minimum;
	}
	double GetResPar0() {
		return fresolution_c0;
	}
	double GetResPar1() {
		return fresolution_c1;
	}

	BeamSpot::ResCovMatrix GetResMatrix() {
		return fres_matrix;
	}
			
  private:

	ModularFunctionMinimizer* theFitter;
	//BSzFcn* theGausszFcn;
	BSpdfsFcn* thePDF;
	
	
	std::string ffit_type;
	std::string ffit_variable;

	double ff_minimum;
	
	static const int fdim         = 7;
	
	std::string fpar_name[fdim];

	Double_t fsqrt2pi;
		
	std::vector < BSTrkParameters > fBSvector;
	
	double fresolution_c0;
	double fresolution_c1;
	BeamSpot::ResCovMatrix fres_matrix;
	
};

#endif

