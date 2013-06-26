#ifndef BeamSpotProducer_BSFitter_h
#define BeamSpotProducer_BSFitter_h

/**_________________________________________________________________
   class:   BSFitter.h
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BSFitter.h,v 1.12 2013/04/11 23:08:42 wmtan Exp $

________________________________________________________________**/


// CMS
#include "RecoVertex/BeamSpotProducer/interface/BSpdfsFcn.h"
#include "RecoVertex/BeamSpotProducer/interface/BSTrkParameters.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

// ROOT
#include "TMatrixD.h"
#include "TMath.h"
#include "Minuit2/VariableMetricMinimizer.h"
#include "TH1F.h"

// C++ standard
#include <vector>
#include <string>

class BSFitter {
  public:

	//typedef std::vector <BSTrkParameters> BSTrkCollection;
	
	BSFitter();
	BSFitter( const std::vector< BSTrkParameters > &BSvector);
	
	virtual ~BSFitter();

	void SetFitType(std::string type) {
		ffit_type = type;
	}

	void SetFitVariable(std::string name) {
		ffit_variable = name;
	}

	reco::BeamSpot Fit();
	
	reco::BeamSpot Fit(double *inipar);
		
	// Fit Z distribution with a gaussian
	reco::BeamSpot Fit_z(std::string type, double *inipar);

	reco::BeamSpot Fit_z_chi2(double *inipar);
	reco::BeamSpot Fit_z_likelihood(double *inipar);
	
	// Fit only d0-phi distribution with a chi2
	reco::BeamSpot Fit_d0phi();
	void SetMaximumZ(double z) { fMaxZ = z; }
	void SetConvergence(double val) { fconvergence = val; }
	void SetMinimumNTrks(int n) { fminNtrks = n; }
	void Setd0Cut_d0phi(double d0cut);
	void SetChi2Cut_d0phi(double chi2cut);
	void SetInputBeamWidth(double val) { finputBeamWidth = val; }
	int GetAcceptedTrks() { return ftmprow; }
	void d0phi_Init() {
		ftmprow = 0;
		ftmp.ResizeTo(4,1);
		ftmp.Zero();
		fnthite=0;
		goodfit=true;
	}
	std::vector < BSTrkParameters > GetData() { return fBSvector; }
	
	reco::BeamSpot Fit_ited0phi();
		
	reco::BeamSpot Fit_d_likelihood(double *inipar);
	reco::BeamSpot Fit_d_z_likelihood(double *inipar, double *error_par);
	reco::BeamSpot Fit_dres_z_likelihood(double *inipar);

    double scanPDF(double *init_pars,int & tracksFailed,int option);
    
	double GetMinimum() {
		return ff_minimum;
	}
	double GetResPar0() {
		return fresolution_c0;
	}
	double GetResPar1() {
		return fresolution_c1;
	}
	double GetResPar0Err() {
		return fres_c0_err;
	}
	double GetResPar1Err() {
		return fres_c1_err;
	}

	reco::BeamSpot::ResCovMatrix GetResMatrix() {
		return fres_matrix;
	}

	TH1F * GetVzHisto() { return h1z; }
	
  private:

	ROOT::Minuit2::ModularFunctionMinimizer* theFitter;
	//BSzFcn* theGausszFcn;
	BSpdfsFcn* thePDF;
	
	reco::BeamSpot::BeamType fbeamtype;
	std::string ffit_type;
	std::string ffit_variable;

	double ff_minimum;
	
	static const int fdim         = 7;
	
	std::string fpar_name[fdim];

	Double_t fsqrt2pi;
		
	std::vector < BSTrkParameters > fBSvector;
    std::vector < BSTrkParameters > fBSvectorBW;
    
	double fresolution_c0;
	double fresolution_c1;
	double fres_c0_err;
	double fres_c1_err;
	reco::BeamSpot::ResCovMatrix fres_matrix;
	//reco::BeamSpot fBSforCuts;
	TMatrixD ftmp;
	bool fapplyd0cut;
	bool fapplychi2cut;
	double fd0cut;
	double fchi2cut;
	int ftmprow;
	int fnthite;
	bool goodfit;
	double fMaxZ;
	double fconvergence;
	int fminNtrks;
	double finputBeamWidth;
	TH1F *h1z;
};

#endif

