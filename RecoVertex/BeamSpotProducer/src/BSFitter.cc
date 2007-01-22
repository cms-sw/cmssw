/**_________________________________________________________________
   class:   BSFitter.cc
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BSFitter.cc,v 1.1 2006/12/15 20:00:37 yumiceva Exp $

________________________________________________________________**/


#include "Minuit2/VariableMetricMinimizer.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnUserParameterState.h"
#include "CLHEP/config/CLHEP.h"

// C++ standard
#include <vector>
#include <cmath>

// CMS
#include "RecoVertex/BeamSpotProducer/interface/BSFitter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

// ROOT
#include "TMatrixD.h"
#include "TMatrixDSym.h"
#include "TH1.h"
#include "TF1.h"

using namespace ROOT::Minuit2;


//_____________________________________________________________________
BSFitter::BSFitter() {
}

//_____________________________________________________________________
BSFitter::BSFitter( std:: vector< BSTrkParameters > BSvector ) {

	ffit_type = "default";
	ffit_variable = "default";
	
	fBSvector = std::vector<BSTrkParameters> (BSvector);

	fsqrt2pi = sqrt(2.* TMath::Pi());
	
	fpar_name[0] = "z0        ";
	fpar_name[1] = "SigmaZ0   ";
	fpar_name[2] = "X0        ";
	fpar_name[3] = "Y0        ";
	fpar_name[4] = "dxdz      ";
	fpar_name[5] = "dydz      ";
	fpar_name[6] = "SigmaBeam ";		   

	//if (theGausszFcn == 0 ) {
	thePDF = new BSpdfsFcn();
		//std::cout << "new BSzFcn object"<<std::endl;

//}
		//if (theFitter == 0 ) {
		
		theFitter    = new VariableMetricMinimizer();
		//std::cout << "new VariableMetricMinimizer object"<<std::endl;
		//}

		//std::cout << "BSFitter:: initialized" << std::endl;
		//std::cout << "fBSvector size = " << fBSvector.size() << std::endl;
	
}

//______________________________________________________________________
BSFitter::~BSFitter()
{
	//delete fBSvector;
	delete thePDF;
	delete theFitter;
}


//______________________________________________________________________
reco::BeamSpot BSFitter::Fit() {

	return this->Fit(0);
	
}

//______________________________________________________________________
reco::BeamSpot BSFitter::Fit(double *inipar = 0) {

	if ( ffit_variable == "z" ) {
		
		if ( ffit_type == "chi2" ) {

			return Fit_z_chi2(inipar);
			
		} else if ( ffit_type == "likelihood" ) {

			return Fit_z_likelihood(inipar);
			
		} else if ( ffit_type == "combined" ) {

			reco::BeamSpot tmp_beamspot = Fit_z_chi2(inipar);
			double tmp_par[2] = {tmp_beamspot.z0(), tmp_beamspot.sigmaZ()};
			return Fit_z_likelihood(tmp_par);
			
		} else {

			throw cms::Exception("LogicError")
			<< "Error in BeamSpotProducer/BSFitter: "
			<< "Illegal fit type, options are chi2,likelihood or combined(ie. first chi2 then likelihood)";
			
		}		
	} else if ( ffit_variable == "d" ) {

		if ( ffit_type == "d0phi" ) {

			return Fit_d0phi();
			
		} else if ( ffit_type == "likelihood" ) {

			return Fit_d_likelihood(inipar);
			
		} else if ( ffit_type == "combined" ) {

			reco::BeamSpot tmp_beamspot = Fit_d0phi();
			double tmp_par[4] = {tmp_beamspot.x0(), tmp_beamspot.y0(), tmp_beamspot.dxdz(), tmp_beamspot.dydz()};
			return Fit_d_likelihood(tmp_par);
			
		} else {
			throw cms::Exception("LogicError")
				<< "Error in BeamSpotProducer/BSFitter: "
				<< "Illegal fit type, options are d0phi, likelihood or combined";
		}
	} else if ( ffit_variable == "d*z" || ffit_variable == "default" ) {

		if ( ffit_type == "likelihood" || ffit_type == "default" ) {

			reco::BeamSpot::CovarianceMatrix matrix;
			// first fit z distribution using a chi2 fit
			reco::BeamSpot tmp_z = Fit_z_chi2(inipar);
			for (int j = 2 ; j < 4 ; ++j) {
				for(int k = j ; k < 4 ; ++k) {
					matrix(j,k) = tmp_z.covariance()(j,k);
				}
			}
		
			// use d0-phi algorithm to extract transverse position
			reco::BeamSpot tmp_d0phi= Fit_d0phi();
			for (int j = 0 ; j < 2 ; ++j) {
				for(int k = j ; k < 2 ; ++k) {
					matrix(j,k) = tmp_d0phi.covariance()(j,k);
				}
			}
		
			// put everything into one object
			reco::BeamSpot spot(reco::BeamSpot::Point(tmp_d0phi.x0(), tmp_d0phi.y0(), tmp_z.z0()),
								tmp_z.sigmaZ(),
								tmp_d0phi.dxdz(),
								tmp_d0phi.dydz(),
								0.,
								matrix);


			
			//reco::BeamSpot tmp_z = Fit_z_chi2(inipar);
			
			//reco::BeamSpot tmp_d0phi = Fit_d0phi();
			// log-likelihood fit
			double tmp_par[6] = {tmp_d0phi.x0(), tmp_d0phi.y0(), tmp_z.z0(),
								 tmp_z.sigmaZ(), tmp_d0phi.dxdz(), tmp_d0phi.dydz()};
			reco::BeamSpot tmp_lh = Fit_d_z_likelihood(tmp_par);

			if ( isnan(ff_minimum) || isinf(ff_minimum) ) {

				if (ffit_type == "likelihood" ) {
					std::cout << "BSFitter: Result is non physical. Log-Likelihood fit did not converge." << std::endl;
					return tmp_lh;
				}
				
			}

			if (ffit_type == "likelihood") {
				return tmp_lh;
			} else {
				return spot;
			}
			
			
		} else if ( ffit_type == "resolution" ) {

			reco::BeamSpot tmp_z = Fit_z_chi2(inipar);
			
			reco::BeamSpot tmp_d0phi = Fit_d0phi();
			
			double tmp_par[6] = {tmp_d0phi.x0(), tmp_d0phi.y0(), tmp_z.z0(),
								 tmp_z.sigmaZ(), tmp_d0phi.dxdz(), tmp_d0phi.dydz()};

			reco::BeamSpot tmp_beam = Fit_d_z_likelihood(tmp_par);

			double tmp_par2[7] = {tmp_beam.x0(), tmp_beam.y0(), tmp_beam.z0(),
								 tmp_beam.sigmaZ(), tmp_beam.dxdz(), tmp_beam.dydz(),
								 tmp_beam.BeamWidth()};
			
			reco::BeamSpot tmp_lh = Fit_dres_z_likelihood(tmp_par2);

			if ( isnan(ff_minimum) || isinf(ff_minimum) ) {
			
				std::cout << "BSFitter: Result is non physical. Log-Likelihood fit did not converge." << std::endl;
			}
			return tmp_lh;
			
		} else {
			
			throw cms::Exception("LogicError")
				<< "Error in BeamSpotProducer/BSFitter: "
				<< "Illegal fit type, options are likelihood or resolution";
		}
	} else {

		throw cms::Exception("LogicError")
			<< "Error in BeamSpotProducer/BSFitter: "
			<< "Illegal variable type, options are \"z\", \"d\", or \"d*z\"";
	}
	
	
}

//______________________________________________________________________
reco::BeamSpot BSFitter::Fit_z_likelihood(double *inipar) {

	//std::cout << "Fit_z(double *) called" << std::endl;
	//std::cout << "inipar[0]= " << inipar[0] << std::endl;
	//std::cout << "inipar[1]= " << inipar[1] << std::endl;
	
	std::vector<double> par(2,0);
	std::vector<double> err(2,0);

	par.push_back(0.0);
	par.push_back(7.0);
	err.push_back(0.0001);
	err.push_back(0.0001);
	//par[0] = 0.0; err[0] = 0.0;
	//par[1] = 7.0; err[1] = 0.0;

	thePDF->SetPDFs("PDFGauss_z");
	thePDF->SetData(fBSvector);
	//std::cout << "data loaded"<< std::endl;
	
	//FunctionMinimum fmin = theFitter->Minimize(*theGausszFcn, par, err, 1, 500, 0.1);
	MnUserParameters upar;
	upar.Add("X0",    0.,0.);
	upar.Add("Y0",    0.,0.);
	upar.Add("Z0",    inipar[0],0.001);
	upar.Add("sigmaZ",inipar[1],0.001);
	
	MnMigrad migrad(*thePDF, upar);
	
	FunctionMinimum fmin = migrad();
	ff_minimum = fmin.Fval();
	//std::cout << " eval= " << ff_minimum
	//		  << "/n params[0]= " << fmin.Parameters().Vec()(0) << std::endl;
	
	/*
	TMinuit *gmMinuit = new TMinuit(2); 

	//gmMinuit->SetFCN(z_fcn);
	gmMinuit->SetFCN(myFitz_fcn);
	
	
	int ierflg = 0;
	double step[2] = {0.001,0.001};
	
	for (int i = 0; i<2; i++) {   
		gmMinuit->mnparm(i,fpar_name[i].c_str(),inipar[i],step[i],0,0,ierflg);
	}
	gmMinuit->Migrad();
	*/
	reco::BeamSpot::CovarianceMatrix matrix;

	for (int j = 2 ; j < 4 ; ++j) {
		for(int k = j ; k < 4 ; ++k) {
			matrix(j,k) = fmin.Error().Matrix()(j,k);
		}
	}
		
	return reco::BeamSpot( reco::BeamSpot::Point(0.,
									 0.,
									 fmin.Parameters().Vec()(2)),
					 fmin.Parameters().Vec()(3),
						   0.,
						   0.,
						   0.,
						   matrix );
}

//______________________________________________________________________
reco::BeamSpot BSFitter::Fit_z_chi2(double *inipar) {

	//std::cout << "Fit_z_chi2() called" << std::endl;

	TH1F *h1z = new TH1F("h1z","z distribution",100,-50.,50.);
	
	std::vector<BSTrkParameters>::const_iterator iparam = fBSvector.begin();
		
	for( iparam = fBSvector.begin(); iparam != fBSvector.end(); ++iparam) {
		
		 h1z->Fill(iparam->z0(), iparam->sigz0());
    }

	h1z->Fit("gaus","Q0");
	//std::cout << "fitted "<< std::endl;
	
	TF1 *fgaus = h1z->GetFunction("gaus");
	//std::cout << "got function" << std::endl;
	double fpar[2] = {fgaus->GetParameter(1), fgaus->GetParameter(2) };

	reco::BeamSpot::CovarianceMatrix matrix;
	// add matrix values.
	matrix(2,2) = fgaus->GetParError(1);
	matrix(3,3) = fgaus->GetParError(2);
	
	delete h1z;

	return reco::BeamSpot( reco::BeamSpot::Point(0.,
												 0.,
												 fpar[0]),
						   fpar[1],
						   0.,
						   0.,
						   0.,
						   matrix );	

	
}

//______________________________________________________________________
reco::BeamSpot BSFitter::Fit_d0phi() {

	//LogDebug ("BSFitter") << " we will use " << fBSvector.size() << " tracks.";
	
	TMatrixD x_result(4,1);
	TMatrixDSym V_result(4);
	
	TMatrixDSym Vint(4);
	TMatrixD b(4,1);
	
//Double_t weightsum = 0;
	
	Vint.Zero();
	b.Zero();
	
	TMatrixD g(4,1);
	TMatrixDSym temp(4);
	
	std::vector<BSTrkParameters>::const_iterator iparam = fBSvector.begin();
	int tmprow=0;

	
	//edm::LogInfo ("BSFitter") << " test";
		
	//std::cout << "BSFitter: fit" << std::endl;
	
	for( iparam = fBSvector.begin() ;
		iparam != fBSvector.end() ; ++iparam) {
		
		
		//if(i->weight2 == 0) continue;
		
		//if (tmprow==0) {
		//std::cout << "d0=" << iparam->d0() << " sigd0=" << iparam->sigd0()
		//<< " phi0="<< iparam->sigd0() << " z0=" << iparam->z0() << std::endl;
		//}
		g(0,0) = sin(iparam->phi0());
		g(1,0) = - cos(iparam->phi0());
		g(2,0) = iparam->z0() * g(0,0);
		g(3,0) = iparam->z0() * g(1,0);
		
		temp.Zero();
		for(int j = 0 ; j < 4 ; ++j) {
			for(int k = j ; k < 4 ; ++k) {
				temp(j,k) += g(j,0) * g(k,0);
			}
		}
		double sigmabeam2 = 0.002 * 0.002;

		//double sigma2 = sigmabeam2 +  (iparam->sigd0())* (iparam->sigd0()) / iparam->weight2;
		double sigma2 = sigmabeam2 +  (iparam->sigd0())* (iparam->sigd0());
		Vint += (temp * (1 / sigma2));
		b += (iparam->d0() / sigma2 * g);
		//weightsum += sqrt(i->weight2);

		tmprow++;
	}
	Double_t determinant;
	V_result = Vint.InvertFast(&determinant);
	x_result = V_result  * b;

	//LogDebug ("BSFitter") << " d0-phi fit done.";
	//std::cout<< " d0-phi fit done." << std::endl;
	
	reco::BeamSpot::CovarianceMatrix matrix;
	// first two parameters
	for (int j = 0 ; j < 2 ; ++j) {
		for(int k = j ; k < 2 ; ++k) {
			matrix(j,k) = V_result(j,k);
		}
	}
	// slope parameters
	for (int j = 4 ; j < 6 ; ++j) {
		for(int k = j ; k < 6 ; ++k) {
			matrix(j,k) = V_result(j-2,k-2);
		}
	}
	
	return reco::BeamSpot( reco::BeamSpot::Point(x_result(0,0),
												 x_result(1,0),
												 0.0),
						   0.,
						   x_result(2,0),
						   x_result(3,0),
			                           0.,
						   matrix );
	
}


//______________________________________________________________________
reco::BeamSpot BSFitter::Fit_d_likelihood(double *inipar) {
	

	thePDF->SetPDFs("PDFGauss_d");
	thePDF->SetData(fBSvector);

	MnUserParameters upar;
	upar.Add("X0",  inipar[0],0.001);
	upar.Add("Y0",  inipar[1],0.001);
	upar.Add("Z0",    0.,0.001);
	upar.Add("sigmaZ",0.,0.001);
	upar.Add("dxdz",inipar[2],0.001);
	upar.Add("dydz",inipar[3],0.001);
		
	
	MnMigrad migrad(*thePDF, upar);
	
	FunctionMinimum fmin = migrad();
	ff_minimum = fmin.Fval();

	reco::BeamSpot::CovarianceMatrix matrix;
	for (int j = 0 ; j < 6 ; ++j) {
		for(int k = j ; k < 6 ; ++k) {
			matrix(j,k) = fmin.Error().Matrix()(j,k);
		}
	}
	
	return reco::BeamSpot( reco::BeamSpot::Point(fmin.Parameters().Vec()(0),
									 fmin.Parameters().Vec()(1),
									 0.),
					 0.,
					 fmin.Parameters().Vec()(4),
					 fmin.Parameters().Vec()(5),
					 0.,
					 matrix );
}

//______________________________________________________________________
reco::BeamSpot BSFitter::Fit_d_z_likelihood(double *inipar) {

	//for ( int i =0; i<6; i++ ) {
	//	std::cout << inipar[i] << std::endl;
	//}
	
	thePDF->SetPDFs("PDFGauss_d*PDFGauss_z");
	thePDF->SetData(fBSvector);

	MnUserParameters upar;
	upar.Add("X0",  inipar[0],0.001);
	upar.Add("Y0",  inipar[1],0.001);
	upar.Add("Z0",    inipar[2],0.001);
	upar.Add("sigmaZ",inipar[3],0.001);
	upar.Add("dxdz",inipar[4],0.001);
	upar.Add("dydz",inipar[5],0.001);
	upar.Add("BeamWidth",0.0020,0.0001);
	
	MnMigrad migrad(*thePDF, upar);
	
	FunctionMinimum fmin = migrad();

	ff_minimum = fmin.Fval();
	
	std::cout << " eval= " << ff_minimum
			  << "/n params[0]= " << fmin.Parameters().Vec()(0) << std::endl;
	
	reco::BeamSpot::CovarianceMatrix matrix;

	for (int j = 0 ; j < 7 ; ++j) {
		for(int k = j ; k < 7 ; ++k) {
			matrix(j,k) = fmin.Error().Matrix()(j,k);
		}
	}
			
	
	return reco::BeamSpot( reco::BeamSpot::Point(fmin.Parameters().Vec()(0),
									 fmin.Parameters().Vec()(1),
									 fmin.Parameters().Vec()(2)),
					 fmin.Parameters().Vec()(3),
					 fmin.Parameters().Vec()(4),
					 fmin.Parameters().Vec()(5),
					 fmin.Parameters().Vec()(6),
					 matrix );
}


//______________________________________________________________________
reco::BeamSpot BSFitter::Fit_dres_z_likelihood(double *inipar) {
	
	
	thePDF->SetPDFs("PDFGauss_d_resolution*PDFGauss_z");
	thePDF->SetData(fBSvector);

	MnUserParameters upar;
	upar.Add("X0",  inipar[0],0.001);
	upar.Add("Y0",  inipar[1],0.001);
	upar.Add("Z0",    inipar[2],0.001);
	upar.Add("sigmaZ",inipar[3],0.001);
	upar.Add("dxdz",inipar[4],0.001);
	upar.Add("dydz",inipar[5],0.001);
	upar.Add("BeamWidth",inipar[6],0.0001);
	upar.Add("c0",0.0010,0.0001);
	upar.Add("c1",0.0090,0.0001);

	// fix beam width
	upar.Fix("BeamWidth");
	// number of parameters in fit are 9-1 = 8
	
	MnMigrad migrad(*thePDF, upar);
		
	FunctionMinimum fmin = migrad();
	ff_minimum = fmin.Fval();
	
	reco::BeamSpot::CovarianceMatrix matrix;

	for (int j = 0 ; j < 6 ; ++j) {
		for(int k = j ; k < 6 ; ++k) {
			matrix(j,k) = fmin.Error().Matrix()(j,k);
		}
	}
	//std::cout << " fill resolution values" << std::endl;
	//std::cout << " matrix size= " << fmin.Error().Matrix().size() << std::endl;
	//std::cout << " vec(6)="<< fmin.Parameters().Vec()(6) << std::endl;
	//std::cout << " vec(7)="<< fmin.Parameters().Vec()(7) << std::endl;
	//std::cout << " vec(8)="<< fmin.Parameters().Vec()(8) << std::endl;
	
	fresolution_c0 = fmin.Parameters().Vec()(6);
	fresolution_c1 = fmin.Parameters().Vec()(7);
	fres_c0_err = sqrt( fmin.Error().Matrix()(6,6) );
	fres_c1_err = sqrt( fmin.Error().Matrix()(7,7) );
	
	for (int j = 6 ; j < 8 ; ++j) {
		for(int k = 6 ; k < 8 ; ++k) {
			fres_matrix(j-7,k-7) = fmin.Error().Matrix()(j,k);
		}
	}
	
	return reco::BeamSpot( reco::BeamSpot::Point(fmin.Parameters().Vec()(0),
									 fmin.Parameters().Vec()(1),
									 fmin.Parameters().Vec()(2)),
					 fmin.Parameters().Vec()(3),
					 fmin.Parameters().Vec()(4),
					 fmin.Parameters().Vec()(5),
					 inipar[6],
					 matrix );
}



