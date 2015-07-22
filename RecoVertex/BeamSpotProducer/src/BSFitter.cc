/**_________________________________________________________________
   class:   BSFitter.cc
   package: RecoVertex/BeamSpotProducer



 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)



________________________________________________________________**/


#include "Minuit2/VariableMetricMinimizer.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnUserParameterState.h"
//#include "CLHEP/config/CLHEP.h"

// C++ standard
#include <vector>
#include <cmath>

// CMS
#include "RecoVertex/BeamSpotProducer/interface/BSFitter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/isFinite.h"

// ROOT
#include "TMatrixD.h"
#include "TMatrixDSym.h"
#include "TDecompBK.h"
#include "TH1.h"
#include "TF1.h"

using namespace ROOT::Minuit2;


//_____________________________________________________________________
BSFitter::BSFitter() {
	fbeamtype = reco::BeamSpot::Unknown;
}

//_____________________________________________________________________
BSFitter::BSFitter( const std:: vector< BSTrkParameters > &BSvector ) {

	ffit_type = "default";
	ffit_variable = "default";

	fBSvector = BSvector;

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


//}
		//if (theFitter == 0 ) {

	theFitter    = new VariableMetricMinimizer();

		//}

	fapplyd0cut = false;
	fapplychi2cut = false;
	ftmprow = 0;
	ftmp.ResizeTo(4,1);
	ftmp.Zero();
	fnthite=0;
	fMaxZ = 50.; //cm
	fconvergence = 0.5; // stop fit when 50% of the input collection has been removed.
	fminNtrks = 100;
	finputBeamWidth = -1; // no input

    h1z = new TH1F("h1z","z distribution",200,-fMaxZ, fMaxZ);

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
	fbeamtype = reco::BeamSpot::Unknown;
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
			this->d0phi_Init();
			return Fit_d0phi();

		} else if ( ffit_type == "likelihood" ) {

			return Fit_d_likelihood(inipar);

		} else if ( ffit_type == "combined" ) {

			this->d0phi_Init();
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
            // we are now fitting Z inside d0phi fitter
			// first fit z distribution using a chi2 fit
			//reco::BeamSpot tmp_z = Fit_z_chi2(inipar);
			//for (int j = 2 ; j < 4 ; ++j) {
            //for(int k = j ; k < 4 ; ++k) {
            //	matrix(j,k) = tmp_z.covariance()(j,k);
            //}
			//}

			// use d0-phi algorithm to extract transverse position
			this->d0phi_Init();
			//reco::BeamSpot tmp_d0phi= Fit_d0phi(); // change to iterative procedure:
			this->Setd0Cut_d0phi(4.0);
			reco::BeamSpot tmp_d0phi= Fit_ited0phi();

			//for (int j = 0 ; j < 2 ; ++j) {
			//	for(int k = j ; k < 2 ; ++k) {
			//		matrix(j,k) = tmp_d0phi.covariance()(j,k);
            //}
			//}
			// slopes
			//for (int j = 4 ; j < 6 ; ++j) {
            // for(int k = j ; k < 6 ; ++k) {
            //  matrix(j,k) = tmp_d0phi.covariance()(j,k);
			//  }
			//}


			// put everything into one object
			reco::BeamSpot spot(reco::BeamSpot::Point(tmp_d0phi.x0(), tmp_d0phi.y0(), tmp_d0phi.z0()),
								tmp_d0phi.sigmaZ(),
								tmp_d0phi.dxdz(),
								tmp_d0phi.dydz(),
								0.,
								tmp_d0phi.covariance(),
								fbeamtype );



			//reco::BeamSpot tmp_z = Fit_z_chi2(inipar);

			//reco::BeamSpot tmp_d0phi = Fit_d0phi();

            // log-likelihood fit
			if (ffit_type == "likelihood") {
                double tmp_par[7] = {tmp_d0phi.x0(), tmp_d0phi.y0(), tmp_d0phi.z0(),
                                     tmp_d0phi.sigmaZ(), tmp_d0phi.dxdz(), tmp_d0phi.dydz(),0.0};

                double tmp_error_par[7];
                for(int s=0;s<6;s++){ tmp_error_par[s] = pow( tmp_d0phi.covariance()(s,s),0.5);}
                tmp_error_par[6]=0.0;

                reco::BeamSpot tmp_lh = Fit_d_z_likelihood(tmp_par,tmp_error_par);

                if (edm::isNotFinite(ff_minimum)) {
                    edm::LogWarning("BSFitter") << "BSFitter: Result is non physical. Log-Likelihood fit to extract beam width did not converge." << std::endl;
                    tmp_lh.setType(reco::BeamSpot::Unknown);
                    return tmp_lh;
                }
                return tmp_lh;

			} else {

                edm::LogInfo("BSFitter") << "default track-based fit does not extract beam width." << std::endl;
				return spot;
            }


		} else if ( ffit_type == "resolution" ) {

			reco::BeamSpot tmp_z = Fit_z_chi2(inipar);
			this->d0phi_Init();
			reco::BeamSpot tmp_d0phi = Fit_d0phi();

			double tmp_par[7] = {tmp_d0phi.x0(), tmp_d0phi.y0(), tmp_z.z0(),
								 tmp_z.sigmaZ(), tmp_d0phi.dxdz(), tmp_d0phi.dydz(),0.0};
            double tmp_error_par[7];
            for(int s=0;s<6;s++){ tmp_error_par[s] = pow(tmp_par[s],0.5);}
            tmp_error_par[6]=0.0;

			reco::BeamSpot tmp_beam = Fit_d_z_likelihood(tmp_par,tmp_error_par);

			double tmp_par2[7] = {tmp_beam.x0(), tmp_beam.y0(), tmp_beam.z0(),
								 tmp_beam.sigmaZ(), tmp_beam.dxdz(), tmp_beam.dydz(),
								 tmp_beam.BeamWidthX()};

			reco::BeamSpot tmp_lh = Fit_dres_z_likelihood(tmp_par2);

			if (edm::isNotFinite(ff_minimum)) {

                edm::LogWarning("BSFitter") << "Result is non physical. Log-Likelihood fit did not converge." << std::endl;
				tmp_lh.setType(reco::BeamSpot::Unknown);
				return tmp_lh;
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
			       matrix,
			       fbeamtype );
}

//______________________________________________________________________
reco::BeamSpot BSFitter::Fit_z_chi2(double *inipar) {

    // N.B. this fit is not performed anymore but now
    // Z is fitted in the same track set used in the d0-phi fit after
    // each iteration


	//std::cout << "Fit_z_chi2() called" << std::endl;
        // FIXME: include whole tracker z length for the time being
        // ==> add protection and z0 cut
	h1z = new TH1F("h1z","z distribution",200,-fMaxZ, fMaxZ);

	std::vector<BSTrkParameters>::const_iterator iparam = fBSvector.begin();

	// HERE check size of track vector

	for( iparam = fBSvector.begin(); iparam != fBSvector.end(); ++iparam) {

		 h1z->Fill( iparam->z0() );
		 //std::cout<<"z0="<<iparam->z0()<<"; sigZ0="<<iparam->sigz0()<<std::endl;
	}

	//Use our own copy for thread safety
	TF1 fgaus("fgaus","gaus");
	h1z->Fit(&fgaus,"QLM0");
	//std::cout << "fitted "<< std::endl;

	//std::cout << "got function" << std::endl;
	double fpar[2] = {fgaus.GetParameter(1), fgaus.GetParameter(2) };
	//std::cout<<"Debug fpar[2] = (" <<fpar[0]<<","<<fpar[1]<<")"<<std::endl;
	reco::BeamSpot::CovarianceMatrix matrix;
	// add matrix values.
	matrix(2,2) = fgaus.GetParError(1) * fgaus.GetParError(1);
	matrix(3,3) = fgaus.GetParError(2) * fgaus.GetParError(2);

	//delete h1z;

	return reco::BeamSpot( reco::BeamSpot::Point(0.,
						     0.,
						     fpar[0]),
			       fpar[1],
			       0.,
			       0.,
			       0.,
			       matrix,
			       fbeamtype );


}

//______________________________________________________________________
reco::BeamSpot BSFitter::Fit_ited0phi() {

	this->d0phi_Init();
    edm::LogInfo("BSFitter") << "number of total input tracks: " << fBSvector.size() << std::endl;

	reco::BeamSpot theanswer;

	if ( (int)fBSvector.size() <= fminNtrks ) {
        edm::LogWarning("BSFitter") << "need at least " << fminNtrks << " tracks to run beamline fitter." << std::endl;
		fbeamtype = reco::BeamSpot::Fake;
		theanswer.setType(fbeamtype);
		return theanswer;
	}

	theanswer = Fit_d0phi(); //get initial ftmp and ftmprow
	if ( goodfit ) fnthite++;
	//std::cout << "Initial tempanswer (iteration 0): " << theanswer << std::endl;

	reco::BeamSpot preanswer = theanswer;

	while ( goodfit &&
			ftmprow > fconvergence * fBSvector.size() &&
			ftmprow > fminNtrks  ) {

		theanswer = Fit_d0phi();
		fd0cut /= 1.5;
		fchi2cut /= 1.5;
		if ( goodfit &&
			ftmprow > fconvergence * fBSvector.size() &&
			ftmprow > fminNtrks ) {
			preanswer = theanswer;
			//std::cout << "Iteration " << fnthite << ": " << preanswer << std::endl;
			fnthite++;
		}
	}
	// FIXME: return fit results from previous iteration for both bad fit and for >50% tracks thrown away
	//std::cout << "The last iteration, theanswer: " << theanswer << std::endl;
	theanswer = preanswer;
	//std::cout << "Use previous results from iteration #" << ( fnthite > 0 ? fnthite-1 : 0 ) << std::endl;
	//if ( fnthite > 1 ) std::cout << theanswer << std::endl;

    edm::LogInfo("BSFitter") << "Total number of successful iterations = " << ( goodfit ? (fnthite+1) : fnthite ) << std::endl;
    if (goodfit) {
        fbeamtype = reco::BeamSpot::Tracker;
        theanswer.setType(fbeamtype);
    }
    else {
        edm::LogWarning("BSFitter") << "Fit doesn't converge!!!" << std::endl;
        fbeamtype = reco::BeamSpot::Unknown;
        theanswer.setType(fbeamtype);
    }
	return theanswer;
}


//______________________________________________________________________
reco::BeamSpot BSFitter::Fit_d0phi() {

	//LogDebug ("BSFitter") << " we will use " << fBSvector.size() << " tracks.";
    if (fnthite > 0) edm::LogInfo("BSFitter") << " number of tracks used: " << ftmprow << std::endl;
	//std::cout << " ftmp = matrix("<<ftmp.GetNrows()<<","<<ftmp.GetNcols()<<")"<<std::endl;
	//std::cout << " ftmp(0,0)="<<ftmp(0,0)<<std::endl;
	//std::cout << " ftmp(1,0)="<<ftmp(1,0)<<std::endl;
	//std::cout << " ftmp(2,0)="<<ftmp(2,0)<<std::endl;
	//std::cout << " ftmp(3,0)="<<ftmp(3,0)<<std::endl;

        h1z->Reset();


	TMatrixD x_result(4,1);
	TMatrixDSym V_result(4);

	TMatrixDSym Vint(4);
	TMatrixD b(4,1);

	//Double_t weightsum = 0;

	Vint.Zero();
	b.Zero();

	TMatrixD g(4,1);
	TMatrixDSym temp(4);

	std::vector<BSTrkParameters>::iterator iparam = fBSvector.begin();
	ftmprow=0;


	//edm::LogInfo ("BSFitter") << " test";

	//std::cout << "BSFitter: fit" << std::endl;

	for( iparam = fBSvector.begin() ;
		iparam != fBSvector.end() ; ++iparam) {


		//if(i->weight2 == 0) continue;

		//if (ftmprow==0) {
		//std::cout << "d0=" << iparam->d0() << " sigd0=" << iparam->sigd0()
		//<< " phi0="<< iparam->phi0() << " z0=" << iparam->z0() << std::endl;
		//std::cout << "d0phi_d0=" << iparam->d0phi_d0() << " d0phi_chi2="<<iparam->d0phi_chi2() << std::endl;
		//}
		g(0,0) = sin(iparam->phi0());
		g(1,0) = -cos(iparam->phi0());
		g(2,0) = iparam->z0() * g(0,0);
		g(3,0) = iparam->z0() * g(1,0);


		// average transverse beam width
		double sigmabeam2 = 0.006 * 0.006;
		if (finputBeamWidth > 0 ) sigmabeam2 = finputBeamWidth * finputBeamWidth;
        else {
	      //edm::LogWarning("BSFitter") << "using in fit beam width = " << sqrt(sigmabeam2) << std::endl;
	     }

		//double sigma2 = sigmabeam2 +  (iparam->sigd0())* (iparam->sigd0()) / iparam->weight2;
		// this should be 2*sigmabeam2?
		double sigma2 = sigmabeam2 +  (iparam->sigd0())* (iparam->sigd0());

		TMatrixD ftmptrans(1,4);
		ftmptrans = ftmptrans.Transpose(ftmp);
		TMatrixD dcor = ftmptrans * g;
		double chi2tmp = (iparam->d0() - dcor(0,0)) * (iparam->d0() - dcor(0,0))/sigma2;
		(*iparam) = BSTrkParameters(iparam->z0(),iparam->sigz0(),iparam->d0(),iparam->sigd0(),
					    iparam->phi0(), iparam->pt(),dcor(0,0),chi2tmp);

		bool pass = true;
		if (fapplyd0cut && fnthite>0 ) {
	       		if ( std::abs(iparam->d0() - dcor(0,0)) > fd0cut ) pass = false;

		}
		if (fapplychi2cut && fnthite>0 ) {
			if ( chi2tmp > fchi2cut ) pass = false;

		}

		if (pass) {
			temp.Zero();
			for(int j = 0 ; j < 4 ; ++j) {
				for(int k = j ; k < 4 ; ++k) {
					temp(j,k) += g(j,0) * g(k,0);
				}
			}


			Vint += (temp * (1 / sigma2));
			b += (iparam->d0() / sigma2 * g);
			//weightsum += sqrt(i->weight2);
			ftmprow++;
            h1z->Fill( iparam->z0() );
		}


	}
	Double_t determinant;
	TDecompBK bk(Vint);
	bk.SetTol(1e-11); //FIXME: find a better way to solve x_result
	if (!bk.Decompose()) {
	  goodfit = false;
      edm::LogWarning("BSFitter")
          << "Decomposition failed, matrix singular ?" << std::endl
          << "condition number = " << bk.Condition() << std::endl;
	}
	else {
	  V_result = Vint.InvertFast(&determinant);
	  x_result = V_result  * b;
	}
	//     	for(int j = 0 ; j < 4 ; ++j) {
	// 	  for(int k = 0 ; k < 4 ; ++k) {
	// 	    std::cout<<"V_result("<<j<<","<<k<<")="<<V_result(j,k)<<std::endl;
	// 	  }
	//     	}
	//         for (int j=0;j<4;++j){
	// 	  std::cout<<"x_result("<<j<<",0)="<<x_result(j,0)<<std::endl;
	// 	}
	//LogDebug ("BSFitter") << " d0-phi fit done.";
	//std::cout<< " d0-phi fit done." << std::endl;

	//Use our own copy for thread safety
	TF1 fgaus("fgaus","gaus");
	//returns 0 if OK
	//auto status = h1z->Fit(&fgaus,"QLM0","",h1z->GetMean() -2.*h1z->GetRMS(),h1z->GetMean() +2.*h1z->GetRMS());
	auto status = h1z->Fit(&fgaus,"QL0","",h1z->GetMean() -2.*h1z->GetRMS(),h1z->GetMean() +2.*h1z->GetRMS());

	//std::cout << "fitted "<< std::endl;

	//std::cout << "got function" << std::endl;
	if (status){
	  //edm::LogError("NoBeamSpotFit")<<"gaussian fit failed. no BS d0 fit";

	  return reco::BeamSpot();
	}
	double fpar[2] = {fgaus.GetParameter(1), fgaus.GetParameter(2) };

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

    // Z0 and sigmaZ
	matrix(2,2) = fgaus.GetParError(1) * fgaus.GetParError(1);
	matrix(3,3) = fgaus.GetParError(2) * fgaus.GetParError(2);

	ftmp = x_result;

	// x0 and y0 are *not* x,y at z=0, but actually at z=0
        // to correct for this, we need to translate them to z=z0
        // using the measured slopes
	//
	double x0tmp = x_result(0,0);
	double y0tmp = x_result(1,0);

	x0tmp += x_result(2,0)*fpar[0];
	y0tmp += x_result(3,0)*fpar[0];


	return reco::BeamSpot( reco::BeamSpot::Point(x0tmp,
						     y0tmp,
						     fpar[0]),
                           fpar[1],
			       x_result(2,0),
			       x_result(3,0),
			       0.,
			       matrix,
			       fbeamtype );

}


//______________________________________________________________________
void BSFitter::Setd0Cut_d0phi(double d0cut) {

	fapplyd0cut = true;

	//fBSforCuts = BSfitted;
	fd0cut = d0cut;
}

//______________________________________________________________________
void BSFitter::SetChi2Cut_d0phi(double chi2cut) {

	fapplychi2cut = true;

	//fBSforCuts = BSfitted;
	fchi2cut = chi2cut;
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
			       matrix,
			       fbeamtype );
}
//______________________________________________________________________
double BSFitter::scanPDF(double *init_pars, int & tracksfixed, int option){

   if(option==1)init_pars[6]=0.0005;  //starting value for any given configuration

   //local vairables with initial values
   double fsqrt2pi=0.0;
   double d_sig=0.0;
   double d_dprime=0.0;
   double d_result=0.0;
   double z_sig=0.0;
   double z_result=0.0;
   double function=0.0;
   double tot_pdf=0.0;
   double last_minvalue=1.0e+10;
   double init_bw=-99.99;
   int iters=0;

  //used to remove tracks if far away from bs by this
   double DeltadCut=0.1000;
   if(init_pars[6]<0.0200){DeltadCut=0.0900; } //worked for high 2.36TeV
   if(init_pars[6]<0.0100){DeltadCut=0.0700;}  //just a guesss for 7 TeV but one should scan for actual values


std::vector<BSTrkParameters>::const_iterator iparam = fBSvector.begin();


if(option==1)iters=500;
if(option==2)iters=1;

for(int p=0;p<iters;p++){

   if(iters==500)init_pars[6]+=0.0002;
    tracksfixed=0;

for( iparam = fBSvector.begin(); iparam != fBSvector.end(); ++iparam)
       {
                    fsqrt2pi = sqrt(2.* TMath::Pi());
                    d_sig = sqrt(init_pars[6]*init_pars[6] + (iparam->sigd0())*(iparam->sigd0()));
                    d_dprime = iparam->d0() - (   (  (init_pars[0] + iparam->z0()*(init_pars[4]))*sin(iparam->phi0()) )
                                               - (  (init_pars[1] + iparam->z0()*(init_pars[5]))*cos(iparam->phi0()) ) );

                    //***Remove tracks before the fit which gives low pdf values to blow up the pdf
                    if(std::abs(d_dprime)<DeltadCut && option==2){ fBSvectorBW.push_back(*iparam);}

                    d_result = (exp(-(d_dprime*d_dprime)/(2.0*d_sig*d_sig)))/(d_sig*fsqrt2pi);
                    z_sig = sqrt(iparam->sigz0() * iparam->sigz0() + init_pars[3]*init_pars[3]);
                    z_result = (exp(-((iparam->z0() - init_pars[2])*(iparam->z0() - init_pars[2]))/(2.0*z_sig*z_sig)))/(z_sig*fsqrt2pi);
                    tot_pdf=z_result*d_result;

                    //for those trcks which gives problems due to very tiny pdf_d values.
                    //Update: This protection will NOT be used with the dprime cut above but still kept here to get
                    // the intial value of beam width reasonably
                    //A warning will appear if there were any tracks with < 10^-5 for pdf_d so that (d-dprime) cut can be lowered
                    if(d_result < 1.0e-05){ tot_pdf=z_result*1.0e-05;
                                           //if(option==2)std::cout<<"last Iter  d-d'   =  "<<(std::abs(d_dprime))<<std::endl;
                                           tracksfixed++; }

                       function = function + log(tot_pdf);


       }//loop over tracks


       function= -2.0*function;
       if(function<last_minvalue){init_bw=init_pars[6];
                                  last_minvalue=function; }
       function=0.0;
   }//loop over beam width

   if(init_bw>0) {
    init_bw=init_bw+(0.20*init_bw); //start with 20 % more

   }
   else{

       if(option==1){
           edm::LogWarning("BSFitter")
               <<"scanPDF:====>>>> WARNING***: The initial guess value of Beam width is negative!!!!!!"<<std::endl
               <<"scanPDF:====>>>> Assigning beam width a starting value of "<<init_bw<<"  cm"<<std::endl;
           init_bw=0.0200;

       }
      }


    return init_bw;

}

//________________________________________________________________________________
reco::BeamSpot BSFitter::Fit_d_z_likelihood(double *inipar, double *error_par) {

      int tracksFailed=0;

      //estimate first guess of beam width and tame 20% extra of it to start
      inipar[6]=scanPDF(inipar,tracksFailed,1);
      error_par[6]=(inipar[6])*0.20;


     //Here remove the tracks which give low pdf and fill into a new vector
     //std::cout<<"Size of Old vector = "<<(fBSvector.size())<<std::endl;
     /* double junk= */ scanPDF(inipar,tracksFailed,2);
     //std::cout<<"Size of New vector = "<<(fBSvectorBW.size())<<std::endl;

     //Refill the fBSVector again with new sets of tracks
     fBSvector.clear();
     std::vector<BSTrkParameters>::const_iterator iparamBW = fBSvectorBW.begin();
     for( iparamBW = fBSvectorBW.begin(); iparamBW != fBSvectorBW.end(); ++iparamBW)
        {          fBSvector.push_back(*iparamBW);
        }


        thePDF->SetPDFs("PDFGauss_d*PDFGauss_z");
        thePDF->SetData(fBSvector);
        MnUserParameters upar;

        upar.Add("X0",  inipar[0],error_par[0]);
        upar.Add("Y0",  inipar[1],error_par[1]);
        upar.Add("Z0",    inipar[2],error_par[2]);
        upar.Add("sigmaZ",inipar[3],error_par[3]);
        upar.Add("dxdz",inipar[4],error_par[4]);
        upar.Add("dydz",inipar[5],error_par[5]);
        upar.Add("BeamWidthX",inipar[6],error_par[6]);


        MnMigrad migrad(*thePDF, upar);

        FunctionMinimum fmin = migrad();

      // std::cout<<"-----how the fit evoves------"<<std::endl;
      // std::cout<<fmin<<std::endl;

        ff_minimum = fmin.Fval();


        bool ff_nfcn=fmin.HasReachedCallLimit();
        bool ff_cov=fmin.HasCovariance();
        bool testing=fmin.IsValid();


        //Print WARNINGS if minimum did not converged
        if( ! testing )
        {
            edm::LogWarning("BSFitter") <<"===========>>>>>** WARNING: MINUIT DID NOT CONVERGES PROPERLY !!!!!!"<<std::endl;
            if(ff_nfcn) edm::LogWarning("BSFitter") <<"===========>>>>>** WARNING: No. of Calls Exhausted"<<std::endl;
            if(!ff_cov) edm::LogWarning("BSFitter") <<"===========>>>>>** WARNING: Covariance did not found"<<std::endl;
        }

        edm::LogInfo("BSFitter") <<"The Total # Tracks used for beam width fit = "<<(fBSvectorBW.size())<<std::endl;


    //Checks after fit is performed
    double lastIter_pars[7];

   for(int ip=0;ip<7;ip++){ lastIter_pars[ip]=fmin.Parameters().Vec()(ip);
                           }



    tracksFailed=0;
    /* double lastIter_scan= */ scanPDF(lastIter_pars,tracksFailed,2);


    edm::LogWarning("BSFitter") <<"WARNING: # of tracks which have very low pdf value (pdf_d < 1.0e-05) are  = "<<tracksFailed<<std::endl;



        //std::cout << " eval= " << ff_minimum
        //                << "/n params[0]= " << fmin.Parameters().Vec()(0) << std::endl;

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

                               matrix,
                               fbeamtype );
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
	upar.Add("BeamWidthX",inipar[6],0.0001);
	upar.Add("c0",0.0010,0.0001);
	upar.Add("c1",0.0090,0.0001);

	// fix beam width
	upar.Fix("BeamWidthX");
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

	fresolution_c0 = fmin.Parameters().Vec()(6);
	fresolution_c1 = fmin.Parameters().Vec()(7);
	fres_c0_err = sqrt( fmin.Error().Matrix()(6,6) );
	fres_c1_err = sqrt( fmin.Error().Matrix()(7,7) );

	for (int j = 6 ; j < 8 ; ++j) {
		for(int k = 6 ; k < 8 ; ++k) {
			fres_matrix(j-6,k-6) = fmin.Error().Matrix()(j,k);
		}
	}

	return reco::BeamSpot( reco::BeamSpot::Point(fmin.Parameters().Vec()(0),
									 fmin.Parameters().Vec()(1),
									 fmin.Parameters().Vec()(2)),
					 fmin.Parameters().Vec()(3),
					 fmin.Parameters().Vec()(4),
					 fmin.Parameters().Vec()(5),
					 inipar[6],
					 matrix,
					 fbeamtype );
}



