#include <iostream>
#include <stdlib.h>
#include <TROOT.h>
#include <TFile.h>
#include <TRint.h>
#include <TH1.h>
#include <TF1.h>
#include <TF2.h>
#include <TH2.h>
#include <TMinuit.h>
#include <TSystem.h>
#include <TGraph.h>
#include "TMatrixD.h"
#include "TMatrixDSym.h"
#include "NtupleHelper.h"
#include "global.h"
//#include "beamfit_fcts.h"


//const char par_name[dim][20]={"z0  ","sigma ","emmitance","beta*"};
const char par_name[dim][20]={"z0  ","sigma ","x0 ", "y0 ", "dxdz ", "dydz ", "sigmaBeam ",
							  "c0  ","c1    ","emittance ","betastar "};

Double_t params[dim],errparams[dim];
Double_t sfpar[dim],errsfpar[dim]; 

static Double_t step[dim] = {1.e-5,1.e-5,1.e-3,1.e-3,1.e-3,1.e-3,1.e-5,1.e-5,1.e-5,1.e-5,1.e-5};
zData zdata;//!


Double_t zdis(Double_t z, Double_t sigma, Double_t *parms)
{ 
  //---------------------------------------------------------------------------
  //  This is a simple function to parameterize the z-vertex distribution. This 
  // is parametrized by a simple normalized gaussian distribution. 
  //---------------------------------------------------------------------------  
		
	Double_t sig = sqrt(sigma*sigma + parms[Par_Sigma]*parms[Par_Sigma]);	
	Double_t result = (exp(-((z-parms[Par_Z0])*(z-parms[Par_Z0]))/(2.0*sig*sig)))/(sig*sqrt2pi);
	return result;
} 


void zfcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *params, Int_t iflag)
{
  //----------------------------------------------------------------------------------
  // this is the function used by minuit to do the unbinned fit to the z distribution
  //----------------------------------------------------------------------------------
  f = 0.0;
  for ( zDataConstIter iter = zdata.begin(); iter != zdata.end(); ++iter) 
  {  
      f = log(zdis(iter->Z,iter->SigZ,params))+f;
    }
  f= -2.0*f;
  return ;
}

float betaf( float betastar,  float emmitance,float z,float z0)
{
  float x = sqrt(emmitance*(betastar+(((z-z0)*(z-z0))/betastar)));
  return x;
}

Double_t ddis(Double_t z, Double_t sigma,Double_t d, Double_t sigmad, Double_t *parms)
{ 
  //---------------------------------------------------------------------------
  // This is a simple function to parameterize the sigma of the beam at a given z.  
  // This is parametrized by a simple normalized gaussian distribution. 
  //---------------------------------------------------------------------------  
  Double_t sig = betaf( parms[Par_beta],parms[Par_eps],z,parms[Par_Z0]);
  sig          = sqrt(sig*sig+sigmad*sigmad);
   Double_t result = (exp(-(d*d)/(2.0*sig*sig)))/(sig*sqrt2pi);
  return result;
}

Double_t allddis(Double_t z,Double_t d, Double_t sigmad,
				 Double_t phi0, Double_t *parms)
{ 
  //---------------------------------------------------------------------------
  // This is a simple function to parameterize the beam parameters
  // This is parametrized by a simple normalized gaussian distribution. 
  //---------------------------------------------------------------------------  
	Double_t sig = sqrt( parms[Par_Sigbeam]*parms[Par_Sigbeam] + 10*sigmad*sigmad  ); //9 //2.5 factor for full simu data
	Double_t dprime = d - (parms[Par_x0] + z*parms[Par_dxdz])*sin(phi0) +
		(parms[Par_y0] + z*parms[Par_dydz])*cos(phi0);
   Double_t result = (exp(-(dprime*dprime)/(2.0*sig*sig)))/(sig*sqrt2pi);
  return result;
}

Double_t allddisBeta(Double_t z, Double_t sigma,Double_t d, Double_t sigmad,
				 Double_t phi0, Double_t *parms)
{ 
  //---------------------------------------------------------------------------
  // This is a simple function to parameterize the beam parameters
  // This is parametrized by a simple normalized gaussian distribution. 
  //---------------------------------------------------------------------------
	Double_t sigmabeam = sqrt(parms[Par_eps]*(parms[Par_beta]+(((z-parms[Par_Z0])*(z-parms[Par_Z0]))/parms[Par_beta])));
	
	Double_t sig = sqrt( sigmabeam*sigmabeam + sigmad*sigmad  );
	
	Double_t dprime = d - (parms[Par_x0] + z*parms[Par_dxdz])*sin(phi0) +
		(parms[Par_y0] + z*parms[Par_dydz])*cos(phi0);
   Double_t result = (exp(-(dprime*dprime)/(2.0*sig*sig)))/(sig*sqrt2pi);
  return result;
}



Double_t allddis2(Double_t z, Double_t sigma,Double_t d, Double_t pt,
				  Double_t phi0, Double_t *parms)
{ 
  //---------------------------------------------------------------------------
  //
  //---------------------------------------------------------------------------
	Double_t sigmad = parms[Par_c0] + parms[Par_c1]/pt;
	Double_t sig = sqrt( parms[Par_Sigbeam]*parms[Par_Sigbeam] + sigmad*sigmad  );
	Double_t dprime = d - (parms[Par_x0] + z*parms[Par_dxdz])*sin(phi0) + (parms[Par_y0] + z*parms[Par_dydz])*cos(phi0);
   Double_t result = (exp(-(dprime*dprime)/(2.0*sig*sig)))/(sig*sqrt2pi);
  return result;
}


void dfcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *params, Int_t iflag)
{
  //----------------------------------------------------------------------------------
  // this is the function used by minuit to do the unbinned fit to the IP distribution 
  //----------------------------------------------------------------------------------
  f = 0.0;
  for ( zDataConstIter iter = zdata.begin(); iter != zdata.end(); ++iter) 
    {
		//if(iter->weight2 == 0) continue;
		
      f = log(allddis(iter->Z,iter->D,iter->SigD,iter->Phi,params))+f;
    }
  f= -2.0*f;
  return ;
}

void dfcnbeta(Int_t &npar, Double_t *gin, Double_t &f, Double_t *params, Int_t iflag)
{
  //----------------------------------------------------------------------------------
  // this is the function used by minuit to do the unbinned fit to the IP distribution 
  //----------------------------------------------------------------------------------
  f = 0.0;
  for ( zDataConstIter iter = zdata.begin(); iter != zdata.end(); ++iter) 
    {
      f = log(allddisBeta(iter->Z,iter->SigZ,iter->D,iter->SigD,iter->Phi,params))+f;
    }
  f= -2.0*f;
  return ;
}

void dfcn2(Int_t &npar, Double_t *gin, Double_t &f, Double_t *params, Int_t iflag)
{
  //----------------------------------------------------------------------------------
  // this is the function used by minuit to do the unbinned fit to the IP distribution 
  //----------------------------------------------------------------------------------
  f = 0.0;
  for ( zDataConstIter iter = zdata.begin(); iter != zdata.end(); ++iter) 
    {
      f = log(allddis2(iter->Z,iter->SigZ,iter->D,iter->Pt,iter->Phi,params)*zdis(iter->Z,iter->SigZ,params))+f;
    }
  f= -2.0*f;
  return ;
}

void cfcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *params, Int_t iflag)
{
  //-----------------------------------------------------------------------------------
  // this is the function used by minuit to do the unbinned finned combined in z an IP.
  //-----------------------------------------------------------------------------------
  f = 0.0;
  for ( zDataConstIter iter = zdata.begin(); iter != zdata.end(); ++iter) 
    {
      f = log(allddis(iter->Z,iter->D,iter->SigD,iter->Phi,params)*zdis(iter->Z,iter->SigZ,params))+f;
    }
  f= -2.0*f;
  return ;
}

void fit(TMatrixD &x, TMatrixDSym &V)
{
	TMatrixDSym Vint(4);
	TMatrixD b(4,1);
	Double_t weightsum = 0;

	Vint.Zero();
	b.Zero();
	TMatrixD g(4,1);
	TMatrixDSym temp(4);
	for(zDataConstIter i = zdata.begin() ; i != zdata.end() ; ++i) {
		//std::cout << "weight  " << sqrt(i->weight2) << "\n";
		if(i->weight2 == 0) continue;

		g(0,0) = sin(i->Phi);
		g(1,0) = -cos(i->Phi);
		g(2,0) = i->Z * g(0,0);
		g(3,0) = i->Z * g(1,0);
    
		temp.Zero();
		for(int j = 0 ; j < 4 ; ++j) {
			for(int k = j ; k < 4 ; ++k) {
				temp(j,k) += g(j,0) * g(k,0);
			}
		}
		double sigmabeam2 = 0.002 * 0.002;
		double sigma2 = sigmabeam2 +  (i->SigD)* (i->SigD) / i->weight2;
		Vint += (temp * (1 / sigma2));
		b += ((i->D / sigma2) * g);
		weightsum += sqrt(i->weight2);
	}
	Double_t determinant;
	V = Vint.InvertFast(&determinant);
	x = V  * b;
	std::cout << "Sum of all weights:" << weightsum << "\n";
	std::cout << "x0                :" << x(0,0)    << " +- " << sqrt(V(0,0)) << " cm \n"; 
	std::cout << "y0                :" << x(1,0)    << " +- " << sqrt(V(1,1)) << " cm \n"; 
	std::cout << "x slope           :" << x(2,0)    << " +- " << sqrt(V(2,2)) << " cm \n";  
	std::cout << "y slope           :" << x(3,0)    << " +- " << sqrt(V(3,3)) << " cm \n";
}

//Double_t cutOnD(const TMatrixD& x, Double_t dcut)
//{
//  TMatrixD g(1,4);
//  Double_t weightsum = 0;
//  for(zDataConstIter i = zdata.begin() ; i != zdata.end() ; ++i) {
//    g(0,0) = - sin(i->Phi);
//    g(0,1) =   cos(i->Phi);
//    g(0,2) = i->Z * g(0,0);
//    g(0,3) = i->Z * g(0,1);
//    TMatrixD dcor = g * x;
//    //std::cout << dcor.GetNrows() << " , " << dcor.GetNcols() << "\n";
//    if(std::abs(i->D -  dcor(0,0)) > dcut) {
//      i->weight2 = 0; 
//    } else {
//      i->weight2 = 1;
//      weightsum += 1;
//    }
//  }
//  return weightsum;
//} 

Double_t cutOnChi2(const TMatrixD& x, Double_t chi2cut)
{
	double sigmabeam2 = 0.002 * 0.002;
  TMatrixD g(1,4);
  Double_t weightsum = 0;
  for(zDataIter i = zdata.begin() ; i != zdata.end() ; ++i) {
    g(0,0) = sin(i->Phi);
    g(0,1) = - cos(i->Phi);
    g(0,2) = i->Z * g(0,0);
    g(0,3) = i->Z * g(0,1);
    TMatrixD dcor = g * x;
    //std::cout << dcor.GetNrows() << " , " << dcor.GetNcols() << "\n";
    Double_t chi2 = (i->D -  dcor(0,0))* (i->D -  dcor(0,0)) / 
      (sigmabeam2 +  (i->SigD)*(i->SigD));
    if(chi2 > chi2cut) {
      i->weight2 = 0; 
    } else {
      i->weight2 = 1;
      weightsum += 1;
    }
  }
  //std::cout << weightsum << "\n";
  return weightsum;
}

int main(int argc, char **argv)
{
 
 //tnt *t = new tnt("PythiaGenerator/lhc_2008_aa_1001.root");
 std::cout << "name: " << argv[1] << std::endl;

 int cut_on_events=0;
 if (argc>=3) {
	 TString tmpst(argv[2]);
	 cut_on_events = tmpst.Atoi();
	 std::cout << " maximum number of tracks = " << cut_on_events << std::endl;
 }
 
 TString in_filename(argv[1]);
 TString tmp_string = in_filename;
 TString out_filename = tmp_string.Replace(in_filename.Length()-5,in_filename.Length(),"_results.root");
 if (argc==4) {
   out_filename = TString(argv[3]);
 }

 //static TROOT beamfit("beamfit","Beam fitting");
 //static TRint app("app",&argc,argv,NULL,0);

 TFile *outroot = new TFile(out_filename,"RECREATE");
 
 NtupleHelper *t = new NtupleHelper(in_filename);
 std::cout << " ntuple initialized" << std::endl;
  
 //TH1F *h_z = (TH1F*) gDirectory->Get("z0");
 //h_z->SetDirectory(0);
  
 
 t->Book();
 std::cout << " ntuple booked" << std::endl;
 zdata=t->Loop(cut_on_events);
 std::cout << " finished loop over events" << std::endl;

   
 //t->hsd->Draw();

 // fit z with simple chi2 fit
 //h_z->Fit("gaus");
 //TF1 *mygaus = h_z->GetFunction("gaus");
 //std::cout << "======== End of Chi2 Fit for Z ========\n" << std::endl;
 
 TMinuit *gmMinuit = new TMinuit(2); 
 gmMinuit->SetFCN(zfcn);
 int ierflg = 0;
 sfpar[Par_Sigma] = 7.55;//mygaus->GetParameter(2);
 sfpar[Par_Z0]    = 0.;//r/mygaus->GetParameter(1);
 errsfpar[Par_Sigma] = 0.;//r/mygaus->GetParError(2);
 errsfpar[Par_Z0] = 0.;//r/mygaus->GetParError(1);
 
 // UML Fit in z
 //Double_t zlimitUp[2] = { 0., 8. };
 //Double_t zlimitDown[2] = { 0., 7. };

 gmMinuit->SetFCN(zfcn);
 for (int i = 0; i<2; i++) {
	 gmMinuit->mnparm(i,par_name[i],sfpar[i],step[i],0,0,ierflg);
	 //gmMinuit->mnparm(i,par_name[i],sfpar[i],step[i],zlimitDown[i],zlimitUp[i],ierflg);
 }
 
 gmMinuit->Migrad();
 // gmMinuit->mncuve();
 // gmMinuit->mnmnos();
 //gmMinuit->Migrad();

 

 // get 1 sigma contour of param 0 vs 1
 ///gmMinuit->SetErrorDef(1);
 ///TGraph *gra1_sig1 = (TGraph*)gMinuit->Contour(60,0,1);
 //gra1_sig1->Draw("alp");
 //gra1_sig1->SetName("gra1_sig1");
 ///gmMinuit->SetErrorDef(4); // two sigma
 ///TGraph *gra1_sig2 = (TGraph*)gMinuit->Contour(60,0,1);
 //gra1_sig2->Draw("alp");
 //gra1_sig2->SetName("gra1_sig2");
 // gr12->Draw("alp");
 
 std::cout << "======== End of Maximum LH Fit for Z ========\n" << std::endl;

 //_______________ d0 phi fit _________________
 TMatrixD xmat(4,1);
 TMatrixDSym Verr(4);
 fit(xmat,Verr);

 Double_t dcut = 4.0;
 Double_t chi2cut = 20;
 //while( cutOnD(x,dcut) > 0.5 * zdata.size() ) {
 // below is a very artificial cut requesting that 50 % of the sample survive
 // we hould investigate if there are better criteria than that.
 //
 int nite = 0;
 while( cutOnChi2(xmat,chi2cut) > 0.5 * zdata.size() ) {
	 fit(xmat,Verr); 
	 dcut /= 1.5;
	 chi2cut /= 1.5;
	 nite++;
 }
 std::cout << " number of iterations: " << nite << std::endl;
 
 std::cout << "======== End of d0-phi Fit ========\n" << std::endl;

 //__________________ LH Fit for beam __________
 
 for (int i = 0; i<2; i++) { 
	 gmMinuit->GetParameter(i,sfpar[i],errsfpar[i]);
 }
 sfpar[Par_x0] = xmat(0,0);
 sfpar[Par_y0] = xmat(1,0);
 sfpar[Par_dxdz]= xmat(2,0);
 sfpar[Par_dydz]= xmat(3,0);
 sfpar[Par_Sigbeam]=0.0020; // for LHC
 //sfpar[Par_Sigbeam]=0.0022; // for CDF
 //sfpar[Par_eps] = 3.7e-8;
 //sfpar[Par_beta]= 55.;
 
 gmMinuit->mncler();
 gmMinuit->SetFCN(dfcn);
 
 for (int i = 2; i<7; i++) {   
   gmMinuit->mnparm(i,par_name[i],sfpar[i],step[i],0,0,ierflg);
 }
 //gmMinuit->FixParameter(Par_x0);
  
 //gmMinuit->mnparm(9,par_name[9],sfpar[9],step[9],0,0,ierflg);
 //gmMinuit->mnparm(10,par_name[10],sfpar[10],step[10],0,0,ierflg);
 
 gmMinuit->Migrad();
 /*
 // fix x0 and fit again
 for (int i = 2; i<7; i++) { 
	 gmMinuit->GetParameter(i,sfpar[i],errsfpar[i]);
 }
 gmMinuit->mncler();
 gmMinuit->SetFCN(dfcn); 
 for (int i = 2; i<7; i++) {   
   gmMinuit->mnparm(i,par_name[i],sfpar[i],step[i],0,0,ierflg);
 }
 gmMinuit->FixParameter(Par_x0);
 gmMinuit->Migrad();
 */
 std::cout << "======== End of Maximum LH Fit for Beam ========\n" << std::endl;

 //__________________ LH Fit for Product __________
 
 for (int i = 2; i<7; i++) { 
	 gmMinuit->GetParameter(i,sfpar[i],errsfpar[i]);
 }
 
 //gmMinuit->GetParameter(5,sfpar[Par_Sigbeam],errsfpar[Par_Sigbeam]);
 
 gmMinuit->mncler();
 gmMinuit->SetFCN(cfcn); 
 for (int i = 0; i<7; i++) {   
   gmMinuit->mnparm(i,par_name[i],sfpar[i],step[i],0,0,ierflg);
 }
 gmMinuit->Migrad();
 gmMinuit->mnhess();
 gmMinuit->Migrad();

 //   for (int i=0; i<2; i++) {
 //  gmMinuit->GetParameter(i,sfpar[i],errsfpar[i]);
 // }

 
 gmMinuit->SetErrorDef(4); // two sigma
 TGraph *gra1_sig2 = (TGraph*)gMinuit->Contour(60,2,3);
 gra1_sig2->Draw("alp");
 gra1_sig2->SetName("gra1_sig2");

 std::cout << "======== End of Maximum LH Fit Product for Beam ========\n" << std::endl;

 //__________________ LH Fit including resolution  __________
 
 for (int i = 0; i<7; i++) { 
	 gmMinuit->GetParameter(i,sfpar[i],errsfpar[i]);
 }

 // only for pixeless resolution use the generated value:
 sfpar[Par_Sigbeam] = 14.e-4;
 errsfpar[Par_Sigbeam] = 0.;
 //
 
 sfpar[Par_c0] = 0.0100;
 sfpar[Par_c1] = 0.0100;
	 
 gmMinuit->mncler();
 gmMinuit->SetFCN(dfcn2); 
 for (int i = 0; i<9; i++) {   
   if (i==7) {
     gmMinuit->mnparm(i,par_name[i],sfpar[i],step[i],1.e-5,1.e-1,ierflg);
   } else if (i==8) {
     gmMinuit->mnparm(i,par_name[i],sfpar[i],step[i],1.e-5,1.e-1,ierflg);
   } else {
     gmMinuit->mnparm(i,par_name[i],sfpar[i],step[i],0,0,ierflg);
   }
 }
 // fix beam width
 gmMinuit->FixParameter(Par_Sigbeam);

 gmMinuit->Migrad();


 std::cout << "======== End of Maximum LH Fit Product for Beam ========\n" << std::endl;


 outroot->cd();
 //r/h_z->Write(); //to draw only markers use "HISTPE1"
 //gra1_sig1->Write();
 gra1_sig2->Write();
 
 //app.Run();   
 return 0;
}
