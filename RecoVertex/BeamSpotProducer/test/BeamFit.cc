#include <iostream>
#include <stdlib.h>
#include <cmath>
#include "TROOT.h"
#include "TFile.h"
#include "TRint.h"
#include "TH1.h"
#include "TF1.h"
#include "TF2.h"
#include "TH2.h"
#include "TCanvas.h"
#include "TMinuit.h"
#include "TSystem.h"
#include "TGraphErrors.h"
#include "TMatrixD.h"
#include "TMatrixDSym.h"
#include "NtupleHelper.h"
#include "global.h"
//#include "beamfit_fcts.h"

//const char par_name[dim][20]={"z0  ","sigma ","emmitance","beta*"};
const char par_name[dim][20] = {
    "z0  ", "sigma ", "x0 ", "y0 ", "dxdz ", "dydz ", "sigmaBeam ", "c0  ", "c1    ", "emittance ", "betastar "};

Double_t params[dim], errparams[dim];
Double_t sfpar[dim], errsfpar[dim];

constexpr Double_t step[dim] = {1.e-5, 1.e-5, 1.e-3, 1.e-3, 1.e-3, 1.e-3, 1.e-5, 1.e-5, 1.e-5, 1.e-5, 1.e-5};
zData zdata;  //!
int tmpNtrks_ = 0;
int fnthite = 0;
TMatrixD ftmp;

Double_t fd0cut = 4.0;

int sequence[11] = {500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000};
Double_t sequenceD[11] = {500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000};
Double_t zeros[11] = {0.};

int iloop_ = 0;

int Nsequence_ = 11;
Double_t res_d0phi_X[11] = {0.};
Double_t res_d0phi_Xerr[11] = {0.};
Double_t res_d0phi_Y[11] = {0.};
Double_t res_d0phi_Yerr[11] = {0.};
Double_t res_d0phi_dXdz[11] = {0.};
Double_t res_d0phi_dXdzerr[11] = {0.};
Double_t res_d0phi_dYdz[11] = {0.};
Double_t res_d0phi_dYdzerr[11] = {0.};

Double_t res_Z[11] = {0.};
Double_t res_sigmaZ[11] = {0.};
Double_t res_Zerr[11] = {0.};
Double_t res_sigmaZerr[11] = {0.};
Double_t res_Z_lh[11] = {0.};
Double_t res_sigmaZ_lh[11] = {0.};
Double_t res_Zerr_lh[11] = {0.};
Double_t res_sigmaZerr_lh[11] = {0.};

TGraphErrors *g_X;
TGraphErrors *g_Y;
TGraphErrors *g_dXdz;
TGraphErrors *g_dYdz;
TGraphErrors *g_Z;
TGraphErrors *g_sigmaZ;
TGraphErrors *g_Z_lh;
TGraphErrors *g_sigmaZ_lh;

Double_t zdis(Double_t z, Double_t sigma, Double_t *parms) {
  //---------------------------------------------------------------------------
  //  This is a simple function to parameterize the z-vertex distribution. This
  // is parametrized by a simple normalized gaussian distribution.
  //---------------------------------------------------------------------------

  Double_t sig = sqrt(sigma * sigma + parms[Par_Sigma] * parms[Par_Sigma]);
  Double_t result = (exp(-((z - parms[Par_Z0]) * (z - parms[Par_Z0])) / (2.0 * sig * sig))) / (sig * sqrt2pi);
  return result;
}

void zfcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *params, Int_t iflag) {
  //----------------------------------------------------------------------------------
  // this is the function used by minuit to do the unbinned fit to the z distribution
  //----------------------------------------------------------------------------------
  f = 0.0;
  for (zDataConstIter iter = zdata.begin(); iter != zdata.end(); ++iter) {
    f = log(zdis(iter->Z, iter->SigZ, params)) + f;
  }
  f = -2.0 * f;
  return;
}

float betaf(float betastar, float emmitance, float z, float z0) {
  float x = sqrt(emmitance * (betastar + (((z - z0) * (z - z0)) / betastar)));
  return x;
}

Double_t ddis(Double_t z, Double_t sigma, Double_t d, Double_t sigmad, Double_t *parms) {
  //---------------------------------------------------------------------------
  // This is a simple function to parameterize the sigma of the beam at a given z.
  // This is parametrized by a simple normalized gaussian distribution.
  //---------------------------------------------------------------------------
  Double_t sig = betaf(parms[Par_beta], parms[Par_eps], z, parms[Par_Z0]);
  sig = sqrt(sig * sig + sigmad * sigmad);
  Double_t result = (exp(-(d * d) / (2.0 * sig * sig))) / (sig * sqrt2pi);
  return result;
}

Double_t allddis(Double_t z, Double_t d, Double_t sigmad, Double_t phi0, Double_t *parms) {
  //---------------------------------------------------------------------------
  // This is a simple function to parameterize the beam parameters
  // This is parametrized by a simple normalized gaussian distribution.
  //---------------------------------------------------------------------------
  Double_t sig =
      sqrt(parms[Par_Sigbeam] * parms[Par_Sigbeam] + 10 * sigmad * sigmad);  //9 //2.5 factor for full simu data
  Double_t dprime =
      d - (parms[Par_x0] + z * parms[Par_dxdz]) * sin(phi0) + (parms[Par_y0] + z * parms[Par_dydz]) * cos(phi0);
  Double_t result = (exp(-(dprime * dprime) / (2.0 * sig * sig))) / (sig * sqrt2pi);
  return result;
}

Double_t allddisBeta(Double_t z, Double_t sigma, Double_t d, Double_t sigmad, Double_t phi0, Double_t *parms) {
  //---------------------------------------------------------------------------
  // This is a simple function to parameterize the beam parameters
  // This is parametrized by a simple normalized gaussian distribution.
  //---------------------------------------------------------------------------
  Double_t sigmabeam =
      sqrt(parms[Par_eps] * (parms[Par_beta] + (((z - parms[Par_Z0]) * (z - parms[Par_Z0])) / parms[Par_beta])));

  Double_t sig = sqrt(sigmabeam * sigmabeam + sigmad * sigmad);

  Double_t dprime =
      d - (parms[Par_x0] + z * parms[Par_dxdz]) * sin(phi0) + (parms[Par_y0] + z * parms[Par_dydz]) * cos(phi0);
  Double_t result = (exp(-(dprime * dprime) / (2.0 * sig * sig))) / (sig * sqrt2pi);
  return result;
}

Double_t allddis2(Double_t z, Double_t sigma, Double_t d, Double_t pt, Double_t phi0, Double_t *parms) {
  //---------------------------------------------------------------------------
  //
  //---------------------------------------------------------------------------
  Double_t sigmad = parms[Par_c0] + parms[Par_c1] / pt;
  Double_t sig = sqrt(parms[Par_Sigbeam] * parms[Par_Sigbeam] + sigmad * sigmad);
  Double_t dprime =
      d - (parms[Par_x0] + z * parms[Par_dxdz]) * sin(phi0) + (parms[Par_y0] + z * parms[Par_dydz]) * cos(phi0);
  Double_t result = (exp(-(dprime * dprime) / (2.0 * sig * sig))) / (sig * sqrt2pi);
  return result;
}

void dfcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *params, Int_t iflag) {
  //----------------------------------------------------------------------------------
  // this is the function used by minuit to do the unbinned fit to the IP distribution
  //----------------------------------------------------------------------------------
  f = 0.0;
  for (zDataConstIter iter = zdata.begin(); iter != zdata.end(); ++iter) {
    //if(iter->weight2 == 0) continue;

    f = log(allddis(iter->Z, iter->D, iter->SigD, iter->Phi, params)) + f;
  }
  f = -2.0 * f;
  return;
}

void dfcnbeta(Int_t &npar, Double_t *gin, Double_t &f, Double_t *params, Int_t iflag) {
  //----------------------------------------------------------------------------------
  // this is the function used by minuit to do the unbinned fit to the IP distribution
  //----------------------------------------------------------------------------------
  f = 0.0;
  for (zDataConstIter iter = zdata.begin(); iter != zdata.end(); ++iter) {
    f = log(allddisBeta(iter->Z, iter->SigZ, iter->D, iter->SigD, iter->Phi, params)) + f;
  }
  f = -2.0 * f;
  return;
}

void dfcn2(Int_t &npar, Double_t *gin, Double_t &f, Double_t *params, Int_t iflag) {
  //----------------------------------------------------------------------------------
  // this is the function used by minuit to do the unbinned fit to the IP distribution
  //----------------------------------------------------------------------------------
  f = 0.0;
  for (zDataConstIter iter = zdata.begin(); iter != zdata.end(); ++iter) {
    f = log(allddis2(iter->Z, iter->SigZ, iter->D, iter->Pt, iter->Phi, params) * zdis(iter->Z, iter->SigZ, params)) +
        f;
  }
  f = -2.0 * f;
  return;
}

void cfcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *params, Int_t iflag) {
  //-----------------------------------------------------------------------------------
  // this is the function used by minuit to do the unbinned finned combined in z an IP.
  //-----------------------------------------------------------------------------------
  f = 0.0;
  for (zDataConstIter iter = zdata.begin(); iter != zdata.end(); ++iter) {
    f = log(allddis(iter->Z, iter->D, iter->SigD, iter->Phi, params) * zdis(iter->Z, iter->SigZ, params)) + f;
  }
  f = -2.0 * f;
  return;
}

void fit(TMatrixD &x, TMatrixDSym &V) {
  TMatrixDSym Vint(4);
  TMatrixD b(4, 1);
  Double_t weightsum = 0;
  tmpNtrks_ = 0;

  Vint.Zero();
  b.Zero();
  TMatrixD g(4, 1);
  TMatrixDSym temp(4);
  for (zDataConstIter i = zdata.begin(); i != zdata.end(); ++i) {
    //std::cout << "weight  " << sqrt(i->weight2) << "\n";
    if (i->weight2 == 0)
      continue;

    g(0, 0) = sin(i->Phi);
    g(1, 0) = -cos(i->Phi);
    g(2, 0) = i->Z * g(0, 0);
    g(3, 0) = i->Z * g(1, 0);

    temp.Zero();
    for (int j = 0; j < 4; ++j) {
      for (int k = j; k < 4; ++k) {
        temp(j, k) += g(j, 0) * g(k, 0);
      }
    }
    double sigmabeam2 = 0.002 * 0.002;
    //double sigma2 = sigmabeam2 +  (i->SigD)* (i->SigD) / i->weight2;
    double sigma2 = sigmabeam2 + (i->SigD) * (i->SigD);

    TMatrixD ftmptrans(1, 4);
    ftmptrans = ftmptrans.Transpose(ftmp);
    TMatrixD dcor = ftmptrans * g;

    bool pass = true;
    if ((fnthite > 0) && (std::abs(i->D - dcor(0, 0)) > fd0cut))
      pass = false;
    //if ( tmpNtrks_ < 10 && fnthite>0 ) {
    //std::cout << " d0 = " << i->D << " dcor(0,0)= " << dcor(0,0) << " fd0cut=" << fd0cut << std::endl;
    //}

    if (pass) {
      Vint += (temp * (1 / sigma2));
      b += ((i->D / sigma2) * g);
      weightsum += sqrt(i->weight2);
      tmpNtrks_++;
    }
  }
  Double_t determinant;
  V = Vint.InvertFast(&determinant);
  x = V * b;
  ftmp = x;

  std::cout << "number of tracks used in this iteration = " << tmpNtrks_ << std::endl;
  std::cout << "Sum of all weights:" << weightsum << "\n";
  std::cout << "x0                :" << x(0, 0) << " +- " << sqrt(V(0, 0)) << " cm \n";
  std::cout << "y0                :" << x(1, 0) << " +- " << sqrt(V(1, 1)) << " cm \n";
  std::cout << "x slope           :" << x(2, 0) << " +- " << sqrt(V(2, 2)) << " cm \n";
  std::cout << "y slope           :" << x(3, 0) << " +- " << sqrt(V(3, 3)) << " cm \n";

  res_d0phi_X[iloop_] = x(0, 0);
  res_d0phi_Xerr[iloop_] = sqrt(V(0, 0));
  res_d0phi_Y[iloop_] = x(1, 0);
  res_d0phi_Yerr[iloop_] = sqrt(V(1, 1));
  res_d0phi_dXdz[iloop_] = x(2, 0);
  res_d0phi_dXdzerr[iloop_] = sqrt(V(2, 2));
  res_d0phi_dYdz[iloop_] = x(3, 0);
  res_d0phi_dYdzerr[iloop_] = sqrt(V(3, 3));
}

//Double_t cutOnD(const TMatrixD& x, Double_t fd0cut)
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
//    if(std::abs(i->D -  dcor(0,0)) > fd0cut) {
//      i->weight2 = 0;
//    } else {
//      i->weight2 = 1;
//      weightsum += 1;
//    }
//  }
//  return weightsum;
//}

Double_t cutOnChi2(const TMatrixD &x, Double_t chi2cut) {
  double sigmabeam2 = 0.002 * 0.002;
  TMatrixD g(1, 4);
  Double_t weightsum = 0;
  for (zDataIter i = zdata.begin(); i != zdata.end(); ++i) {
    g(0, 0) = sin(i->Phi);
    g(0, 1) = -cos(i->Phi);
    g(0, 2) = i->Z * g(0, 0);
    g(0, 3) = i->Z * g(0, 1);
    TMatrixD dcor = g * x;
    //std::cout << dcor.GetNrows() << " , " << dcor.GetNcols() << "\n";
    Double_t chi2 = (i->D - dcor(0, 0)) * (i->D - dcor(0, 0)) / (sigmabeam2 + (i->SigD) * (i->SigD));
    if (chi2 > chi2cut) {
      i->weight2 = 0;
    } else {
      i->weight2 = 1;
      weightsum += 1;
    }
  }
  //std::cout << weightsum << "\n";
  return weightsum;
}

void fitAll() {
  std::cout << "starting fits" << std::endl;

  // chi2 fit for Z
  TH1F *h1z = new TH1F("h1z", "z distribution", 100, -50., 50.);
  for (zDataConstIter i = zdata.begin(); i != zdata.end(); ++i) {
    h1z->Fill(i->Z, i->SigZ);
  }
  h1z->Sumw2();
  h1z->Fit("gaus");
  //std::cout << "fitted "<< std::endl;

  TF1 *fgaus = h1z->GetFunction("gaus");
  //std::cout << "got function" << std::endl;
  res_Z[iloop_] = fgaus->GetParameter(1);
  res_sigmaZ[iloop_] = fgaus->GetParameter(2);
  res_Zerr[iloop_] = fgaus->GetParError(1);
  res_sigmaZerr[iloop_] = fgaus->GetParError(2);

  std::cout << "======== End of Chi2 Fit for Z ========\n" << std::endl;

  // LH fit for Z
  TMinuit *gmMinuit = new TMinuit(2);
  gmMinuit->SetFCN(zfcn);
  std::cout << "SetFCN done" << std::endl;
  int ierflg = 0;
  sfpar[Par_Sigma] = 7.55;   //mygaus->GetParameter(2);
  sfpar[Par_Z0] = 0.;        //r/mygaus->GetParameter(1);
  errsfpar[Par_Sigma] = 0.;  //r/mygaus->GetParError(2);
  errsfpar[Par_Z0] = 0.;     //r/mygaus->GetParError(1);

  // UML Fit in z
  //Double_t zlimitUp[2] = { 0., 8. };
  //Double_t zlimitDown[2] = { 0., 7. };

  gmMinuit->SetFCN(zfcn);
  for (int i = 0; i < 2; i++) {
    gmMinuit->mnparm(i, par_name[i], sfpar[i], step[i], 0, 0, ierflg);
    //gmMinuit->mnparm(i,par_name[i],sfpar[i],step[i],zlimitDown[i],zlimitUp[i],ierflg);
  }

  //std::cout << "fit..." << std::endl;

  gmMinuit->Migrad();
  // gmMinuit->mncuve();
  // gmMinuit->mnmnos();
  //gmMinuit->Migrad();
  for (int i = 0; i < 2; i++) {
    gmMinuit->GetParameter(i, sfpar[i], errsfpar[i]);
  }

  res_Z_lh[iloop_] = sfpar[Par_Z0];
  res_sigmaZ_lh[iloop_] = sfpar[Par_Sigma];
  res_Zerr_lh[iloop_] = errsfpar[Par_Z0];
  res_sigmaZerr_lh[iloop_] = errsfpar[Par_Sigma];

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

  tmpNtrks_ = 0;  // reset track counter

  TMatrixD xmat(4, 1);
  TMatrixDSym Verr(4);
  fit(xmat, Verr);  // get initial values
  fnthite++;

  //Double_t chi2cut = 20;

  int fminNtrks = 100;
  double fconvergence = 0.9;

  while (tmpNtrks_ > fconvergence * zdata.size()) {
    // below is a very artificial cut requesting that 50 % of the sample survive
    // we hould investigate if there are better criteria than that.
    //

    // while( cutOnChi2(xmat,chi2cut) > 0.5 * zdata.size() ) {
    //	 tmpNtrks_ = 0;

    fit(xmat, Verr);
    fd0cut /= 1.5;
    //chi2cut /= 1.5;
    if (tmpNtrks_ > fconvergence * zdata.size() && tmpNtrks_ > fminNtrks)
      fnthite++;
  }
  std::cout << " number of iterations: " << fnthite << std::endl;

  std::cout << "======== End of d0-phi Fit ========\n" << std::endl;

  //__________________ LH Fit for beam __________
  /*
   
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
 
 // fix x0 and fit again
 //for (int i = 2; i<7; i++) { 
//	 gmMinuit->GetParameter(i,sfpar[i],errsfpar[i]);
// }
 //gmMinuit->mncler();
 //gmMinuit->SetFCN(dfcn); 
 //for (int i = 2; i<7; i++) {   
  // gmMinuit->mnparm(i,par_name[i],sfpar[i],step[i],0,0,ierflg);
// }
 //gmMinuit->FixParameter(Par_x0);
 //gmMinuit->Migrad();
 
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

*/

  //outroot->cd();
  //r/h_z->Write(); //to draw only markers use "HISTPE1"
  //gra1_sig1->Write();

  //gra1_sig2->Write();

  //app.Run();
  // return 0;
}

int main(int argc, char **argv) {
  //gROOT->SetBatch(true);

  //tnt *t = new tnt("PythiaGenerator/lhc_2008_aa_1001.root");
  std::cout << "name: " << argv[1] << std::endl;

  ftmp.ResizeTo(4, 1);
  ftmp.Zero();

  int cut_on_events = 0;
  bool run_sequence = false;

  if (argc >= 3) {
    TString tmpst(argv[2]);
    cut_on_events = tmpst.Atoi();
    if (cut_on_events == -1) {
      run_sequence = true;
      std::cout << " run a sequence of fits for 0.5k, 1k, 2k, 5k, 10k" << std::endl;
    } else {
      std::cout << " maximum number of tracks = " << cut_on_events << std::endl;
    }
  }

  TString in_filename(argv[1]);
  TString tmp_string = in_filename;
  TString out_filename = tmp_string.Replace(in_filename.Length() - 5, in_filename.Length(), "_results.root");
  if (argc == 4) {
    out_filename = TString(argv[3]);
  }

  //static TROOT beamfit("beamfit","Beam fitting");
  //static TRint app("app",&argc,argv,NULL,0);

  TFile *outroot = new TFile(out_filename, "RECREATE");

  NtupleHelper *t = new NtupleHelper(in_filename);
  std::cout << " ntuple initialized" << std::endl;

  //TH1F *h_z = (TH1F*) gDirectory->Get("z0");
  //h_z->SetDirectory(0);

  t->Book();
  std::cout << " ntuple booked" << std::endl;

  if (run_sequence) {
    for (int iloop = 0; iloop < Nsequence_; iloop++) {
      std::cout << " loop over # " << sequence[iloop] << " tracks." << std::endl;
      zdata = t->Loop(sequence[iloop]);
      fitAll();
      std::cout << " loop has finished." << std::endl;
      iloop_++;
    }

    outroot->cd();
    g_X = new TGraphErrors(Nsequence_, sequenceD, res_d0phi_X, zeros, res_d0phi_Xerr);
    g_Y = new TGraphErrors(Nsequence_, sequenceD, res_d0phi_Y, zeros, res_d0phi_Yerr);
    g_dXdz = new TGraphErrors(Nsequence_, sequenceD, res_d0phi_dXdz, zeros, res_d0phi_dXdzerr);
    g_dYdz = new TGraphErrors(Nsequence_, sequenceD, res_d0phi_dYdz, zeros, res_d0phi_dYdzerr);

    g_Z = new TGraphErrors(Nsequence_, sequenceD, res_Z, zeros, res_Zerr);
    g_sigmaZ = new TGraphErrors(Nsequence_, sequenceD, res_sigmaZ, zeros, res_sigmaZerr);

    g_Z_lh = new TGraphErrors(Nsequence_, sequenceD, res_Z_lh, zeros, res_Zerr_lh);
    g_sigmaZ_lh = new TGraphErrors(Nsequence_, sequenceD, res_sigmaZ_lh, zeros, res_sigmaZerr_lh);

    g_X->SetName("beamX");
    g_Y->SetName("beamY");
    g_dXdz->SetName("beamdXdz");
    g_dYdz->SetName("beamdYdz");
    g_Z->SetName("beamZ");
    g_sigmaZ->SetName("beamSigmaZ");
    g_Z_lh->SetName("beamZ_likelihood");
    g_sigmaZ_lh->SetName("beamSigmaZ_likelihood");

    //TCanvas *cv_x = new TCanvas("cv_x","beam_X",700,700);
    //g_X->Draw("AP");

    g_X->Write();
    g_Y->Write();
    g_dXdz->Write();
    g_dYdz->Write();
    g_Z->Write();
    g_sigmaZ->Write();
    g_Z_lh->Write();
    g_sigmaZ_lh->Write();

    outroot->Close();

  } else {
    zdata = t->Loop(cut_on_events);
    fitAll();
    std::cout << " finished loop over events" << std::endl;
  }

  //cv_x->Print("beam_X.png");

  return 0;
}
