// fitting function
double crystalball(double *x, double *par) {
  
  // par[0]:  mean
  // par[1]:  sigma
  // par[2]:  alpha, crossover point
  // par[3]:  n, length of tail
  // par[4]:  N, normalization
  
  double cb = 0.0;
  double exponent = 0.0;

/*   std::cout << " x = " << x[0] << " Par = "  */
/*             << par[0] << " "  */
/*             << par[1] << " "  */
/*             << par[2] << " "  */
/*             << par[3] << " "  */
/*             << par[4] << std::endl; */

  if (x[0] > par[0] - par[2]*par[1]) {
    exponent = (x[0] - par[0])/par[1];
    cb = exp(-exponent*exponent/2.);
  } 
  else {
    double nenner  = pow(par[3]/par[2], par[3])*exp(-par[2]*par[2]/2.);
    double zaehler = (par[0] - x[0])/par[1] + par[3]/par[2] - par[2];
    zaehler = pow(zaehler, par[3]);
    cb = nenner/zaehler;
  }
  if (par[4] > 0.) { cb *= par[4]; }

  //  std::cout << "CB = " << std::endl;

  return cb;
}

void photonContainmentAnalysis() {

  gStyle->SetOptFit(1111);
  
  TFile *infile = new TFile("contCorrAnalyzer.root");
  infile->ls();
  
  TH1F *theHistos[2];
  theHistos[0] = (TH1F*)infile->Get("contCorrAnalyzer/EB_e25EtrueReference");
  theHistos[1] = (TH1F*)infile->Get("contCorrAnalyzer/EE_e25EtrueReference");
  
  // fitting the distributions to extract the parameter
  TF1 *gausa;
  TF1 *cb_p;
  for(int myH=0; myH<2; myH++) {
  
    // histos parameters
    int peakBin   = theHistos[myH]->GetMaximumBin();
    double h_norm = theHistos[myH]->GetMaximum();
    double h_rms  = theHistos[myH]->GetRMS();
    double h_mean = theHistos[myH]->GetMean();
    double h_peak = theHistos[myH]->GetBinCenter(peakBin);
    
    // gaussian fit to initialize
    gausa = new TF1 ("gausa","[0]*exp(-1*(x-[1])*(x-[1])/2/[2]/[2])",h_peak-10*h_rms,h_peak+10*h_rms);
    gausa ->SetParameters(h_norm,h_peak,h_rms);
    theHistos[myH]->Fit("gausa","","",h_peak-3*h_rms,h_peak+3*h_rms);
    double gausNorm  = gausa->GetParameter(0);
    double gausMean  = gausa->GetParameter(1);
    double gausSigma = fabs(gausa->GetParameter(2));
    double gausChi2  = gausa->GetChisquare()/gausa->GetNDF();
    if (gausChi2>100){ gausMean = h_peak; gausSigma = h_rms; }
    
    // crystalball limits
    double myXmin = gausMean - 7.*gausSigma;
    double myXmax = gausMean + 5.*gausSigma;
    
    // crystalball fit
    cb_p = new TF1 ("cb_p",crystalball,myXmin,myXmax, 5) ;
    cb_p->SetParNames ("Mean","Sigma","alpha","n","Norm","Constant");
    cb_p->SetParameter(0, gausMean);
    cb_p->SetParameter(1, gausSigma);
    cb_p->SetParameter(2, 1.);
    cb_p->SetParLimits(2, 0.1, 5.);
    cb_p->FixParameter(3, 5.);
    cb_p->SetParameter(4, gausNorm);
    theHistos[myH]->Fit("cb_p","lR","",myXmin,myXmax);
    theHistos[myH]->GetXaxis()->SetRangeUser(0.95,1.05); 
    double matrix_gmean      = cb_p->GetParameter(0);
    double matrix_gmean_err  = cb_p->GetParError(0);
    
    c1->Update();
    if(myH == 0) { 
      c1->SaveAs("e25EtrueReference_EB.eps");
      std::cout << "E25 barrel containment: " << matrix_gmean << " +/- " << matrix_gmean_err << std::endl; 
    }
    if(myH == 1) { 
      c1->SaveAs("e25EtrueReference_EE.eps");
      std::cout << "E25 endcap containment: " << matrix_gmean << " +/- " << matrix_gmean_err << std::endl; 
    }
  }
}
