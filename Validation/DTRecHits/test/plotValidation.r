//------------------------------
//
// A macro to decorate validation plots produced with "local" option.
//
// Usage:
//
// .x plotValidation.r("file",wheel,station)
//
// Some drawing options are set in the code (see below).
//
//
// Author: N. Amapane
//
//------------------------------


void plotValidation(){
  cout << endl << "Usage: .x plotValidation.r(\"inputFile.root\",<wheel>,<station>)" << endl << endl;
}


void plotValidation(TString filename, int wheel, int station) {

  if (! TString(gSystem->GetLibraries()).Contains("Histograms_h")) {
    gROOT->LoadMacro("$CMSSW_BASE/src/Validation/DTRecHits/test/Histograms.h+");
    gROOT->LoadMacro("macros.C");
  }

  //----------------------------------------------------------------------
  //  Configurable options
   addProfile = false;
   addSlice = true;

  int rbx =2; // rebin x in scatter plots
  int rby =1; // rebin y in scatter plots
  int rbp = 1; // rebin profiles
  float nsigma = 2; // interval for the fit of residual distributions

  // Canvases to plot
  bool doPhiAndThetaS3 =true;
  bool doHitPull = true;
  bool doEff = true;
  bool doT0= true;
  bool doSegRes=true;
  bool doSegPull = true;
  bool doAngularDeps = true;
  bool doEff4D = true;
  bool doNSeg = true;


  //----------------------------------------------------------------------
 
  TStyle * style = getStyle("tdr");
  style->cd();  
  setPalette();
  gStyle->SetTitleSize(0.05,"XYZ"); // Set larger axis titles
  gStyle->SetTitleOffset(1.3,"Y");
  //  gStyle->SetOptTitle(0); // remove histogram titles

  //  gStyle->SetOptStat(111);

  float cmToMicron = 10000.;
  float vdrift = 54.3;


  TFile *file = new TFile(filename);
  
  HRes1DHit *hResPhi1  = new HRes1DHit(file, wheel, station, 1, "S3");
  HRes1DHit *hResTheta = new HRes1DHit(file, wheel, station, 2, "S3");
  //  HRes1DHit *hResPhi2  = new HRes1DHit(file, wheel, station, 4, "S3");
  HRes1DHit *hResPhi = hResPhi1;

  HEff1DHit* hEffS1RPhi= new HEff1DHit(file, wheel, station, 1, "S1");
  HEff1DHit* hEffS3RPhi= new HEff1DHit(file, wheel, station, 1, "S3");
  HEff1DHit* hEffS1RZ=0;
  HEff1DHit* hEffS3RZ=0;
  if (station!=4) {
    hEffS1RZ=   new HEff1DHit(file, wheel, station, 2, "S1");
    hEffS3RZ=   new HEff1DHit(file, wheel, station, 2, "S3");
  }
  

  HRes4DHit* hRes4D= new HRes4DHit(file, wheel, station, 0);
  HEff4DHit* hEff4D = new HEff4DHit(file, wheel, station, 0);


  // Result of fits
  float m_phi = 0.;
  float s_phi = 0.;
  float m_theta = 0.;
  float s_theta = 0.;
  float m_phi1 = 0.;
  float s_phi1 = 0.;
  float m_phi2 = 0.;
  float s_phi2 = 0.;
  float m_phiS1 = 0.;
  float s_phiS1 = 0.;
  float m_phiS2 = 0.;
  float s_phiS2 = 0.;
  float m_thetaS1 = 0.;
  float s_thetaS1 = 0.;
  float m_thetaS2 = 0.;
  float s_thetaS2 = 0.;
  float t0phi   =0.;
  float t0theta =0.;

  
  TString canvbasename = filename;
  canvbasename = canvbasename.Replace(canvbasename.Length()-5,5,"") + TString("_W") + (long) wheel + "_St" + (long) station ;


  //-------------------- Hit Residuals at step 3 in phi and theta (full distrib and vs distance from wire)
  if (doPhiAndThetaS3) {
    TCanvas* c1= new TCanvas;
    c1->SetTitle(canvbasename+"_ResPhiTheta"); 
    c1->SetName(canvbasename+"_ResPhiTheta");
    c1->Divide(2,2);
    
    c1->cd(1);

    TH1F* hRes;
    hRes=hResPhi->hRes;

    hResPhi->hRes->Rebin(2);
    TF1* fphi=drawGFit(hRes, nsigma, -0.4, 0.4);

    c1->cd(2);

    plotAndProfileX(hResPhi->hResVsPos,rbx,rby,rbp,-.1, .1, 0, 2.1);


    m_phi = fphi->GetParameter("Mean")*cmToMicron;
    s_phi = fphi->GetParameter("Sigma")*cmToMicron;

    if (hResTheta->hRes) {

      c1->cd(3);
      hRes=hResTheta->hRes;

      hResTheta->hRes->Rebin(2);
      TF1* ftheta=drawGFit(hRes, nsigma, -0.4, 0.4);

      c1->cd(4);  
      plotAndProfileX(hResTheta->hResVsPos,rbx,rby,rbp,-.1, .1, 0, 2.1);  
    
      m_theta = ftheta->GetParameter("Mean")*cmToMicron;
      s_theta = ftheta->GetParameter("Sigma")*cmToMicron;  
    }
    

    cout << canvbasename << "  Step3 W" << wheel << " St" << station << endl
	 << "   Res:          Phi: M= " << int(floor(m_phi+0.5))
	 << " S= "      << int(s_phi+0.5)
	 << "; Theta: M= " << int(floor(m_theta+0.5))
	 << " S= "  << int(s_theta+0.5) << endl;
  }


  //-------------------- Hit pulls
  if (doHitPull){
    TCanvas* c1= new TCanvas;
    c1->SetTitle(canvbasename+"_PullPhiTheta"); 
    c1->SetName(canvbasename+"_PullPhiTheta");
    c1->Divide(2,2);
    
    c1->cd(1);

    TH1F* hPull;
    hPull=hResPhi->hPull;

    //    hResPhi->hPull->Rebin(2);
    TF1* fphi=drawGFit(hPull, nsigma, -5, 5);

    c1->cd(2);

    plotAndProfileX(hResPhi->hPullVsPos,rbx,rby,rbp,-5, 5, 0, 2.1);


    m_phi = fphi->GetParameter("Mean")*cmToMicron;
    s_phi = fphi->GetParameter("Sigma")*cmToMicron;

    if (hResTheta->hPull) {

      c1->cd(3);
      hPull=hResTheta->hPull;

      //      hResTheta->hPull->Rebin(2);
      TF1* ftheta=drawGFit(hPull, nsigma, -5, 5);

      c1->cd(4);  
      plotAndProfileX(hResTheta->hPullVsPos,rbx,rby,rbp,-5, 5, 0, 2.1);  
    
      m_theta = ftheta->GetParameter("Mean");
      s_theta = ftheta->GetParameter("Sigma");  
    }
    

    cout << canvbasename << "  Step3 W" << wheel << " St" << station << endl
	 << "   Pulls:        Phi: M= " << int(floor(m_phi+0.5))
	 << " S= "      << int(s_phi+0.5)
	 << "; Theta: M= " << int(floor(m_theta+0.5))
	 << " S= "  << int(s_theta+0.5) << endl;
  }


  //-------------------- Hit efficiencies as a function of distance from wire
  if (doEff) {
    TCanvas* c1= new TCanvas;
    c1->SetTitle(canvbasename+"_EffPhiTheta");  
    c1->SetName(canvbasename+"_EffPhiTheta");
    c1->SetWindowSize(325,750);
    c1->Divide(1,2);
    c1->cd(1);
    plotEff(hEffS1RPhi->hEffVsDist, hEffS3RPhi->hEffVsDist);
    c1->cd(2);
    if (station!=4) plotEff(hEffS1RZ->hEffVsDist, hEffS3RZ->hEffVsDist);
//     c1->cd(2);
//     //plotEff(hEffS1RPhi->hEffVsPhi, hEffS3RPhi->hEffVsPhi);
//     plotEff(hEffS1RPhi->hEffVsEta, hEffS3RPhi->hEffVsEta);
//     c1->cd(4);
//     //    if (station!=4) plotEff(hEffS1RZ->hEffVsPhi, hEffS3RZ->hEffVsPhi);
//     if (station!=4) plotEff(hEffS1RZ->hEffVsEta, hEffS3RZ->hEffVsEta);

  }


  //-------------------- #hits, t0s
  if (doT0) {
    TCanvas* c1= new TCanvas;
    c1->SetTitle(canvbasename+"_NHitsT0");   
    c1->SetName(canvbasename+"_NHitsT0");
    c1->Divide(2,2);
    c1->cd(1);
    
    TH2F* hNh =  hRes4D->hHitMult;
 
    hNh->SetXTitle("#phi hits");
    hNh->SetYTitle("#theta hits");
    hNh->Draw("BOX");
    c1->cd(2);
    hNh->ProjectionY()->Draw();
    c1->cd(3);
    hNh->ProjectionX()->Draw();
    c1->cd(4);
   
    TH2F* ht =  hRes4D->ht0;
    ht->SetXTitle("t0 #phi");
    ht->SetYTitle("t0 #theta");
    ht->Draw("BOX");


  }

  //-------------------- Segment x, y, alpha, beta resolutions
  if (doSegRes){
    TCanvas* c1= new TCanvas;
    c1->SetTitle(canvbasename+"_ResSeg"); 
    c1->SetName(canvbasename+"_ResSeg");
    c1->Divide(2,2);
    c1->cd(1);
    //    hRes4D->hResX->Rebin(2);
    drawGFit(hRes4D->hResX, nsigma, -0.1, 0.1);
    c1->cd(2);
    //    hRes4D->hResAlpha->Rebin(2);
    drawGFit(hRes4D->hResAlpha, nsigma, -0.01, 0.01);
    c1->cd(3);
    hRes4D->hResYRZ->Rebin(2);
    drawGFit(hRes4D->hResYRZ, nsigma, -0.4, 0.4);
    c1->cd(4);
    hRes4D->hResBeta->Rebin(2);
    drawGFit(hRes4D->hResBeta, nsigma, -0.4, 0.4);
  }


 //-------------------- Angular dependencies
  if (doAngularDeps) {
    TCanvas* c1= new TCanvas;
    
    float min;
    float max;
    c1->SetTitle(canvbasename+"_MeanVsAngles");
    c1->SetName(canvbasename+"_MeanVsAngles");
    c1->SetTitle(canvbasename+" Angles");
    
    c1->Divide(2,2);

    c1->cd(1);
    hRes4D->hSimAlpha->SetLineColor(kGreen);
    hRes4D->hSimAlpha->Draw();
    hRes4D->hRecAlpha->Draw("same");

    c1->cd(2);  
    plotAndProfileX(hResPhi->hResVsAngle,1,1,1,-.04, .04, -1., 1.);

    c1->cd(3);
    hRes4D->hSimBetaRZ->SetLineColor(kGreen);
    hRes4D->hSimBetaRZ->Draw();
    hRes4D->hRecBetaRZ->Draw("same");
       
    c1->cd(4);
    plotAndProfileX(hResTheta->hResVsAngle,1,1,1,-.04, .04, -1.2.,1.2);
   

  }



  //------------------- Efficiencies Vs X, Y. alpha, beta

  if(doEff4D){
    // FIXME: should rebin histograms.
    
    TH1F* hEffX;
    TH1F* hEffY;
    TH1F* hEffalpha;
    TH1F* hEffbeta;

    TCanvas* c1= new TCanvas;
    c1->SetTitle(canvbasename+"_Efficiencies"); 
    c1->SetName(canvbasename+"_Efficiencies");
    c1->SetTitle(canvbasename+"_Efficiencies");
    
    c1->Divide(2,2);
    
    c1->cd(1);
    hEffX = getEffPlot(hEff4D->hXRecHit, hEff4D->hXSimSegm,2);
    hEffX->Draw();

    c1->cd(2);
    hEffalpha = getEffPlot(hEff4D->hAlphaRecHit, hEff4D->hAlphaSimSegm,2);
    hEffalpha->Draw();

    c1->cd(3);
    hEffY = getEffPlot(hEff4D->hYRecHit, hEff4D->hYSimSegm,2);
    hEffY->Draw();

    c1->cd(4);
    hEffbeta = getEffPlot(hEff4D->hBetaRecHit, hEff4D->hBetaSimSegm,2);
    hEffbeta->Draw();



    }

 //-------------------- Segment x, y, alpha, beta pull
  if (doSegPull){
    TCanvas* c1= new TCanvas;
    c1->SetTitle(canvbasename+"_PullSeg"); 
    c1->SetName(canvbasename+"_PullSeg");
    c1->Divide(2,2);
    c1->cd(1);

    hRes4D->hPullX->Rebin(2);
    drawGFit(hRes4D->hPullX, nsigma, -10.,10.);
    c1->cd(2);
    hRes4D->hPullAlpha->Rebin(2);
    drawGFit(hRes4D->hPullAlpha, nsigma, -10., 10.);
    c1->cd(3);
    hRes4D->hPullYRZ->Rebin(2);
    drawGFit(hRes4D->hPullYRZ, nsigma, -10., 10.);
    c1->cd(4);
    hRes4D->hPullBetaRZ->Rebin(2);
    drawGFit(hRes4D->hPullBetaRZ, nsigma, -10.,10.);

    //Fixme: Move these to another canvas. Note that the error used for hPullY is not computed correctly.q
//     c1->cd(3);
//     hRes4D->hPullY->Rebin(2);
//     drawGFit(hRes4D->hPullY, nsigma, -10., 10.);
//     c1->cd(4);
//     hRes4D->hPullBeta->Rebin(2);
//     drawGFit(hRes4D->hPullBeta, nsigma, -10.,10.);  


  }

  //-------------------- #segments
  if (doNSeg) {
    TCanvas* c1= new TCanvas;
    c1->SetTitle(canvbasename+"_NSeg");   
    c1->SetName(canvbasename+"_NSeg");
    // c1->Divide(2,2);
//     c1->cd(1);
    
    TH1F* hNs =  hEff4D->hNSeg;
 
    hNs->SetXTitle("#segments");
    hNs->Draw();

    double int_1 = hNs->Integral(2,21);
    double int_2 = hNs->Integral(3,21);
    
    double ratio = int_2/int_1;
    
    cout << "int_1: " << int_1 <<" int_2: " << int_2 << " ratio: " << ratio << endl;
    
  }


} 
