//------------------------------
//
// Write table of true hit resolutions, pull width, and efficiencies from DQM files produced in local mode
//
// Usage:
// .x writeSummaryTable.r("inputFile.root",doEff,doRes,doPull)
//
// Author: N. Amapane
//
//------------------------------
#include <iomanip>


void writeSummaryTable(){
  cout << endl << "Usage: .x writeSummaryTable.r(\"inputFile.root\",doEff,doEffab,doRes,doResab,doPull)" << endl << endl;
}


void writeSummaryTable(TString filename, bool doEff=true, bool doEffab = true, bool doRes=true, bool doResab = true, bool doPull=true,bool doPullabxy=true, bool doSegRes=true, bool doNSeg= true, bool doPar = false) {

  //----------------------------------------------------------------------
  // Configurable options  

  // Interval for the fit of residual distributions
  float nsigma = 2.;

  //----------------------------------------------------------------------

  if (! TString(gSystem->GetLibraries()).Contains("Histograms_h")) {
    gROOT->LoadMacro("$CMSSW_BASE/src/Validation/DTRecHits/test/Histograms.h+");
    gROOT->LoadMacro("macros.C");
    gROOT->LoadMacro("ranges.C");
  }

  float cmToMicron = 10000.;

  TFile *file = new TFile(filename);

  TString table = filename.Replace(filename.Length()-5,5,"_summary.txt");
  ofstream f(table,ios_base::out);
  f << fixed;  
  f << "# W St sec SL effS1RPhi effS3RPhi effSeg resHit pullHit meanAngle  sigmaAngle meanSegPos sigmaSegPos" << endl;

  // All sectors are collapsed together as of now
  int smin = 0;
  int smax = 0;

  for (int wheel = -2; wheel<3; ++wheel) {
    for (int station =1; station<=4; ++station) { 
      for (int sector = smin; sector<=smax; ++sector) {

	if (station!=4 && sector>12) continue;

	double stheta = -1.;
	double sphi = -1.;
	double ptheta = -1.;
	double pphi = -1.;
	double salpha_res = -1.;
	double sbeta_res = -1.;
	double malpha_res = -1.;
	double mbeta_res = -1.;
	double salpha_pull = -1.;
	double sbeta_pull = -1.;
	double malpha_pull = -1.;
	double mbeta_pull = -1.;
	double sx = -1.;
	double sy = -1.;
	double mx = -1.;
	double my = -1.;	
	double sx_pull = -1.;
	double sy_pull = -1.;
	double mx_pull = -1.;
	double my_pull = -1.;

	float effS1RPhi=0.;
	float effS3RPhi=0.;
	float effS1RZ=0.;
	float effS3RZ=0.;
	float effSeg = 0;



	HRes1DHit *hResPhi1  = new HRes1DHit(file, wheel, station, 1, "S3");
	HRes1DHit *hResTheta = new HRes1DHit(file, wheel, station, 2, "S3");

	HEff1DHit* hEffS1RPhi= new HEff1DHit(file, wheel, station, 1, "S1");
	HEff1DHit* hEffS3RPhi= new HEff1DHit(file, wheel, station, 1, "S3");
	HEff1DHit* hEffS1RZ=0;
	HEff1DHit* hEffS3RZ=0;

	HRes4DHit* hRes4D= new HRes4DHit(file, wheel, station, 0);
	HEff4DHit* hEff4D = new HEff4DHit(file, wheel, station, 0);


	if (station!=4) {
	  hEffS1RZ=   new HEff1DHit(file, wheel, station, 2, "S1");
	  hEffS3RZ=   new HEff1DHit(file, wheel, station, 2, "S3");
	}



	TCanvas c;

	//--- Hit efficiency
	if (doEff) {
	  if (station!=4) {
	    effS1RZ = hEffS1RZ->hDistRecHit->Integral()/hEffS1RZ->hDistMuSimHit->Integral();
	    effS3RZ = hEffS3RZ->hDistRecHit->Integral()/hEffS3RZ->hDistMuSimHit->Integral();
	  }
	  effS1RPhi = hEffS1RPhi->hDistRecHit->Integral()/hEffS1RPhi->hDistMuSimHit->Integral();
	  effS3RPhi = hEffS3RPhi->hDistRecHit->Integral()/hEffS3RPhi->hDistMuSimHit->Integral();
	  
	  cout << " " << wheel << " " << station << " " << sector <<  " effPhi: " << effS1RPhi << " " << effS3RPhi;
	  if (station!=4) cout <<  " effZ: " << effS1RZ << " " << effS3RZ;
	  cout << endl;	  
	}
	

	//--- Segment efficiency
	if (doEffab) {
	  if (station!=4) {
	    
	    effSeg = hEff4D->hBetaRecHit->Integral()/hEff4D->hBetaSimSegm->Integral();
	   	    
	    cout << " " << wheel << " " << station << " " << sector <<  " effSeg: " << effSeg << endl;
	   
	    
	  }
	}


	//--- Hit resolution
	if (doRes) {
	  if (station!=4) {
	    TH1F* tmpTheta = (TH1F*) hResTheta->hRes->Clone("tmpTheta");
	    tmpTheta->Rebin(2);
	    TF1* ftheta = drawGFit(tmpTheta, nsigma, -2. , 2.);
	    stheta = ftheta->GetParameter("Sigma");
	  }
	  TH1F* tmpPhi = (TH1F*) hResPhi1->hRes->Clone("tmpPhi");
	  tmpPhi->Rebin(2);
	  TF1* fphi = drawGFit(tmpPhi, nsigma, -2. , 2. );
	  sphi = fphi->GetParameter("Sigma");

	  cout << " " << wheel << " " << station << " " << sector  <<  " sphi: " << sphi;
	  if (station!=4) cout  << " stheta: " << stheta ;
	  cout << endl;
	}


	//--- Hit pull
	if (doPull) {
	  if (station!=4) {
	    TH1F* tmpTheta = (TH1F*) hResTheta->hPull->Clone("tmpTheta");
	    TF1* ftheta = drawGFit(tmpTheta, nsigma, -2. , 2.);
	    ptheta = ftheta->GetParameter("Sigma");
	  }
	  TH1F* tmpPhi = (TH1F*) hResPhi1->hPull->Clone("tmpPhi");	
	  TF1* fphi = drawGFit(tmpPhi, nsigma, -2. , 2. );
	  pphi = fphi->GetParameter("Sigma");

	  cout << " " << wheel << " " << station << " " << sector 
	       <<  " pphi: " << pphi;
	  if (station!=4) {
	    cout  << " ptheta: " << ptheta ;
	  } 
	  cout << endl;
	}


	//--- Segment position resolution
	if (doSegRes) {
	  if (station!=4) {
	    TH1F* tmpY = (TH1F*) hRes4D->hResYRZ->Clone("tmpY");
	    TF1* fy = drawGFit(tmpY, nsigma, -2. , 2. );
	    sy = fy->GetParameter("Sigma");
	    my = fy->GetParameter("Mean");
	  }
	  
	  TH1F* tmpX = (TH1F*) hRes4D->hResX->Clone("tmpX");
	  TF1* fx = drawGFit(tmpX, nsigma, -2. , 2.);
	  sx = fx->GetParameter("Sigma");
	  mx = fx->GetParameter("Mean");
	}
	
	//--- Segment angle resolution
	if (doResab) {
	  if (station!=4) {
	    TH1F* tmpBeta = (TH1F*) hRes4D->hResBeta->Clone("tmpBeta");
	    tmpBeta->Rebin(2);
	    TF1* fbeta = drawGFit(tmpBeta, nsigma, -2. , 2.);
	    sbeta_res = fbeta->GetParameter("Sigma");
	    mbeta_res = fbeta->GetParameter("Mean");
	  }
	  
	  TH1F* tmpAlpha = (TH1F*) hRes4D->hResAlpha->Clone("tmpAlpha");
	  // tmpAlpha->Rebin(2);
	  TF1* falpha = drawGFit(tmpAlpha, nsigma, -2. , 2. );
	  salpha_res = falpha->GetParameter("Sigma");
	  malpha_res = falpha->GetParameter("Mean");

	  cout << " " << wheel << " " << station << " " << sector  <<  " salpha_res: " << salpha_res  << " malpha_res: " << malpha_res;
	  cout  << " sbeta_res: " << sbeta_res  << " mbeta_res: " << mbeta_res;
	  cout << endl;
	 	  
	}
	
	// Segment angle pull
	if (doPullabxy) {
	  if (station!=4) {
	    TH1F* tmpBeta = (TH1F*) hRes4D->hPullBetaRZ->Clone("tmpBeta");
	    tmpBeta->Rebin(2);
	    TF1* fbeta = drawGFit(tmpBeta, nsigma, -2. , 2.);
	    sbeta_pull = fbeta->GetParameter("Sigma");
	    mbeta_pull = fbeta->GetParameter("Mean");
	    
	    TH1F* tmpY = (TH1F*) hRes4D->hPullYRZ->Clone("tmpY");
	    tmpY->Rebin(2);
	    TF1* fy = drawGFit(tmpY, nsigma, -2. , 2. );
	    sy_pull = fy->GetParameter("Sigma");
	    my_pull = fy->GetParameter("Mean");
	  }
	  
	  TH1F* tmpAlpha = (TH1F*) hRes4D->hPullAlpha->Clone("tmpAlpha");
	  tmpAlpha->Rebin(2);
	  TF1* falpha = drawGFit(tmpAlpha, nsigma, -2. , 2. );
	  salpha_pull = falpha->GetParameter("Sigma");
	  malpha_pull = falpha->GetParameter("Mean");
	    
	  TH1F* tmpX = (TH1F*) hRes4D->hPullX->Clone("tmpX");
	  tmpX->Rebin(2);
	  TF1* fx = drawGFit(tmpX, nsigma, -2. , 2.);
	  sx_pull = fx->GetParameter("Sigma");
	  mx_pull = fx->GetParameter("Mean");
	    
	    
	  cout << " " << wheel << " " << station << " " << sector  <<  " salpha_pull: " << salpha_pull  << " malpha_pull: " << malpha_pull;
	  cout  << " sbeta_pull: " << sbeta_pull  << " mbeta_pull: " << mbeta_pull;
	  cout  << " sx_pull: " << sx_pull  << " mx_pull: " << mx_pull;     
	  cout  << " sy_pull: " << sy_pull  << " my_pull: " << my_pull;
	  cout << endl;
	  
	}
	

	double ratio=0;

	if (doNSeg) {
	  TH1F* hNs =  hEff4D->hNSeg;
	  double int_1 = hNs->Integral(2,21);
	  double int_2 = hNs->Integral(3,21);
	  ratio = int_2/int_1;
	  //  cout << "int_1: " << int_1 <<" int_2: " << int_2 << " ratio: " << ratio << endl;
	}



	float p0_Phi=0;
	float p1_Phi=0;
	float p2_Phi=0;
	float p0_Theta=0;
	float p1_Theta=0;
	float p2_Theta=0;

	if(doPar){
	  float min_Phi;
	  float max_Phi;
	  float min_Theta;
	  float max_Theta;

	  rangeAngle(abs(wheel), station, 1, min_Phi, max_Phi);	  
	  TF1 *angleDep_Phi= new TF1("angleDep_Phi","[0]*cos(x+[2])+[1]", min_Phi, max_Phi);
	  angleDep_Phi->SetParameters(0.01,0.001, 0.1);
  
	  TH1F* hprof_Phi = plotAndProfileX(hResPhi1->hResVsAngle,1,1,1,-0.04, 0.04, min_Phi-0.03, max_Phi+0.03);
	  if((station!=1 ||(station==1 && wheel==0))) angleDep_Phi->FixParameter(2,0.);
	  if((abs(wheel) == 1 ||wheel == -2) && station == 1)  angleDep_Phi->SetParameter(2,-0.12);
  	  
	  hprof_Phi->Fit(angleDep_Phi,"RQN"); 
	  
	  angleDep_Phi->Draw("same");
	  p0_Phi = angleDep_Phi->GetParameter(0);
	  p1_Phi = angleDep_Phi->GetParameter(1);
	  p2_Phi = angleDep_Phi->GetParameter(2);

 	  if (station!=4){
	    rangeAngle(abs(wheel), station, 2, min_Theta, max_Theta);	  
	    TF1 *angleDep_Theta= new TF1("angleDep_Theta","[0]*cos(x+[2])+[1]", min_Theta, max_Theta);
	    angleDep_Theta->SetParameters(0.01,0.001, 0.1);
	    angleDep_Theta->FixParameter(2,0.);
	    TH1F* hprof_Theta = plotAndProfileX(hResTheta->hResVsAngle,1,1,1,-0.04, 0.04, min_Theta-0.03, max_Theta+0.03);
	    
	    if((wheel==-2 || wheel==2) &&(station == 1 || station == 2)) { 
	      angleDep_Theta->FixParameter(0,0.);
	      angleDep_Theta->FixParameter(1,0.);
	    }
		    
	    hprof_Theta->Fit(angleDep_Theta,"RQN"); 
	    angleDep_Theta->Draw("same");
	    //angleDep_Theta->Draw("same");
	    p0_Theta = angleDep_Theta->GetParameter(0);
	    p1_Theta = angleDep_Theta->GetParameter(1);
	    p2_Theta = angleDep_Theta->GetParameter(2);

	  }
	}


	// Write summary table (one row per SL)
 	int secmin=sector;
 	int secmax=sector;
 	if (sector ==0) {
 	  secmin=1;
 	  secmax=14;  
 	}
 	for (int sec = secmin; sec<=secmax; sec++) {
 	  if (station!=4 && sec>12) continue;
	  f                 << wheel << " " << station << " " << sec << " " << 1 << " " << effS1RPhi << " " << effS3RPhi << " " << effSeg << " " << sphi   << " " << pphi << " " << malpha_res <<  " " <<salpha_res  << " " << malpha_pull <<  " " <<salpha_pull << " " << mx_pull <<  " " <<sx_pull << " " << mx << " " << sx << " " <<ratio << " " << p0_Phi << " " << p1_Phi << " " << p2_Phi << endl;
	  if (station!=4) f << wheel << " " << station << " " << sec << " " << 2 << " " << effS1RZ   << " " << effS3RZ   << " " <<  effSeg <<" " << stheta << " " << ptheta << " " << mbeta_res <<  " " <<sbeta_res  << " " << mbeta_pull <<  " " <<sbeta_pull << " " << my_pull <<  " " <<sy_pull << " " << my << " " << sy << " " <<ratio <<" " << p0_Theta << " " << p1_Theta << " " << p2_Theta <<endl;
	  f                 << wheel << " " << station << " " << sec << " " << 3 << " " << effS1RPhi << " " << effS3RPhi << " " << effSeg << " " << sphi   << " " << pphi << " " << malpha_res <<  " " <<salpha_res  << " " << malpha_pull <<  " " <<salpha_pull  << " " << mx_pull <<  " " <<sx_pull << " " << mx << " " << sx <<  " " <<ratio << " " << p0_Phi  << " " << p1_Phi << " " << p2_Phi <<endl;
 	}
      } // sector
    } //station
  } //wheel
}
