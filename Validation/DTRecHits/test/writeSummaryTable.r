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
  cout << endl << "Usage: .x writeSummaryTable.r(\"inputFile.root\",doEff,doRes,doPull)" << endl << endl;
}


void writeSummaryTable(TString filename, bool doEff=true, bool doRes=true, bool doPull=true) {

  //----------------------------------------------------------------------
  // Configurable options  

  // Interval for the fit of residual distributions
  float nsigma = 2.;

  //----------------------------------------------------------------------

  if (! TString(gSystem->GetLibraries()).Contains("Histograms_h")) {
    gROOT->LoadMacro("$CMSSW_BASE/src/Validation/DTRecHits/test/Histograms.h+");
    gROOT->LoadMacro("macros.C");
  }

  float cmToMicron = 10000.;

  TFile *file = new TFile(filename);

  TString table = filename.Replace(filename.Length()-5,5,"_summary.txt");
  ofstream f(table,ios_base::out);
  f << fixed;  

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
	
	float effS1RPhi=0.;
	float effS3RPhi=0.;
	float effS1RZ=0.;
	float effS3RZ=0.;

	HRes1DHit *hResPhi1  = new HRes1DHit(file, wheel, station, 1, "S3");
	HRes1DHit *hResTheta = new HRes1DHit(file, wheel, station, 2, "S3");

	HEff1DHit* hEffS1RPhi= new HEff1DHit(file, wheel, station, 1, "S1");
	HEff1DHit* hEffS3RPhi= new HEff1DHit(file, wheel, station, 1, "S3");
	HEff1DHit* hEffS1RZ=0;
	HEff1DHit* hEffS3RZ=0;
	if (station!=4) {
	  hEffS1RZ=   new HEff1DHit(file, wheel, station, 2, "S1");
	  hEffS3RZ=   new HEff1DHit(file, wheel, station, 2, "S3");
	}



	TCanvas c;

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
	
	
	if (doRes) {
	  if (station!=4) {
	    TH1F* tmpTheta = (TH1F*) hResTheta->hRes->Clone("tmpTheta");
	    TF1* ftheta = drawGFit(tmpTheta, nsigma, -2. , 2.);
	    stheta = ftheta->GetParameter("Sigma");
	  }
	  TH1F* tmpPhi = (TH1F*) hResPhi1->hRes->Clone("tmpPhi");
	  TF1* fphi = drawGFit(tmpPhi, nsigma, -2. , 2. );
	  sphi = fphi->GetParameter("Sigma");

	  cout << " " << wheel << " " << station << " " << sector  <<  " sphi: " << sphi;
	  if (station!=4) cout  << " stheta: " << stheta ;
	  cout << endl;
	}
	

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


	// Write summary table (one row per SL)
 	int secmin=sector;
 	int secmax=sector;
 	if (sector ==0) {
 	  secmin=1;
 	  secmax=14;  
 	}
 	for (int sec = secmin; sec<=secmax; sec++) {
 	  if (station!=4 && sec>12) continue;
	  f                 << wheel << " " << station << " " << sec << " " << 1 << " " << effS1RPhi << " " << effS3RPhi << " " << sphi   << " " << pphi << endl;
	  if (station!=4) f << wheel << " " << station << " " << sec << " " << 2 << " " << effS1RZ   << " " << effS3RZ   << " " << stheta << " " << ptheta <<endl;
	  f                 << wheel << " " << station << " " << sec << " " << 3 << " " << effS1RPhi << " " << effS3RPhi << " " << sphi   << " " << pphi << endl;
 	}
      } // sector
    } //station
  } //wheel
}
