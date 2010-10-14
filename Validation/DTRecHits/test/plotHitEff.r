
/*
 * Format plots of efficiency produced by DTRecHits validation package.
 * The root tree containing the histograms must be already open when 
 * executing this macro
 * 
 * G. Cerminara 2006
 */

class HEff1DHit;
class hEff2DHit;
class hEff4DHit;

void plotHitEff();
void plotWWWHitEff();
void plot1DEffVsPos(HEff1DHit *hS1, HEff1DHit *hS2, HEff1DHit *hS3, bool ThreeIn1);
void plot1DEffVsEta(HEff1DHit *hS1, HEff1DHit *hS2, HEff1DHit *hS3, bool ThreeIn1);
void plot1DEffVsPhi(HEff1DHit *hS1, HEff1DHit *hS2, HEff1DHit *hS3, bool ThreeIn1);
void plotHisto(TH1F *h1, TH1F *h2, bool ThreeIn1);

bool setPreferences(bool& do1DRecHit, bool& do2DRecHit, bool& do2DSLPhiRecHit, bool& do4DRecHit, bool& ThreeIn1);
// 
void plotHitEff(){
  // Load needed macros and files
  gROOT->LoadMacro("macros.C");     // Load service macros
  gROOT->LoadMacro("../src/Histograms.h"); // Load definition of histograms

  // Get the style
  TStyle * style = getStyle();

  //Main switches
  bool do1DRecHit = false; 
  bool do2DRecHit = false; 
  bool do2DSLPhiRecHit = false; 
  bool do4DRecHit = false; 
  bool ThreeIn1 = false;  // Plot the 3 steps in a single canvas (where appl.)

  //--------------------------------------------------------------------------------------
  //-------------------- Set your preferences here ---------------------------------------

  // What plots should be produced:
  while(!setPreferences(do1DRecHit, do2DRecHit, do2DSLPhiRecHit, do4DRecHit, ThreeIn1));
  //   do1DRecHit = true; 
  //   do2DRecHit = true; 
  //   do2DSLPhiRecHit = true; 
  //   do4DRecHit = true; 



  int form = 2;          // Form factor of the canvases (where applicable)
                         //       1. For rectangular shape
                         //       2. For squared shape


  // Style options
  //  style->SetOptStat("OURMEN");
  style->SetOptStat("RME");
  style->SetOptFit(101);

  //--------------------------------------------------------------------------------------

  // Check the choice
  if(!do1DRecHit && !do2DRecHit && !do2DSLPhiRecHit && !do4DRecHit) {
    cout << "[plotHitEff]***Error: Nothing to do! Set do1DRecHit, do2DRecHit, do2DSLPhiRecHit, do4DRecHit correctly!"
	 << endl;
    return;
  }
  style->cd();                      // Apply style 

  draw(do1DRecHit, do2DRecHit, do2DSLPhiRecHit, do4DRecHit, ThreeIn1, form);
}


void plotWWWHitEff() {
  // Load needed macros and files
  gROOT->LoadMacro("macros.C");     // Load service macros
  gROOT->LoadMacro("../src/Histograms.h"); // Load definition of histograms
  
  // Get the style
  TStyle * style = getStyle();
  
  //Main switches
  bool do1DRecHit = true; 
  bool do2DRecHit = false; 
  bool do2DSLPhiRecHit = false; 
  bool do4DRecHit = false; 
  bool ThreeIn1 = false;  // Plot the 3 steps in a single canvas (where appl.)

  int form = 2;          // Form factor of the canvases (where applicable)
                         //       1. For rectangular shape
                         //       2. For squared shape


  // Style options
  //  style->SetOptStat("OURMEN");
  style->SetOptStat("RME");
  style->SetOptFit(101);

  //--------------------------------------------------------------------------------------

  // Check the choice
  if(!do1DRecHit && !do2DRecHit && !do2DSLPhiRecHit && !do4DRecHit) {
    cout << "[plotHitReso]***Error: Nothing to do! Set do1DRecHit, do2DRecHit, do2DSLPhiRecHit, do4DRecHit correctly!"
	 << endl;
    return;
  }
  style->cd();                      // Apply style 


  draw(do1DRecHit, do2DRecHit, do2DSLPhiRecHit, do4DRecHit, ThreeIn1, form);
  cout << "Set the name of the www directory: " << endl;
  TString nameS;
  cin >> nameS;

  //gSystem->MakeDirectory("/afs/cern.ch/user/c/cerminar/www/DTLocalRecoQualityTest/"+nameS);
  gSystem->ChangeDirectory("/afs/cern.ch/user/c/cerminar/www/DTLocalRecoQualityTest/"+nameS);


  printCanvases("gif");
}



void draw(bool do1DRecHit, bool do2DRecHit, bool do2DSLPhiRecHit, bool do4DRecHit, bool ThreeIn1, int form) {
  
  // Retrieve histogram sets
  TFile *f = gROOT->GetListOfFiles()->Last();
  cout << "Loading file: " << f->GetName() << endl;

  if(do1DRecHit) {
    HEff1DHit *hN_S1RPhi = new HEff1DHit("S1RPhi", f);     // RecHits, 1. step, RPhi
    HEff1DHit *hN_S2RPhi = new HEff1DHit("S2RPhi", f);     // RecHits, 2. step, RPhi
    HEff1DHit *hN_S3RPhi = new HEff1DHit("S3RPhi", f);     // RecHits, 3. step, RPhi

    HEff1DHit *hN_S1RZ = new HEff1DHit("S1RZ", f);         // RecHits, 1. step, RZ
    HEff1DHit *hN_S2RZ = new HEff1DHit("S2RZ", f);	    // RecHits, 2. step, RZ
    HEff1DHit *hN_S3RZ = new HEff1DHit("S3RZ", f);	    // RecHits, 3. step, RZ

    HEff1DHit *hN_S1RZ_W0 = new HEff1DHit("S1RZ_W0", f);   // RecHits, 1. step, RZ, wheel 0
    HEff1DHit *hN_S2RZ_W0 = new HEff1DHit("S2RZ_W0", f);   // RecHits, 2. step, RZ, wheel 0
    HEff1DHit *hN_S3RZ_W0 = new HEff1DHit("S3RZ_W0", f);   // RecHits, 3. step, RZ, wheel 0

    HEff1DHit *hN_S1RZ_W1 = new HEff1DHit("S1RZ_W1", f);   // RecHits, 1. step, RZ, wheel +-1
    HEff1DHit *hN_S2RZ_W1 = new HEff1DHit("S2RZ_W1", f);   // RecHits, 2. step, RZ, wheel +-1
    HEff1DHit *hN_S3RZ_W1 = new HEff1DHit("S3RZ_W1", f);   // RecHits, 3. step, RZ, wheel +-1

    HEff1DHit *hN_S1RZ_W2 = new HEff1DHit("S1RZ_W2", f);   // RecHits, 1. step, RZ, wheel +-2
    HEff1DHit *hN_S2RZ_W2 = new HEff1DHit("S2RZ_W2", f);   // RecHits, 2. step, RZ, wheel +-2
    HEff1DHit *hN_S3RZ_W2 = new HEff1DHit("S3RZ_W2", f);   // RecHits, 3. step, RZ, wheel +-2

  }

  if(do2DRecHit) {
    HEff2DHit *h2DHitEff_RPhi = new HEff2DHit("RPhi", f);
    HEff2DHit *h2DHitEff_RZ = new HEff2DHit("RZ", f);
    HEff2DHit *h2DHitEff_RZ_W0 = new HEff2DHit("RZ_W0", f);
    HEff2DHit *h2DHitEff_RZ_W1 = new HEff2DHit("RZ_W1", f);
    HEff2DHit *h2DHitEff_RZ_W2 = new HEff2DHit("RZ_W2", f);
  }

  if(do2DSLPhiRecHit) {
    HEff2DHit *h2DSLPhiHitEff = new HEff2DHit("SuperPhi", f);
  }

  if(do4DRecHit) {
    HEff4DHit *hEff_All = new HEff4DHit("All", f);
    HEff4DHit *hEff_W0 = new  HEff4DHit("W0", f);
    HEff4DHit *hEff_W1 = new  HEff4DHit("W1", f);
    HEff4DHit *hEff_W2 = new  HEff4DHit("W2", f);
  }


  TCanvas * c1;
  int i = 1;

  
 eff:

    
  if(do1DRecHit) {
    plot1DEffVsPos(hN_S1RPhi, hN_S2RPhi, hN_S3RPhi, ThreeIn1);
    plot1DEffVsEta(hN_S1RPhi, hN_S2RPhi, hN_S3RPhi, ThreeIn1);
    plot1DEffVsPhi(hN_S1RPhi, hN_S2RPhi, hN_S3RPhi, ThreeIn1);

    plot1DEffVsPos(hN_S1RZ, hN_S2RZ, hN_S3RZ, ThreeIn1);
    plot1DEffVsEta(hN_S1RZ, hN_S2RZ, hN_S3RZ, ThreeIn1);
    plot1DEffVsPhi(hN_S1RZ, hN_S2RZ, hN_S3RZ, ThreeIn1);

    plot1DEffVsPos(hN_S1RZ_W0, hN_S2RZ_W0, hN_S3RZ_W0, ThreeIn1);
    plot1DEffVsEta(hN_S1RZ_W0, hN_S2RZ_W0, hN_S3RZ_W0, ThreeIn1);
    plot1DEffVsPhi(hN_S1RZ_W0, hN_S2RZ_W0, hN_S3RZ_W0, ThreeIn1);

    plot1DEffVsPos(hN_S1RZ_W1, hN_S2RZ_W1, hN_S3RZ_W1, ThreeIn1);
    plot1DEffVsEta(hN_S1RZ_W1, hN_S2RZ_W1, hN_S3RZ_W1, ThreeIn1);
    plot1DEffVsPhi(hN_S1RZ_W1, hN_S2RZ_W1, hN_S3RZ_W1, ThreeIn1);

    plot1DEffVsPos(hN_S1RZ_W2, hN_S2RZ_W2, hN_S3RZ_W2, ThreeIn1);
    plot1DEffVsEta(hN_S1RZ_W2, hN_S2RZ_W2, hN_S3RZ_W2, ThreeIn1);
    plot1DEffVsPhi(hN_S1RZ_W2, hN_S2RZ_W2, hN_S3RZ_W2, ThreeIn1);
  }


  if(do2DRecHit) {
    ThreeIn1 = false;
    plotHisto(h2DHitEff_RPhi->hEffVsEta, h2DHitEff_RPhi->hEffVsPhi, ThreeIn1);
    plotHisto(h2DHitEff_RZ->hEffVsEta, h2DHitEff_RZ->hEffVsPhi, ThreeIn1);
    plotHisto(h2DHitEff_RZ_W0->hEffVsEta, h2DHitEff_RZ_W0->hEffVsPhi, ThreeIn1);
    plotHisto(h2DHitEff_RZ_W1->hEffVsEta, h2DHitEff_RZ_W1->hEffVsPhi, ThreeIn1);
    plotHisto(h2DHitEff_RZ_W2->hEffVsEta, h2DHitEff_RZ_W2->hEffVsPhi, ThreeIn1);

    plotHisto(h2DHitEff_RPhi->hEffVsPos, h2DHitEff_RPhi->hEffVsAngle, ThreeIn1);
    plotHisto(h2DHitEff_RZ->hEffVsPos, h2DHitEff_RZ->hEffVsAngle, ThreeIn1);
    plotHisto(h2DHitEff_RZ_W0->hEffVsPos, h2DHitEff_RZ_W0->hEffVsAngle, ThreeIn1);
    plotHisto(h2DHitEff_RZ_W1->hEffVsPos, h2DHitEff_RZ_W1->hEffVsAngle, ThreeIn1);
    plotHisto(h2DHitEff_RZ_W2->hEffVsPos, h2DHitEff_RZ_W2->hEffVsAngle, ThreeIn1);

  }

  if(do2DSLPhiRecHit) {
    ThreeIn1 = false;
    plotHisto(h2DSLPhiHitEff->hEffVsEta, h2DSLPhiHitEff->hEffVsPhi, ThreeIn1);
    plotHisto(h2DSLPhiHitEff->hEffVsPos, h2DSLPhiHitEff->hEffVsAngle, ThreeIn1);
 }

  if(do4DRecHit) {
    ThreeIn1 = false;
    plotHisto(hEff_All->hEffVsEta, hEff_All->hEffVsPhi, ThreeIn1);
    plotHisto(hEff_W0->hEffVsEta, hEff_W0->hEffVsPhi, ThreeIn1);
    plotHisto(hEff_W1->hEffVsEta, hEff_W1->hEffVsPhi, ThreeIn1);
    plotHisto(hEff_W2->hEffVsEta, hEff_W2->hEffVsPhi, ThreeIn1);

    plotHisto(hEff_All->hEffVsX, hEff_All->hEffVsY, ThreeIn1);
    plotHisto(hEff_W0->hEffVsX, hEff_W0->hEffVsY, ThreeIn1);
    plotHisto(hEff_W1->hEffVsX, hEff_W1->hEffVsY, ThreeIn1);
    plotHisto(hEff_W2->hEffVsX, hEff_W2->hEffVsY, ThreeIn1);

    plotHisto(hEff_All->hEffVsAlpha, hEff_All->hEffVsBeta, ThreeIn1);
    plotHisto(hEff_W0->hEffVsAlpha, hEff_W0->hEffVsBeta, ThreeIn1);
    plotHisto(hEff_W1->hEffVsAlpha, hEff_W1->hEffVsBeta, ThreeIn1);
    plotHisto(hEff_W2->hEffVsAlpha, hEff_W2->hEffVsBeta, ThreeIn1);
    
  }




  return;

 
 end:

}


void plot1DEffVsPos(HEff1DHit *hS1, HEff1DHit *hS2, HEff1DHit *hS3, bool ThreeIn1) {
  int i = 2;

  TString N1 = "c_1D_" + hS1->name;
  TString N2 = "c_1D_" + hS2->name;
  TString N3 = "c_1D_" + hS3->name;
  int form = 2;
  if (ThreeIn1) c1 = newCanvas(N1.Remove(5,2)+"_hEffVsDist",3,1,1200,500);
  else newCanvas(N1+"_hEffVsDist",form);
  hS1->hEffVsDist->Draw("h");

  if (ThreeIn1) c1->cd(i++);
  else c1 = newCanvas(N2+"_hEffVsDist",form);
  hS2->hEffVsDist->Draw("h");

  if (ThreeIn1) c1->cd(i++);
  else c1 = newCanvas(N3+"_hEffVsDist",form);
  hS3->hEffVsDist->Draw("h");

}


void plot1DEffVsEta(HEff1DHit *hS1, HEff1DHit *hS2, HEff1DHit *hS3, bool ThreeIn1) {
  int i = 2;

  TString N1 = "c_1D_" + hS1->name;
  TString N2 = "c_1D_" + hS2->name;
  TString N3 = "c_1D_" + hS3->name;
  int form = 2;
  if (ThreeIn1) c1 = newCanvas(N1.Remove(5,2)+"_hEffVsEta",3,1,1200,500);
  else newCanvas(N1+"_hEffVsEta",form);
  hS1->hEffVsEta->Draw("h");

  if (ThreeIn1) c1->cd(i++);
  else c1 = newCanvas(N2+"_hEffVsEta",form);
  hS2->hEffVsEta->Draw("h");

  if (ThreeIn1) c1->cd(i++);
  else c1 = newCanvas(N3+"_hEffVsEta",form);
  hS3->hEffVsEta->Draw("h");

}


void plot1DEffVsPhi(HEff1DHit *hS1, HEff1DHit *hS2, HEff1DHit *hS3, bool ThreeIn1) {
  int i = 2;

  TString N1 = "c_1D_" + hS1->name;
  TString N2 = "c_1D_" + hS2->name;
  TString N3 = "c_1D_" + hS3->name;
  int form = 2;
  if (ThreeIn1) c1 = newCanvas(N1.Remove(5,2)+"_hEffVsPhi",3,1,1200,500);
  else newCanvas(N1+"_hEffVsPhi",form);
  hS1->hEffVsPhi->Draw("h");

  if (ThreeIn1) c1->cd(i++);
  else c1 = newCanvas(N2+"_hEffVsPhi",form);
  hS2->hEffVsPhi->Draw("h");

  if (ThreeIn1) c1->cd(i++);
  else c1 = newCanvas(N3+"_hEffVsPhi",form);
  hS3->hEffVsPhi->Draw("h");

}


void plotHisto(TH1F *h1, TH1F *h2, bool ThreeIn1) {
  int i = 2;
  
  TString N1 = "c_" + TString(h1->GetName());
  TString N2 = "c_" + TString(h2->GetName());
  int form = 2;
  if (ThreeIn1) c1 = newCanvas(N1 ,2,1,800,500);
  else newCanvas(N1, form);
  h1->Draw("h");

  if (ThreeIn1) c1->cd(i++);
  else c1 = newCanvas(N2, form);
  h2->Draw("h");
}




bool setPreferences(bool& do1DRecHit, bool& do2DRecHit, bool& do2DSLPhiRecHit, bool& do4DRecHit, bool& ThreeIn1) {



  int dimension = 0;

  cout << "===================================================" << endl;
  cout << "==== plotHitEff User Menu =====================" << endl;
  cout << "Chose the plot you want to produce:" << endl;
  cout << "1 - 1D RecHit Plots" << endl;
  cout << "2 - 2D RecHit Plots" << endl;
  cout << "3 - 2D RecHit Plots (only SLPhi from 4D RecHit)" << endl;
  cout << "4 - 4D RecHit Plots" << endl;
  cout << "-->";
  cin >> dimension;

  switch(dimension) 
    {
    case 1:
      {
	do1DRecHit = true;
	break;
      }
    case 2:
      {
	do2DRecHit = true;
	break;
      }
    case 3:
      {
	do2DSLPhiRecHit = true;
	break;
      }
    case 4:
      {
	do4DRecHit = true;
	break;
      }
    default:
      {
	cout << "Error: option not Valid, try again!" << endl;
	return false;
	break;
      }
    }

  int threeInOne = 0;

  cout << "" << endl;
  cout << "Do you want to Plot three histos in one?" << endl;
  cout << "0 - No" << endl;
  cout << "1 - Yes" << endl;
  cout << "-->";

  cin >> threeInOne;

  switch(threeInOne) 
    {
    case 0:
      {
	ThreeIn1 = false;
	break;
      }
    case 1:
      {
	ThreeIn1 = true;
	break;
      }
    default:
      {
	cout << "Not a valid option: default used!" << endl;	
	ThreeIn1 = true;
	break;
      }
    }

  return true;
}

