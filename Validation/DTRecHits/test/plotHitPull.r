
/*
 * Format plots of Pulllutions, etc. produced by DTRecHits validation.
 * The root tree containing the histograms must be already open when 
 * executing this macro.
 * 
 * G. Cerminara 2004
 */

// class hRHit;
class HRes1DHit;
class HRes2DHit;
class HRes4DHit;

void plotHitPull();
void plotWWWHitPull(int dimSwitch = 1, TString nameDir = "");
void draw(bool do1DRecHit, bool do2DRecHit, bool do2DSLPhiRecHit, bool do4DRecHit, bool ThreeIn1, int form);
void plot1DResiduals(HRes1DHit * h1, HRes1DHit * h2, HRes1DHit * h3, bool ThreeIn1);
void plot1DResidualsVsPos(TString name, HRes1DHit * h1, HRes1DHit * h2, HRes1DHit * h3, bool ThreeIn1) ;
void plot2DResiduals(HRes2DHit * h1);

void plot4DResiduals(HRes4DHit * h1);
void plot4DPullVsEta(HRes4DHit * h1);
void plot4DPullVsPhi(HRes4DHit * h1);

void plot2DAngles(HRes2DHit * h1, bool ThreeIn1);
void plot4DAngles(HRes4DHit * h1, bool ThreeIn1);

// Read user input
bool setPreferences(bool& do1DRecHit, bool& do2DRecHit, bool& do2DSLPhiRecHit, bool& do4DRecHit, bool& ThreeIn1);
// 




// This is the main function
void plotHitPull(){
  // Load needed macros and files
  gROOT->LoadMacro("macros.C");     // Load service macros
  gROOT->LoadMacro("../plugins/Histograms.h"); // Load definition of histograms

  // Get the style
  TStyle * style = getStyle("tdr");
  /// TStyle * style = getStyle();

  //Main switches
  bool do1DRecHit = false; 
  bool do2DRecHit = false; 
  bool do2DSLPhiRecHit = false; 
  bool do4DRecHit = false; 
  bool ThreeIn1 = true;  // Plot the 3 steps in a single canvas (where appl.)

  //--------------------------------------------------------------------------------------
  //-------------------- Set your preferences here ---------------------------------------

  // Read user input, namely what plots should be produced

  while(!setPreferences(do1DRecHit, do2DRecHit, do2DSLPhiRecHit, do4DRecHit, ThreeIn1));
  //   do1DRecHit = true; 
  //   do2DRecHit = true; 
  //   do4DRecHit = true; 


  int form = 2;          // Form factor of the canvases (where applicable)
  //       1. For rectangular shape
  //       2. For squared shape


  // Style options
  //  style->SetOptStat("OURMEN");
  // style->SetOptStat("RME");
  style->SetOptStat(0);
  style->SetFitFormat("4.2g");
  style->SetOptFit(11);

  //--------------------------------------------------------------------------------------

  // Check the choice
  if(!do1DRecHit && !do2DRecHit && !do2DSLPhiRecHit && !do4DRecHit) {
    cout << "[plotHitPull]***Error: Nothing to do! Set do1DRecHit, do2DRecHit, do4DRecHit correctly!"
      << endl;
    return;
  }
  style->cd();                      // Apply style 

  draw(do1DRecHit, do2DRecHit, do2DSLPhiRecHit, do4DRecHit, ThreeIn1, form);

}

void plotWWWHitPull(int dimSwitch, TString nameDir) {
  // Load needed macros and files
  gROOT->LoadMacro("macros.C");     // Load service macros
  gROOT->LoadMacro("../plugins/Histograms.h"); // Load definition of histograms

  // Get the style
  TStyle * style = getStyle();

  //Main switches
  bool do1DRecHit = false; 
  bool do2DRecHit = false; 
  bool do2DSLPhiRecHit = false; 
  bool do4DRecHit = false; 

  if(dimSwitch == 1) {
    do1DRecHit = true;
  } else if(dimSwitch == 2) {
    do2DRecHit = true;
  } else if(dimSwitch == 3) {
    do2DSLPhiRecHit = true;
  } else if(dimSwitch == 4) {
    do4DRecHit = true;
  } else {
    cout << "Not a valid option!" << endl;
    return;
  }



  bool ThreeIn1 = true;  // Plot the 3 steps in a single canvas (where appl.)

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
    cout << "[plotHitPull]***Error: Nothing to do! Set do1DRecHit, do2DRecHit, do4DRecHit correctly!"
      << endl;
    return;
  }
  style->cd();                      // Apply style 


  draw(do1DRecHit, do2DRecHit, do2DSLPhiRecHit, do4DRecHit, ThreeIn1, form);
  TString nameS;
  if(nameDir == "") {
    cout << "Set the name of the www directory: " << endl;
    cin >> nameS;
  } else {
    nameS = nameDir;
  }
  TString pwd = gSystem->WorkingDirectory();
  gSystem->MakeDirectory("/afs/cern.ch/cms/Physics/muon/CMSSW/Performance/DT/DTLocalRecoQualityTest/"+nameS);
  gSystem->ChangeDirectory("/afs/cern.ch/cms/Physics/muon/CMSSW/Performance/DT/DTLocalRecoQualityTest/"+nameS);


  printCanvases(".gif");
  gSystem->ChangeDirectory(pwd.Data());

}


void draw(bool do1DRecHit, bool do2DRecHit, bool do2DSLPhiRecHit, bool do4DRecHit, bool ThreeIn1, int form) {
  // Retrieve histogram sets
  TFile *f = gROOT->GetListOfFiles()->Last();
  cout << "Loading file: " << f->GetName() << endl;

  if(do1DRecHit) {
    HRes1DHit *h1RPhi = new HRes1DHit("S1RPhi",f);     // RecHits, 1. step, RPhi
    HRes1DHit *h2RPhi = new HRes1DHit("S2RPhi",f);     // RecHits, 2. step, RPhi
    HRes1DHit *h3RPhi = new HRes1DHit("S3RPhi",f);     // RecHits, 3. step, RPhi

    HRes1DHit *h1RZ = new HRes1DHit("S1RZ",f);         // RecHits, 1. step, RZ
    HRes1DHit *h2RZ = new HRes1DHit("S2RZ",f);	    // RecHits, 2. step, RZ
    HRes1DHit *h3RZ = new HRes1DHit("S3RZ",f);	    // RecHits, 3. step, RZ

    HRes1DHit *h1RZ_W0 = new HRes1DHit("S1RZ_W0",f);   // RecHits, 1. step, RZ, wheel 0
    HRes1DHit *h2RZ_W0 = new HRes1DHit("S2RZ_W0",f);   // RecHits, 2. step, RZ, wheel 0
    HRes1DHit *h3RZ_W0 = new HRes1DHit("S3RZ_W0",f);   // RecHits, 3. step, RZ, wheel 0

    HRes1DHit *h1RZ_W1 = new HRes1DHit("S1RZ_W1",f);   // RecHits, 1. step, RZ, wheel +-1
    HRes1DHit *h2RZ_W1 = new HRes1DHit("S2RZ_W1",f);   // RecHits, 2. step, RZ, wheel +-1
    HRes1DHit *h3RZ_W1 = new HRes1DHit("S3RZ_W1",f);   // RecHits, 3. step, RZ, wheel +-1

    HRes1DHit *h1RZ_W2 = new HRes1DHit("S1RZ_W2",f);   // RecHits, 1. step, RZ, wheel +-2
    HRes1DHit *h2RZ_W2 = new HRes1DHit("S2RZ_W2",f);   // RecHits, 2. step, RZ, wheel +-2
    HRes1DHit *h3RZ_W2 = new HRes1DHit("S3RZ_W2",f);   // RecHits, 3. step, RZ, wheel +-2

  }

  if(do2DRecHit) {
    HRes2DHit *h2DHitRPhi = new HRes2DHit("RPhi",f);
    HRes2DHit *h2DHitRZ = new HRes2DHit("RZ",f);
    HRes2DHit *h2DHitRZ_W0 = new HRes2DHit("RZ_W0",f);
    HRes2DHit *h2DHitRZ_W1 = new HRes2DHit("RZ_W1",f);
    HRes2DHit *h2DHitRZ_W2 = new HRes2DHit("RZ_W2",f);
  }
  if(do2DSLPhiRecHit) {
    HRes2DHit *h2DSLPhiHit = new HRes2DHit("SuperPhi",f);
  }

  if(do4DRecHit) {
    HRes4DHit *h4DHit = new HRes4DHit("All", f);
    HRes4DHit *h4DHit_W0 = new HRes4DHit("W0", f);
    HRes4DHit *h4DHit_W1 = new HRes4DHit("W1", f);
    HRes4DHit *h4DHit_W2 = new HRes4DHit("W2", f);
  }


  TCanvas * c1;
  int i = 1;


  if(do1DRecHit) {
    // Residual, Rphi
    plot1DResiduals(h1RPhi,h2RPhi,h3RPhi,ThreeIn1);

    // Residual, RZ 
    plot1DResiduals(h1RZ,h2RZ,h3RZ,ThreeIn1);

    // Residual, RZ, per wheel
    plot1DResiduals(h1RZ_W0,h2RZ_W0,h3RZ_W0,ThreeIn1);
    plot1DResiduals(h1RZ_W1,h2RZ_W1,h3RZ_W1,ThreeIn1);
    plot1DResiduals(h1RZ_W2,h2RZ_W2,h3RZ_W2,ThreeIn1);
  }

  if(do2DRecHit) {
    cout << "h2DHitRPhi " << h2DHitRPhi << endl;
    plot2DResiduals(h2DHitRPhi);
    plot2DResiduals(h2DHitRZ);
    // plot2DResiduals(h2DHitRZ_W0);
    // plot2DResiduals(h2DHitRZ_W1);
    // plot2DResiduals(h2DHitRZ_W2);
  }

  if(do2DSLPhiRecHit) {
    plot2DResiduals(h2DSLPhiHit);
  }

  if(false && do4DRecHit) {
    plot4DResiduals(h4DHit);
    plot4DResidualsRZ(h4DHit);
    // plot4DResiduals(h4DHit_W0);
    // plot4DResiduals(h4DHit_W1);
    // plot4DResiduals(h4DHit_W2);
  }

  // Pullvseta:

  // if(do1DRecHit) {
  //   const float min = -0.6;
  //   const float max =  0.6;
  //   c1 = newCanvas("c_1D_S1RPhi_hPullVsEta",form);
  //   plotAndProfileXSpread(h1RPhi->hPullVsEta,min,max);

  //   c1 = newCanvas("c_1D_S2RPhi_hPullVsEta",form);
  //   plotAndProfileXSpread(h2RPhi->hPullVsEta,min,max);

  //   c1 = newCanvas("c_1D_S3RPhi_hPullVsEta",form);
  //   plotAndProfileXSpread(h3RPhi->hPullVsEta,min,max);

  //   c1 = newCanvas("c_1D_S1RZ_hPullVsEta",form);
  //   plotAndProfileXSpread(h1RZ->hPullVsEta,min,max);

  //   c1 = newCanvas("c_1D_S2RZ_hPullVsEta",form);
  //   plotAndProfileXSpread(h2RZ->hPullVsEta,min,max);

  //   c1 = newCanvas("c_1D_S3RZ_hPullVsEta",form);
  //   plotAndProfileXSpread(h3RZ->hPullVsEta,min,max);
  // }

  if(false && do2DRecHit) {
    c1 = newCanvas("c_2D_RPhi_hPullPosVsEta",form);
    plotAndProfileXSpread(h2DHitRPhi->hPullPosVsEta,-5,5);

    c1 = newCanvas("c_2D_RZ_hPullPosVsEta",form);
    plotAndProfileXSpread(h2DHitRZ->hPullPosVsEta,-5,5);

    c1 = newCanvas("c_2D_RZ_W0_hPullPosVsEta",form);
    plotAndProfileXSpread(h2DHitRZ_W0->hPullPosVsEta,-5,5);

    c1 = newCanvas("c_2D_RZ_W1_hPullPosVsEta",form);
    plotAndProfileXSpread(h2DHitRZ_W1->hPullPosVsEta,-5,5);

    c1 = newCanvas("c_2D_RZ_W2_hPullPosVsEta",form);
    plotAndProfileXSpread(h2DHitRZ_W2->hPullPosVsEta,-5,5);

    c1 = newCanvas("c_2D_RPhi_hPullAngleVsEta",form);
    plotAndProfileXSpread(h2DHitRPhi->hPullAngleVsEta,-5,5);

    c1 = newCanvas("c_2D_RZ_hPullAngleVsEta",form);
    plotAndProfileXSpread(h2DHitRZ->hPullAngleVsEta,-5,5);

    c1 = newCanvas("c_2D_RZ_W0_hPullAngleVsEta",form);
    plotAndProfileXSpread(h2DHitRZ_W0->hPullAngleVsEta,-5,5);

    c1 = newCanvas("c_2D_RZ_W1_hPullAngleVsEta",form);
    plotAndProfileXSpread(h2DHitRZ_W1->hPullAngleVsEta,-5,5);

    c1 = newCanvas("c_2D_RZ_W2_hPullAngleVsEta",form);
    plotAndProfileXSpread(h2DHitRZ_W2->hPullAngleVsEta,-5,5);

  }

  if(do2DSLPhiRecHit) {
    c1 = newCanvas("c_2D_SuperPhi_hPullPosVsEta",form);
    plotAndProfileXSpread(h2DSLPhiHit->hPullPosVsEta,-5,5);

    c1 = newCanvas("c_2D_SuperPhi_hPullAngleVsEta",form);
    plotAndProfileXSpread(h2DSLPhiHit->hPullAngleVsEta,-5,5);

  }

  if(do4DRecHit) {
    //// plot4DPullVsEta(h4DHit);
    plot4DPullVsEtaRZ(h4DHit);
    // plot4DPullVsEta(h4DHit_W0);
    // plot4DPullVsEta(h4DHit_W1);
    // plot4DPullVsEta(h4DHit_W2);
  }

  // Pullvsphi:

  // if(do1DRecHit) {
  //   const float min = -0.6;
  //   const float max =  0.6;
  //   c1 = newCanvas("c_1D_S1RPhi_hPullVsPhi",form);
  //   plotAndProfileXSpread(h1RPhi->hPullVsPhi,min, max);

  //   c1 = newCanvas("c_1D_S2RPhi_hPullVsPhi",form);
  //   plotAndProfileXSpread(h2RPhi->hPullVsPhi,min, max);

  //   c1 = newCanvas("c_1D_S3RPhi_hPullVsPhi",form);
  //   plotAndProfileXSpread(h3RPhi->hPullVsPhi,min, max);

  //   c1 = newCanvas("c_1D_S1RZ_hPullVsPhi",form);
  //   plotAndProfileXSpread(h1RZ->hPullVsPhi,min, max);

  //   c1 = newCanvas("c_1D_S2RZ_hPullVsPhi",form);
  //   plotAndProfileXSpread(h2RZ->hPullVsPhi,min, max);

  //   c1 = newCanvas("c_1D_S3RZ_hPullVsPhi",form);
  //   plotAndProfileXSpread(h3RZ->hPullVsPhi,min, max);

  // }

  if(false && do2DRecHit) {
    c1 = newCanvas("c_2D_RPhi_hPullPosVsPhi",form);
    plotAndProfileXSpread(h2DHitRPhi->hPullPosVsPhi,-5,5);

    c1 = newCanvas("c_2D_RZ_hPullPosVsPhi",form);
    plotAndProfileXSpread(h2DHitRZ->hPullPosVsPhi,-5,5);

    c1 = newCanvas("c_2D_RZ_W0_hPullPosVsPhi",form);
    plotAndProfileXSpread(h2DHitRZ_W0->hPullPosVsPhi,-5,5);

    c1 = newCanvas("c_2D_RZ_W1_hPullPosVsPhi",form);
    plotAndProfileXSpread(h2DHitRZ_W1->hPullPosVsPhi,-5,5);

    c1 = newCanvas("c_2D_RZ_W2_hPullPosVsPhi",form);
    plotAndProfileXSpread(h2DHitRZ_W2->hPullPosVsPhi,-5,5);

    c1 = newCanvas("c_2D_RPhi_hPullAngleVsPhi",form);
    plotAndProfileXSpread(h2DHitRPhi->hPullAngleVsPhi,-5,5);

    c1 = newCanvas("c_2D_RZ_hPullAngleVsPhi",form);
    plotAndProfileXSpread(h2DHitRZ->hPullAngleVsPhi,-5,5);

    c1 = newCanvas("c_2D_RZ_W0_hPullAngleVsPhi",form);
    plotAndProfileXSpread(h2DHitRZ_W0->hPullAngleVsPhi,-5,5);

    c1 = newCanvas("c_2D_RZ_W1_hPullAngleVsPhi",form);
    plotAndProfileXSpread(h2DHitRZ_W1->hPullAngleVsPhi,-5,5);

    c1 = newCanvas("c_2D_RZ_W2_hPullAngleVsPhi",form);
    plotAndProfileXSpread(h2DHitRZ_W2->hPullAngleVsPhi,-5,5);

  }

  if(do2DSLPhiRecHit) {
    c1 = newCanvas("c_2D_SuperPhi_hPullPosVsPhi",form);
    plotAndProfileXSpread(h2DSLPhiHit->hPullPosVsPhi,-5,5);

    c1 = newCanvas("c_2D_SuperPhi_hPullAngleVsPhi",form);
    plotAndProfileXSpread(h2DSLPhiHit->hPullAngleVsPhi,-5,5);
  }

  if(false && do4DRecHit) {
    plot4DPullVsPhi(h4DHit);
    plot4DPullVsPhiRZ(h4DHit);
    // plot4DPullVsPhi(h4DHit_W0);
    // plot4DPullVsPhi(h4DHit_W1);
    // plot4DPullVsPhi(h4DHit_W2);
  }



  // Pullvspos:

  if(do1DRecHit) {
    plot1DResidualsVsPos("Rphi", h1RPhi, h2RPhi, h3RPhi, ThreeIn1) ;
    plot1DResidualsVsPos("RZ", h1RZ, h2RZ, h3RZ, ThreeIn1) ;
    plot1DResidualsVsPos("RZ_W0", h1RZ_W0, h2RZ_W0, h3RZ_W0, ThreeIn1) ;
    plot1DResidualsVsPos("RZ_W1", h1RZ_W1, h2RZ_W1, h3RZ_W1, ThreeIn1) ;
    plot1DResidualsVsPos("RZ_W2", h1RZ_W2, h2RZ_W2, h3RZ_W2, ThreeIn1) ;
  }

  // Pullvsangle:

  if(do1DRecHit) {
    plot1DResidualsVsAngle("Rphi", h1RPhi, h2RPhi, h3RPhi, ThreeIn1) ;
    plot1DResidualsVsAngle("RZ", h1RZ, h2RZ, h3RZ, ThreeIn1) ;
    plot1DResidualsVsAngle("RZ_W0", h1RZ_W0, h2RZ_W0, h3RZ_W0, ThreeIn1) ;
    plot1DResidualsVsAngle("RZ_W1", h1RZ_W1, h2RZ_W1, h3RZ_W1, ThreeIn1) ;
    plot1DResidualsVsAngle("RZ_W2", h1RZ_W2, h2RZ_W2, h3RZ_W2, ThreeIn1) ;
  }


  // if(do2DRecHit) {
  //   //cout << "h2DHitRPhi: " << (int)h2DHitRPhi << endl;
  //   plot2DAngles(h2DHitRPhi, ThreeIn1);
  //   plot2DAngles(h2DHitRZ, ThreeIn1);
  //   plot2DAngles(h2DHitRZ_W0, ThreeIn1);
  //   plot2DAngles(h2DHitRZ_W1, ThreeIn1);
  //   plot2DAngles(h2DHitRZ_W2, ThreeIn1);
  // }

  // if(do2DSLPhiRecHit) {
  //   plot2DAngles(h2DSLPhiHit, ThreeIn1);
  // }

  // if(do4DRecHit) {
  //   plot4DAngles(h4DHit, ThreeIn1);
  //   plot4DAngles(h4DHit_W0, ThreeIn1);
  //   plot4DAngles(h4DHit_W1, ThreeIn1);
  //   plot4DAngles(h4DHit_W2, ThreeIn1);
  // }

  return;

}


void plot1DResiduals(HRes1DHit * h1, HRes1DHit * h2, HRes1DHit * h3, bool ThreeIn1) {
  int i = 2;

  if(ThreeIn1)
    cout << "ThreeIn1 = true!" << endl;
  else
    cout << "ThreeIn1 = false!" << endl;

  TString N1 = "c_1D_" + h1->name;
  TString N2 = "c_1D_" + h2->name;
  TString N3 = "c_1D_" + h3->name;
  int form = 2;
  if (ThreeIn1) c1 = newCanvas(N1.Remove(5,2)+"_hPull",3,1,800,400);
  else newCanvas(N1+"_hPull",form);
  cout << "h1->hPull " << h1->hPull << endl;
  drawGFit(h1->hPull, -5,5,-5,5);

  if (ThreeIn1) c1->cd(i++);
  else c1 = newCanvas(N2+"_hPull",form);
  drawGFit(h2->hPull, -5,5,-5,5);

  if (ThreeIn1) c1->cd(i++);
  else c1 = newCanvas(N3+"_hPull",form);
  drawGFit(h3->hPull, -5,5,-5,5);

}

void plot1DResidualsVsPos(TString name, HRes1DHit * h1, HRes1DHit * h2, HRes1DHit * h3, bool ThreeIn1) {
    bool profile = true;
    const float min = -5;
    const float max =  5;
    
    int i = 2;
    if (ThreeIn1) c1 = newCanvas("c_1D_S1"+name+"_hPullVsPos",3,1,800,400);
    else newCanvas("c_1D_S1"+name+"_hPullVsPos",form);
    plotAndProfileXSpread(h1->hPullVsPos, min, max, profile);

    if (ThreeIn1) c1->cd(i++);
    else c1 = newCanvas("c_1D_S2"+name+"_hPullVsPos",form);
    plotAndProfileXSpread(h2->hPullVsPos, min, max, profile);

    if (ThreeIn1) c1->cd(i++);
    else c1 = newCanvas("c_1D_S3"+name+"_hPullVsPos",form);
    plotAndProfileXSpread(h3->hPullVsPos, min, max, profile);
}

void plot1DResidualsVsAngle(TString name, HRes1DHit * h1, HRes1DHit * h2, HRes1DHit * h3, bool ThreeIn1) {
    bool profile = true;
    const float min = -5;
    const float max =  5;
    
    int i = 2;
    if (ThreeIn1) c1 = newCanvas("c_1D_S1"+name+"_hPullVsAngle",3,1,800,400);
    else newCanvas("c_1D_S1"+name+"_hPullVsAngle",form);
    plotAndProfileXSpread(h1->hPullVsAngle, min, max, profile);

    if (ThreeIn1) c1->cd(i++);
    else c1 = newCanvas("c_1D_S2"+name+"_hPullVsAngle",form);
    plotAndProfileXSpread(h2->hPullVsAngle, min, max, profile);

    if (ThreeIn1) c1->cd(i++);
    else c1 = newCanvas("c_1D_S3"+name+"_hPullVsAngle",form);
    plotAndProfileXSpread(h3->hPullVsAngle, min, max, profile);
}

void plot2DResiduals(HRes2DHit * h1) {
  int i = 2;

  TString N1 = "c_2D_" + h1->name;

  int form = 2;
  newCanvas(N1+"_hPullPos",form);
  drawGFit(h1->hPullPos, -5,5,-5,5);

  newCanvas(N1+"_hPullAngle",form);
  cout << "h1->hPullAngle " << h1->hPullAngle << endl;
  drawGFit(h1->hPullAngle, -5,5,-5,5);
}

void plot4DResiduals(HRes4DHit * h1) {
  int i = 2;

  TString N1 = "c_4D_" + h1->name;

  int form = 2;
  newCanvas(N1+"_hPullX",form);
  drawGFit(h1->hPullX, -5,5,-2.5,2.5);

  newCanvas(N1+"_hPullY",form);
  drawGFit(h1->hPullY, -5,5,-2.5,2.5);

  newCanvas(N1+"_hPullAlpha",form);
  drawGFit(h1->hPullAlpha, -5,5,-2.5,2.5);

  newCanvas(N1+"_hPullBeta",form);
  drawGFit(h1->hPullBeta, -5,5,-2.5,2.5);

}

void plot4DPullVsEta(HRes4DHit * h1) {
  int i = 2;

  TString N1 = "c_4D_" + h1->name;

  int form = 2;
  newCanvas(N1+"_hPullXVsEta",form);
  plotAndProfileXSpread(h1->hPullXVsEta,-5,5);

  newCanvas(N1+"_hPullYVsEta",form);
  plotAndProfileXSpread(h1->hPullYVsEta,-5,5);

  newCanvas(N1+"_hPullAlphaVsEta",form);
  plotAndProfileXSpread(h1->hPullAlphaVsEta,-5,5);

  newCanvas(N1+"_hPullBetaVsEta",form);
  plotAndProfileXSpread(h1->hPullBetaVsEta,-5,5);
}

void plot4DPullVsPhi(HRes4DHit * h1) {
  int i = 2;

  TString N1 = "c_4D_" + h1->name;

  int form = 2;
  newCanvas(N1+"_hPullXVsPhi",form);
  plotAndProfileXSpread(h1->hPullXVsPhi,-5,5);

  newCanvas(N1+"_hPullYVsPhi",form);
  plotAndProfileXSpread(h1->hPullYVsPhi,-5,5);

  newCanvas(N1+"_hPullAlphaVsPhi",form);
  plotAndProfileXSpread(h1->hPullAlphaVsPhi,-5,5);

  newCanvas(N1+"_hPullBetaVsPhi",form);
  plotAndProfileXSpread(h1->hPullBetaVsPhi,-5,5);
}

void plot2DAngles(HRes2DHit * h1, bool ThreeIn1) {
  int i = 2;

  TString N1 = "c_2D_" + h1->name;

  int form = 2;
  if (ThreeIn1) c1 = newCanvas(N1+"_hAngle",3,1,800,400);
  else newCanvas(N1+"_hRecAngle",form);
  h1->hRecAngle->Draw();

  if (ThreeIn1) c1->cd(i++);
  else c1 = newCanvas(N1+"_hSimAngle",form);
  h1->hSimAngle->Draw();

  //cout << "h1->hRecVsSimAngle: " << (int)h1->hRecVsSimAngle << endl;

  if (ThreeIn1) c1->cd(i++);
  else c1 = newCanvas(N1+"_hRecVsSimAngle",form);
  plotAndProfileXSpread(h1->hRecVsSimAngle,-5,5);

}

void plot4DAngles(HRes4DHit * h1, bool ThreeIn1) {
  int i = 2;

  TString N1 = "c_4D_" + h1->name;

  int form = 2;
  if (ThreeIn1) c1 = newCanvas(N1+"_hAlpha",3,1,800,400);
  else newCanvas(N1+"_hRecAlpha",form);
  h1->hRecAlpha->Draw();

  if (ThreeIn1) c1->cd(i++);
  else c1 = newCanvas(N1+"_hSimAlpha",form);
  h1->hSimAlpha->Draw();

  if (ThreeIn1) c1->cd(i++);
  else c1 = newCanvas(N1+"_hRecVsSimAlpha",form);
  plotAndProfileXSpread(h1->hRecVsSimAlpha,-5,5);

  i=2;

  if (ThreeIn1) c2 = newCanvas(N1+"_hBeta",3,1,800,400);
  else newCanvas(N1+"_hRecBeta",form);
  h1->hRecBeta->Draw();

  if (ThreeIn1) c2->cd(i++);
  else c1 = newCanvas(N1+"_hSimBeta",form);
  h1->hSimBeta->Draw();

  if (ThreeIn1) c2->cd(i++);
  else c1 = newCanvas(N1+"_hRecVsSimBeta",form);
  plotAndProfileXSpread(h1->hRecVsSimBeta,-5,5);

}

void plot4DResidualsRZ(HRes4DHit * h1) {
  int i = 2;

  TString N1 = "c_4D_" + h1->name;

  int form = 2;
  newCanvas(N1+"_hPullYRZ",form);
  drawGFit(h1->hPullYRZ, -5,5,-2.5,2.5);

  newCanvas(N1+"_hPullBetaRZ",form);
  drawGFit(h1->hPullBetaRZ, -5,5,-2.5,2.5);

}

void plot4DPullVsEtaRZ(HRes4DHit * h1) {
  int i = 2;

  TString N1 = "c_4D_" + h1->name;

  int form = 2;
  newCanvas(N1+"_hPullYVsEtaRZ",form);
  plotAndProfileXSpreadSpread(h1->hPullYVsEtaRZ,-5.,5., true,-2.5,2.5);

  newCanvas(N1+"_hPullBetaVsEtaRZ",form);
  plotAndProfileXSpreadSpread(h1->hPullBetaVsEtaRZ,-5.,5., true,-2.5,2.5);
}

void plot4DPullVsPhiRZ(HRes4DHit * h1) {
  int i = 2;

  TString N1 = "c_4D_" + h1->name;

  int form = 2;
  newCanvas(N1+"_hPullYVsPhiRZ",form);
  plotAndProfileXSpread(h1->hPullYVsPhiRZ,-5,5);

  newCanvas(N1+"_hPullBetaVsPhiRZ",form);
  plotAndProfileXSpread(h1->hPullBetaVsPhiRZ,-5,5);
}


bool setPreferences(bool& do1DRecHit,
    bool& do2DRecHit,
    bool& do2DSLPhiRecHit,
    bool& do4DRecHit,
    bool& ThreeIn1) {



  int dimension = 0;

  cout << "===================================================" << endl;
  cout << "==== plotHitPull User Menu =====================" << endl;
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
        //setPreferences(do1DRecHit, do2DRecHit, do2DSLPhiRecHit, do4DRecHit, ThreeIn1);
        break;
      }
  }

  return true;
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

