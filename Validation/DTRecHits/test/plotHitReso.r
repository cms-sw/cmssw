
/*
 * Format plots of resolutions, etc. produced by DTRecHits validation.
 * The root tree containing the histograms must be already open when 
 * executing this macro.
 * 
 * G. Cerminara 2004
 */

// class hRHit;
class HRes1DHit;
class HRes2DHit;
class HRes4DHit;

void plotHitReso();
void plotWWWHitReso(int dimSwitch = 1, TString nameDir = "");
void draw(bool do1DRecHit, bool do2DRecHit, bool do2DSLPhiRecHit, bool do4DRecHit, bool ThreeIn1, int form);
void plot1DResiduals(HRes1DHit * h1, HRes1DHit * h2, HRes1DHit * h3, bool ThreeIn1);
void plot2DResiduals(HRes2DHit * h1);

void plot4DResiduals(HRes4DHit * h1);
void plot4DResVsEta(HRes4DHit * h1);
void plot4DResVsPhi(HRes4DHit * h1);

void plot2DAngles(HRes2DHit * h1, bool ThreeIn1);
void plot4DAngles(HRes4DHit * h1, bool ThreeIn1);

// Read user input
bool setPreferences(bool& do1DRecHit, bool& do2DRecHit, bool& do2DSLPhiRecHit, bool& do4DRecHit, bool& ThreeIn1);
// 




// This is the main function
void plotHitReso(){
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
  bool ThreeIn1 = false;  // Plot the 3 steps in a single canvas (where appl.)

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
  style->SetFitFormat("5.3g");
  style->SetOptFit(11);

  //--------------------------------------------------------------------------------------

  // Check the choice
  if(!do1DRecHit && !do2DRecHit && !do2DSLPhiRecHit && !do4DRecHit) {
    cout << "[plotHitReso]***Error: Nothing to do! Set do1DRecHit, do2DRecHit, do4DRecHit correctly!"
      << endl;
    return;
  }
  style->cd();                      // Apply style 

  draw(do1DRecHit, do2DRecHit, do2DSLPhiRecHit, do4DRecHit, ThreeIn1, form);

}

void plotWWWHitReso(int dimSwitch, TString nameDir) {
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
    cout << "[plotHitReso]***Error: Nothing to do! Set do1DRecHit, do2DRecHit, do4DRecHit correctly!"
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

  if(do4DRecHit) {
    plot4DResiduals(h4DHit);
    plot4DResidualsRZ(h4DHit);
    // plot4DResiduals(h4DHit_W0);
    // plot4DResiduals(h4DHit_W1);
    // plot4DResiduals(h4DHit_W2);
  }

  // resovseta:

  if(do1DRecHit) {
    const float min = -0.6;
    const float max =  0.6;
    c1 = newCanvas("c_1D_S1RPhi_hResVsEta",form);
    plotAndProfileX(h1RPhi->hResVsEta,min,max);

    c1 = newCanvas("c_1D_S2RPhi_hResVsEta",form);
    plotAndProfileX(h2RPhi->hResVsEta,min,max);

    c1 = newCanvas("c_1D_S3RPhi_hResVsEta",form);
    plotAndProfileX(h3RPhi->hResVsEta,min,max);

    c1 = newCanvas("c_1D_S1RZ_hResVsEta",form);
    plotAndProfileX(h1RZ->hResVsEta,min,max);

    c1 = newCanvas("c_1D_S2RZ_hResVsEta",form);
    plotAndProfileX(h2RZ->hResVsEta,min,max);

    c1 = newCanvas("c_1D_S3RZ_hResVsEta",form);
    plotAndProfileX(h3RZ->hResVsEta,min,max);
  }

  if(false && do2DRecHit) {
    c1 = newCanvas("c_2D_RPhi_hResPosVsEta",form);
    plotAndProfileX(h2DHitRPhi->hResPosVsEta,-3,3);

    c1 = newCanvas("c_2D_RZ_hResPosVsEta",form);
    plotAndProfileX(h2DHitRZ->hResPosVsEta,-3,3);

    c1 = newCanvas("c_2D_RZ_W0_hResPosVsEta",form);
    plotAndProfileX(h2DHitRZ_W0->hResPosVsEta,-3,3);

    c1 = newCanvas("c_2D_RZ_W1_hResPosVsEta",form);
    plotAndProfileX(h2DHitRZ_W1->hResPosVsEta,-3,3);

    c1 = newCanvas("c_2D_RZ_W2_hResPosVsEta",form);
    plotAndProfileX(h2DHitRZ_W2->hResPosVsEta,-3,3);

    c1 = newCanvas("c_2D_RPhi_hResAngleVsEta",form);
    plotAndProfileX(h2DHitRPhi->hResAngleVsEta,-3,3);

    c1 = newCanvas("c_2D_RZ_hResAngleVsEta",form);
    plotAndProfileX(h2DHitRZ->hResAngleVsEta,-3,3);

    c1 = newCanvas("c_2D_RZ_W0_hResAngleVsEta",form);
    plotAndProfileX(h2DHitRZ_W0->hResAngleVsEta,-3,3);

    c1 = newCanvas("c_2D_RZ_W1_hResAngleVsEta",form);
    plotAndProfileX(h2DHitRZ_W1->hResAngleVsEta,-3,3);

    c1 = newCanvas("c_2D_RZ_W2_hResAngleVsEta",form);
    plotAndProfileX(h2DHitRZ_W2->hResAngleVsEta,-3,3);

  }

  if(do2DSLPhiRecHit) {
    c1 = newCanvas("c_2D_SuperPhi_hResPosVsEta",form);
    plotAndProfileX(h2DSLPhiHit->hResPosVsEta,-3,3);

    c1 = newCanvas("c_2D_SuperPhi_hResAngleVsEta",form);
    plotAndProfileX(h2DSLPhiHit->hResAngleVsEta,-3,3);

  }

  if(do4DRecHit) {
    plot4DResVsEta(h4DHit);
    plot4DResVsEtaRZ(h4DHit);
    // plot4DResVsEta(h4DHit_W0);
    // plot4DResVsEta(h4DHit_W1);
    // plot4DResVsEta(h4DHit_W2);
  }

  // resovsphi:

  if(do1DRecHit) {
    const float min = -0.6;
    const float max =  0.6;
    c1 = newCanvas("c_1D_S1RPhi_hResVsPhi",form);
    plotAndProfileX(h1RPhi->hResVsPhi,min, max);

    c1 = newCanvas("c_1D_S2RPhi_hResVsPhi",form);
    plotAndProfileX(h2RPhi->hResVsPhi,min, max);

    c1 = newCanvas("c_1D_S3RPhi_hResVsPhi",form);
    plotAndProfileX(h3RPhi->hResVsPhi,min, max);

    c1 = newCanvas("c_1D_S1RZ_hResVsPhi",form);
    plotAndProfileX(h1RZ->hResVsPhi,min, max);

    c1 = newCanvas("c_1D_S2RZ_hResVsPhi",form);
    plotAndProfileX(h2RZ->hResVsPhi,min, max);

    c1 = newCanvas("c_1D_S3RZ_hResVsPhi",form);
    plotAndProfileX(h3RZ->hResVsPhi,min, max);

  }

  if(false && do2DRecHit) {
    c1 = newCanvas("c_2D_RPhi_hResPosVsPhi",form);
    plotAndProfileX(h2DHitRPhi->hResPosVsPhi,-3,3);

    c1 = newCanvas("c_2D_RZ_hResPosVsPhi",form);
    plotAndProfileX(h2DHitRZ->hResPosVsPhi,-3,3);

    c1 = newCanvas("c_2D_RZ_W0_hResPosVsPhi",form);
    plotAndProfileX(h2DHitRZ_W0->hResPosVsPhi,-3,3);

    c1 = newCanvas("c_2D_RZ_W1_hResPosVsPhi",form);
    plotAndProfileX(h2DHitRZ_W1->hResPosVsPhi,-3,3);

    c1 = newCanvas("c_2D_RZ_W2_hResPosVsPhi",form);
    plotAndProfileX(h2DHitRZ_W2->hResPosVsPhi,-3,3);

    c1 = newCanvas("c_2D_RPhi_hResAngleVsPhi",form);
    plotAndProfileX(h2DHitRPhi->hResAngleVsPhi,-3,3);

    c1 = newCanvas("c_2D_RZ_hResAngleVsPhi",form);
    plotAndProfileX(h2DHitRZ->hResAngleVsPhi,-3,3);

    c1 = newCanvas("c_2D_RZ_W0_hResAngleVsPhi",form);
    plotAndProfileX(h2DHitRZ_W0->hResAngleVsPhi,-3,3);

    c1 = newCanvas("c_2D_RZ_W1_hResAngleVsPhi",form);
    plotAndProfileX(h2DHitRZ_W1->hResAngleVsPhi,-3,3);

    c1 = newCanvas("c_2D_RZ_W2_hResAngleVsPhi",form);
    plotAndProfileX(h2DHitRZ_W2->hResAngleVsPhi,-3,3);

  }

  if(do2DSLPhiRecHit) {
    c1 = newCanvas("c_2D_SuperPhi_hResPosVsPhi",form);
    plotAndProfileX(h2DSLPhiHit->hResPosVsPhi,-3,3);

    c1 = newCanvas("c_2D_SuperPhi_hResAngleVsPhi",form);
    plotAndProfileX(h2DSLPhiHit->hResAngleVsPhi,-3,3);
  }

  if(do4DRecHit) {
    plot4DResVsPhi(h4DHit);
    plot4DResVsPhiRZ(h4DHit);
    // plot4DResVsPhi(h4DHit_W0);
    // plot4DResVsPhi(h4DHit_W1);
    // plot4DResVsPhi(h4DHit_W2);
  }



  // resovspos:

  if(do1DRecHit) {
    bool profile = true;
    const float min = -1;
    const float max =  1;
    c1 = newCanvas("c_1D_S1RPhi_hResVsPos",form);
    plotAndProfileX(h1RPhi->hResVsPos, min, max, profile);

    c1 = newCanvas("c_1D_S2RPhi_hResVsPos",form);
    plotAndProfileX(h2RPhi->hResVsPos, min, max, profile);

    c1 = newCanvas("c_1D_S3RPhi_hResVsPos",form);
    plotAndProfileX(h3RPhi->hResVsPos, min, max, profile);

    c1 = newCanvas("c_1D_S1RZ_hResVsPos",form);
    plotAndProfileX(h1RZ->hResVsPos, min, max, profile);

    c1 = newCanvas("c_1D_S1RZ_W0_hResVsPos",form);
    plotAndProfileX(h1RZ_W0->hResVsPos, min, max, profile);

    c1 = newCanvas("c_1D_S1RZ_W1_hResVsPos",form);
    plotAndProfileX(h1RZ_W1->hResVsPos, min, max, profile);

    c1 = newCanvas("c_1D_S1RZ_W2_hResVsPos",form);
    plotAndProfileX(h1RZ_W2->hResVsPos, min, max, profile);

    c1 = newCanvas("c_1D_S2RZ_hResVsPos",form);
    plotAndProfileX(h2RZ->hResVsPos, min, max, profile);

    c1 = newCanvas("c_1D_S2RZ_W0_hResVsPos",form);
    plotAndProfileX(h2RZ_W0->hResVsPos, min, max, profile);

    c1 = newCanvas("c_1D_S2RZ_W1_hResVsPos",form);
    plotAndProfileX(h2RZ_W1->hResVsPos, min, max, profile);

    c1 = newCanvas("c_1D_S2RZ_W2_hResVsPos",form);
    plotAndProfileX(h2RZ_W2->hResVsPos, min, max, profile);

    c1 = newCanvas("c_1D_S3RZ_hResVsPos",form);
    plotAndProfileX(h3RZ->hResVsPos, min, max, profile);

    c1 = newCanvas("c_1D_S3RZ_W0_hResVsPos",form);
    plotAndProfileX(h3RZ_W0->hResVsPos, min, max, profile);

    c1 = newCanvas("c_1D_S3RZ_W1_hResVsPos",form);
    plotAndProfileX(h3RZ_W1->hResVsPos, min, max, profile);

    c1 = newCanvas("c_1D_S3RZ_W2_hResVsPos",form);
    plotAndProfileX(h3RZ_W2->hResVsPos, min, max, profile);

  }

  // resovsangle:

  if(do1DRecHit) {
    bool profile = true;
    const float min = -1;
    const float max =  1;
    c1 = newCanvas("c_1D_S1RPhi_hResVsAngle",form);
    plotAndProfileX(h1RPhi->hResVsAngle, min, max, profile);

    c1 = newCanvas("c_1D_S2RPhi_hResVsAngle",form);
    plotAndProfileX(h2RPhi->hResVsAngle, min, max, profile);

    c1 = newCanvas("c_1D_S3RPhi_hResVsAngle",form);
    plotAndProfileX(h3RPhi->hResVsAngle, min, max, profile);

    c1 = newCanvas("c_1D_S1RZ_hResVsAngle",form);
    plotAndProfileX(h1RZ->hResVsAngle, min, max, profile);

    c1 = newCanvas("c_1D_S1RZ_W0_hResVsAngle",form);
    plotAndProfileX(h1RZ_W0->hResVsAngle, min, max, profile);

    c1 = newCanvas("c_1D_S1RZ_W1_hResVsAngle",form);
    plotAndProfileX(h1RZ_W1->hResVsAngle, min, max, profile);

    c1 = newCanvas("c_1D_S1RZ_W2_hResVsAngle",form);
    plotAndProfileX(h1RZ_W2->hResVsAngle, min, max, profile);

    c1 = newCanvas("c_1D_S2RZ_hResVsAngle",form);
    plotAndProfileX(h2RZ->hResVsAngle, min, max, profile);

    c1 = newCanvas("c_1D_S2RZ_W0_hResVsAngle",form);
    plotAndProfileX(h2RZ_W0->hResVsAngle, min, max, profile);

    c1 = newCanvas("c_1D_S2RZ_W1_hResVsAngle",form);
    plotAndProfileX(h2RZ_W1->hResVsAngle, min, max, profile);

    c1 = newCanvas("c_1D_S2RZ_W2_hResVsAngle",form);
    plotAndProfileX(h2RZ_W2->hResVsAngle, min, max, profile);

    c1 = newCanvas("c_1D_S3RZ_hResVsAngle",form);
    plotAndProfileX(h3RZ->hResVsAngle, min, max, profile);

    c1 = newCanvas("c_1D_S3RZ_W0_hResVsAngle",form);
    plotAndProfileX(h3RZ_W0->hResVsAngle, min, max, profile);

    c1 = newCanvas("c_1D_S3RZ_W1_hResVsAngle",form);
    plotAndProfileX(h3RZ_W1->hResVsAngle, min, max, profile);

    c1 = newCanvas("c_1D_S3RZ_W2_hResVsAngle",form);
    plotAndProfileX(h3RZ_W2->hResVsAngle, min, max, profile);

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
  if (ThreeIn1) c1 = newCanvas(N1.Remove(5,2)+"_hRes",3,1,1200,500);
  else newCanvas(N1+"_hRes",form);
  drawGFit(h1->hRes, -0.2,0.2,-0.1,0.1);

  if (ThreeIn1) c1->cd(i++);
  else c1 = newCanvas(N2+"_hRes",form);
  drawGFit(h2->hRes, -0.2,0.2,-0.1,0.1);

  if (ThreeIn1) c1->cd(i++);
  else c1 = newCanvas(N3+"_hRes",form);
  drawGFit(h3->hRes, -0.2,0.2,-0.1,0.1);

}

void plot2DResiduals(HRes2DHit * h1) {
  int i = 2;

  TString N1 = "c_2D_" + h1->name;

  int form = 2;
  newCanvas(N1+"_hResPos",form);
  cout << "h1->hResPos " << h1->hResPos << endl;
  drawGFit(h1->hResPos, -0.1,0.1,-0.1,0.1);

  newCanvas(N1+"_hResAngle",form);
  cout << "h1->hResAngle " << h1->hResAngle << endl;
  drawGFit(h1->hResAngle, -0.1,0.1,-0.1,0.1);
}

void plot4DResiduals(HRes4DHit * h1) {
  int i = 2;

  TString N1 = "c_4D_" + h1->name;

  int form = 2;
  newCanvas(N1+"_hResX",form);
  drawGFit(h1->hResX, -0.2,0.2,-0.1,0.1);

  newCanvas(N1+"_hResY",form);
  drawGFit(h1->hResY, -0.2,0.2,-0.1,0.1);

  newCanvas(N1+"_hResAlpha",form);
  drawGFit(h1->hResAlpha, -0.2,0.2,-0.1,0.1);

  newCanvas(N1+"_hResBeta",form);
  drawGFit(h1->hResBeta, -0.2,0.2,-0.1,0.1);

}

void plot4DResVsEta(HRes4DHit * h1) {
  int i = 2;

  TString N1 = "c_4D_" + h1->name;

  int form = 2;
  newCanvas(N1+"_hResXVsEta",form);
  plotAndProfileX(h1->hResXVsEta,-3,3);

  newCanvas(N1+"_hResYVsEta",form);
  plotAndProfileX(h1->hResYVsEta,-3,3);

  newCanvas(N1+"_hResAlphaVsEta",form);
  plotAndProfileX(h1->hResAlphaVsEta,-3,3);

  newCanvas(N1+"_hResBetaVsEta",form);
  plotAndProfileX(h1->hResBetaVsEta,-3,3);
}

void plot4DResVsPhi(HRes4DHit * h1) {
  int i = 2;

  TString N1 = "c_4D_" + h1->name;

  int form = 2;
  newCanvas(N1+"_hResXVsPhi",form);
  plotAndProfileX(h1->hResXVsPhi,-3,3);

  newCanvas(N1+"_hResYVsPhi",form);
  plotAndProfileX(h1->hResYVsPhi,-3,3);

  newCanvas(N1+"_hResAlphaVsPhi",form);
  plotAndProfileX(h1->hResAlphaVsPhi,-3,3);

  newCanvas(N1+"_hResBetaVsPhi",form);
  plotAndProfileX(h1->hResBetaVsPhi,-3,3);
}

void plot2DAngles(HRes2DHit * h1, bool ThreeIn1) {
  int i = 2;

  TString N1 = "c_2D_" + h1->name;

  int form = 2;
  if (ThreeIn1) c1 = newCanvas(N1+"_hAngle",3,1,1200,500);
  else newCanvas(N1+"_hRecAngle",form);
  h1->hRecAngle->Draw();

  if (ThreeIn1) c1->cd(i++);
  else c1 = newCanvas(N1+"_hSimAngle",form);
  h1->hSimAngle->Draw();

  //cout << "h1->hRecVsSimAngle: " << (int)h1->hRecVsSimAngle << endl;

  if (ThreeIn1) c1->cd(i++);
  else c1 = newCanvas(N1+"_hRecVsSimAngle",form);
  plotAndProfileX(h1->hRecVsSimAngle,-3,3);

}

void plot4DAngles(HRes4DHit * h1, bool ThreeIn1) {
  int i = 2;

  TString N1 = "c_4D_" + h1->name;

  int form = 2;
  if (ThreeIn1) c1 = newCanvas(N1+"_hAlpha",3,1,1200,500);
  else newCanvas(N1+"_hRecAlpha",form);
  h1->hRecAlpha->Draw();

  if (ThreeIn1) c1->cd(i++);
  else c1 = newCanvas(N1+"_hSimAlpha",form);
  h1->hSimAlpha->Draw();

  if (ThreeIn1) c1->cd(i++);
  else c1 = newCanvas(N1+"_hRecVsSimAlpha",form);
  plotAndProfileX(h1->hRecVsSimAlpha,-3,3);

  i=2;

  if (ThreeIn1) c2 = newCanvas(N1+"_hBeta",3,1,1200,500);
  else newCanvas(N1+"_hRecBeta",form);
  h1->hRecBeta->Draw();

  if (ThreeIn1) c2->cd(i++);
  else c1 = newCanvas(N1+"_hSimBeta",form);
  h1->hSimBeta->Draw();

  if (ThreeIn1) c2->cd(i++);
  else c1 = newCanvas(N1+"_hRecVsSimBeta",form);
  plotAndProfileX(h1->hRecVsSimBeta,-3,3);

}

void plot4DResidualsRZ(HRes4DHit * h1) {
  int i = 2;

  TString N1 = "c_4D_" + h1->name;

  int form = 2;
  newCanvas(N1+"_hResYRZ",form);
  drawGFit(h1->hResYRZ, -0.1,0.1,-0.1,0.1);

  newCanvas(N1+"_hResBetaRZ",form);
  drawGFit(h1->hResBetaRZ, -0.1,0.1,-0.1,0.1);

}

void plot4DResVsEtaRZ(HRes4DHit * h1) {
  int i = 2;

  TString N1 = "c_4D_" + h1->name;

  int form = 2;
  newCanvas(N1+"_hResYVsEtaRZ",form);
  plotAndProfileX(h1->hResYVsEtaRZ,-3,3);

  newCanvas(N1+"_hResBetaVsEtaRZ",form);
  plotAndProfileX(h1->hResBetaVsEtaRZ,-3,3);
}

void plot4DResVsPhiRZ(HRes4DHit * h1) {
  int i = 2;

  TString N1 = "c_4D_" + h1->name;

  int form = 2;
  newCanvas(N1+"_hResYVsPhiRZ",form);
  plotAndProfileX(h1->hResYVsPhiRZ,-3,3);

  newCanvas(N1+"_hResBetaVsPhiRZ",form);
  plotAndProfileX(h1->hResBetaVsPhiRZ,-3,3);
}


bool setPreferences(bool& do1DRecHit,
    bool& do2DRecHit,
    bool& do2DSLPhiRecHit,
    bool& do4DRecHit,
    bool& ThreeIn1) {



  int dimension = 0;

  cout << "===================================================" << endl;
  cout << "==== plotHitReso User Menu =====================" << endl;
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

