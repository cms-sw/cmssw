
class HRes1DHit;
class HRes2DHit;
class HRes4DHit;

void plotHitReso();
void draw(bool, bool, bool, bool, bool, bool ThreeInOne);
void plot1DRes(HRes1DHit * h1, HRes1DHit * h2, HRes1DHit * h3, bool ThreeInOne);
void plot1DResVsPos(HRes1DHit * h1, HRes1DHit * h2, HRes1DHit * h3, bool ThreeInOne);
bool setPreferences(bool& doRes, bool& doResVsPos, bool& doResCloseWire, bool&, bool&, bool& ThreeIn1) ;

// This is the main function
void plotHit1DReso(){
  // Load needed macros and files
  gROOT->LoadMacro("macros.C");     // Load service macros
  gROOT->LoadMacro("../plugins/Histograms.h"); // Load definition of histograms

  // Get the style
  TStyle * style = getStyle("tdr");

  // //style->SetOptStat("RME");
  // style->SetOptStat(0);
  // style->SetOptFit(11);
  style->SetFitFormat("3.2e");
  // style->SetStatFontSize(0.03);
  // style->SetStatColor(0);
  // style->SetStatY(0.99);
  // style->SetStatX(0.99);
  // style->SetStatH(0.3);
  // style->SetStatW(0.25);
  // // style->SetStatW(0.25);
  // // style->SetStatH(0.35);
  // style->SetTitleFillColor(0);

  style->cd();                      // Apply style 

  //Main switches
  bool ThreeInOne = true;  // Plot the 3 steps in a single canvas (where appl.)
  bool doRes = false; 
  bool doResVsPos = false; 
  bool doResCloseWire = false; 
  bool doResVsAngle = false; 
  bool doResVsFE = false; 
  // Read user input, namely what plots should be produced
  
  while(!setPreferences(doRes, doResVsPos, doResCloseWire, doResVsAngle, doResVsFE, ThreeInOne));

  draw(doRes, doResVsPos, doResCloseWire, doResVsAngle, doResVsFE, ThreeInOne);
}

void draw(bool doRes, bool doResVsPos, bool doResCloseWire, bool doResVsAngle, bool doResVsFE, bool ThreeInOne) {
  // Retrieve histogram sets
  TFile *f = gROOT->GetListOfFiles()->Last();
  if (!f) {
    cout << "No file loaded" << endl;
    return;
  }
  cout << "Loading file: " << f->GetName() << endl;

  HRes1DHit *h1RPhi = new HRes1DHit("S1RPhi",f);     // RecHits, 1. step, RPhi
  HRes1DHit *h2RPhi = new HRes1DHit("S2RPhi",f);     // RecHits, 2. step, RPhi
  HRes1DHit *h3RPhi = new HRes1DHit("S3RPhi",f);     // RecHits, 3. step, RPhi

  HRes1DHit *h1RZ = new HRes1DHit("S1RZ",f);         // RecHits, 1. step, RZ
  HRes1DHit *h2RZ = new HRes1DHit("S2RZ",f);        // RecHits, 2. step, RZ
  HRes1DHit *h3RZ = new HRes1DHit("S3RZ",f);        // RecHits, 3. step, RZ

  HRes1DHit *h1RZ_W0 = new HRes1DHit("S1RZ_W0",f);   // RecHits, 1. step, RZ, wheel 0
  HRes1DHit *h2RZ_W0 = new HRes1DHit("S2RZ_W0",f);   // RecHits, 2. step, RZ, wheel 0
  HRes1DHit *h3RZ_W0 = new HRes1DHit("S3RZ_W0",f);   // RecHits, 3. step, RZ, wheel 0

  HRes1DHit *h1RZ_W1 = new HRes1DHit("S1RZ_W1",f);   // RecHits, 1. step, RZ, wheel +-1
  HRes1DHit *h2RZ_W1 = new HRes1DHit("S2RZ_W1",f);   // RecHits, 2. step, RZ, wheel +-1
  HRes1DHit *h3RZ_W1 = new HRes1DHit("S3RZ_W1",f);   // RecHits, 3. step, RZ, wheel +-1

  HRes1DHit *h1RZ_W2 = new HRes1DHit("S1RZ_W2",f);   // RecHits, 1. step, RZ, wheel +-2
  HRes1DHit *h2RZ_W2 = new HRes1DHit("S2RZ_W2",f);   // RecHits, 2. step, RZ, wheel +-2
  HRes1DHit *h3RZ_W2 = new HRes1DHit("S3RZ_W2",f);   // RecHits, 3. step, RZ, wheel +-2

  if (doRes) {
    // Res, Rphi
    plot1DRes(h1RPhi,h2RPhi,h3RPhi,ThreeInOne);

    // Res, RZ 
    plot1DRes(h1RZ,h2RZ,h3RZ,ThreeInOne);

    // Res, RZ, per wheel
    plot1DRes(h1RZ_W0,h2RZ_W0,h3RZ_W0,ThreeInOne);
    plot1DRes(h1RZ_W1,h2RZ_W1,h3RZ_W1,ThreeInOne);
    plot1DRes(h1RZ_W2,h2RZ_W2,h3RZ_W2,ThreeInOne);
  }

  if (doResVsPos){
    // Res, Rphi
    plot1DResVsPos(h1RPhi,h2RPhi,h3RPhi,ThreeInOne);

    // Res, RZ 
    plot1DResVsPos(h1RZ,h2RZ,h3RZ,ThreeInOne);

    // Res, RZ, per wheel
    plot1DResVsPos(h1RZ_W0,h2RZ_W0,h3RZ_W0,ThreeInOne);
    plot1DResVsPos(h1RZ_W1,h2RZ_W1,h3RZ_W1,ThreeInOne);
    plot1DResVsPos(h1RZ_W2,h2RZ_W2,h3RZ_W2,ThreeInOne);
  }

  if (doResCloseWire){
    // Res, Rphi
    plot1DResCloseWire(h1RPhi,h2RPhi,h3RPhi,ThreeInOne);

    // Res, RZ 
    plot1DResCloseWire(h1RZ,h2RZ,h3RZ,ThreeInOne);

    // Res, RZ, per wheel
    plot1DResCloseWire(h1RZ_W0,h2RZ_W0,h3RZ_W0,ThreeInOne);
    plot1DResCloseWire(h1RZ_W1,h2RZ_W1,h3RZ_W1,ThreeInOne);
    plot1DResCloseWire(h1RZ_W2,h2RZ_W2,h3RZ_W2,ThreeInOne);
  }
  if (doResVsAngle){
    // Res, Rphi
    plot1DResVsAngle(h1RPhi,h2RPhi,h3RPhi,ThreeInOne);

    // Res, RZ 
    plot1DResVsAngle(h1RZ,h2RZ,h3RZ,ThreeInOne);

    // Res, RZ, per wheel
    plot1DResVsAngle(h1RZ_W0,h2RZ_W0,h3RZ_W0,ThreeInOne);
    plot1DResVsAngle(h1RZ_W1,h2RZ_W1,h3RZ_W1,ThreeInOne);
    plot1DResVsAngle(h1RZ_W2,h2RZ_W2,h3RZ_W2,ThreeInOne);
  }

  if (doResVsFE){
    // Res, Rphi
    plot1DResVsFE(h1RPhi,h2RPhi,h3RPhi,ThreeInOne);

    // Res, RZ 
    plot1DResVsFE(h1RZ,h2RZ,h3RZ,ThreeInOne);

    // // Res, RZ, per wheel
    // plot1DResVsFE(h1RZ_W0,h2RZ_W0,h3RZ_W0,ThreeInOne);
    // plot1DResVsFE(h1RZ_W1,h2RZ_W1,h3RZ_W1,ThreeInOne);
    // plot1DResVsFE(h1RZ_W2,h2RZ_W2,h3RZ_W2,ThreeInOne);
  }
}

void plot1DRes(HRes1DHit* h1, HRes1DHit* h2, HRes1DHit* h3, bool ThreeInOne) {
  int i = 2;

  if(ThreeInOne)
    cout << "ThreeInOne = true!" << endl;
  else
    cout << "ThreeInOne = false!" << endl;

  TString N1 = "c_1D_" + h1->name;
  TString N2 = "c_1D_" + h2->name;
  TString N3 = "c_1D_" + h3->name;
  int form = 2;
  if (ThreeInOne) c1 = newCanvas(N1.Remove(5,2)+"_hRes",3,1,800,400);
  else newCanvas(N1+"_hRes",form);
  h1->hRes->SetXTitle("(d_{reco}-d_{sim})");
  h1->hRes->SetYTitle("# events");
  drawGFit(h1->hRes, -.15,.15,-.15,.15);

  if (ThreeInOne) c1->cd(i++);
  else c1 = newCanvas(N2+"_hRes",form);
  h2->hRes->SetXTitle("(d_{reco}-d_{sim})");
  h2->hRes->SetYTitle("# events");
  drawGFit(h2->hRes,  -.15,.15,-.15,.15);

  if (ThreeInOne) c1->cd(i++);
  else c1 = newCanvas(N3+"_hRes",form);
  h3->hRes->SetXTitle("(d_{reco}-d_{sim})");
  h3->hRes->SetYTitle("# events");
  drawGFit(h3->hRes,  -.15,.15,-.15,.15);

}

void plot1DResVsPos(HRes1DHit* h1, HRes1DHit* h2, HRes1DHit* h3, bool ThreeInOne) {
  int i = 2;

  if(ThreeInOne)
    cout << "ThreeInOne = true!" << endl;
  else
    cout << "ThreeInOne = false!" << endl;

  TString N1 = "c_1D_" + h1->name;
  TString N2 = "c_1D_" + h2->name;
  TString N3 = "c_1D_" + h3->name;
  int form = 2;
  if (ThreeInOne) c1 = newCanvas(N1.Remove(5,2)+"_hResVsPos",3,1,800,400);
  else newCanvas(N1+"_hResVsPos",form);
  plotAndProfileXSpread(h1->hResVsPos,  0.,2.1 ,true,-.3,.3);

  if (ThreeInOne) c1->cd(i++);
  else c1 = newCanvas(N2+"_hResVsPos",form);
  plotAndProfileXSpread(h2->hResVsPos, 0.,2.1 ,true,-.5,.5);

  if (ThreeInOne) c1->cd(i++);
  else c1 = newCanvas(N3+"_hResVsPos",form);
  plotAndProfileXSpread(h3->hResVsPos, 0.,2.1 ,true,-.5,.5);

}

void plot1DResCloseWire(HRes1DHit* h1, HRes1DHit* h2, HRes1DHit* h3, bool ThreeInOne) {
  int i = 2;

  if(ThreeInOne)
    cout << "ThreeInOne = true!" << endl;
  else
    cout << "ThreeInOne = false!" << endl;

  TString N1 = "c_1D_" + h1->name;
  TString N2 = "c_1D_" + h2->name;
  TString N3 = "c_1D_" + h3->name;
  int form = 2;
  if (ThreeInOne) c1 = newCanvas(N1.Remove(5,2)+"_hResVsPos",3,1,800,400);
  else newCanvas(N1+"_hResCloseWire",form);
  drawCloseWire(h1->hResVsPos, 10);

  if (ThreeInOne) c1->cd(i++);
  else c1 = newCanvas(N2+"_hResCloseWire",form);
  drawCloseWire(h2->hResVsPos, 10);

  if (ThreeInOne) c1->cd(i++);
  else c1 = newCanvas(N3+"_hResCloseWire",form);
  drawCloseWire(h3->hResVsPos, 10);

}

void drawCloseWire(TH2* h2, int border) {
  gPad->SetGrid(1,1);
  gStyle->SetGridColor(15);
  TH1D* hall= h2->ProjectionY("_all",-1,-1);
  TH1D* hclose= h2->ProjectionY("_cl",-1,border);
  TH1D* hfar= h2->ProjectionY("_far",border+1,-1);
  //hfar->DrawNormalized("",hfar->GetEntries());
  hall->DrawCopy();
  hfar->SetLineColor(4);
  hfar->DrawCopy("same");
  //hclose->DrawNormalized("same",hfar->GetEntries()/2.);
  hclose->SetLineColor(2);
  hclose->DrawCopy("same");

}

void plot1DResVsAngle(HRes1DHit* h1, HRes1DHit* h2, HRes1DHit* h3, bool ThreeInOne) {
  int i = 2;

  TString N1 = "c_1D_" + h1->name;
  TString N2 = "c_1D_" + h2->name;
  TString N3 = "c_1D_" + h3->name;
  int form = 2;
  if (ThreeInOne) c1 = newCanvas(N1.Remove(5,2)+"_hResVsAngle",3,1,800,400);
  else newCanvas(N1+"_hResVsAngle",form);
  plotAndProfileXSpread(h1->hResVsAngle,  0.,.6 ,true,-0.2,0.2);

  if (ThreeInOne) c1->cd(i++);
  else c1 = newCanvas(N2+"_hResVsAngle",form);
  plotAndProfileXSpread(h2->hResVsAngle, 0.,.6 ,true,-0.2,0.2);

  if (ThreeInOne) c1->cd(i++);
  else c1 = newCanvas(N3+"_hResVsAngle",form);
  plotAndProfileXSpread(h3->hResVsAngle, 0.,.6 ,true,-0.2,0.2);

}

void plot1DResVsFE(HRes1DHit* h1, HRes1DHit* h2, HRes1DHit* h3, bool ThreeInOne) {
  int i = 2;

  TString N1 = "c_1D_" + h1->name;
  TString N2 = "c_1D_" + h2->name;
  TString N3 = "c_1D_" + h3->name;
  int form = 2;
  if (ThreeInOne) c1 = newCanvas(N1.Remove(5,2)+"_hResVsDistFE",3,1,800,400);
  else newCanvas(N1+"_hResVsDistFE",form);
  plotAndProfileXSpread(h1->hResVsDistFE,  0.,400. ,true,-0.2,0.2);

  if (ThreeInOne) c1->cd(i++);
  else c1 = newCanvas(N2+"_hResVsDistFE",form);
  plotAndProfileXSpread(h2->hResVsDistFE, 0.,400. ,true,-0.2,0.2);

  if (ThreeInOne) c1->cd(i++);
  else c1 = newCanvas(N3+"_hResVsDistFE",form);
  plotAndProfileXSpread(h3->hResVsDistFE, 0.,400. ,true,-0.2,0.2);

}
bool setPreferences(bool& doRes,
                    bool& doResVsPos,
                    bool& doResCloseWire,
                    bool& doResVsAngle,
                    bool& doResVsFE,
                    bool& ThreeIn1) {

  int dimension = 0;

  cout << "===================================================" << endl;
  cout << "==== plotHitReso User Menu =====================" << endl;
  cout << "Chose the plot you want to produce:" << endl;
  cout << "1 - 1D RecHit Res" << endl;
  cout << "2 - 1D RecHit Res vs Pos" << endl;
  cout << "3 - 1D RecHit Res close wire" << endl;
  cout << "4 - 1D RecHit Res vs angle" << endl;
  cout << "5 - 1D RecHit Res vs FE distance" << endl;
  cout << "-->";
  cin >> dimension;

  switch(dimension) 
  {
    case 1:
      {
        doRes = true;
        break;
      }
    case 2:
      {
        doResVsPos = true;
        break;
      }
    case 3:
      {
        doResCloseWire = true;
        break;
      }
    case 4:
      {
        doResVsAngle = true;
        break;
      }
    case 5:
      {
        doResVsFE = true;
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
