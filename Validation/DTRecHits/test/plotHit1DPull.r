
class HRes1DHit;
class HRes2DHit;
class HRes4DHit;

void plotHitPull();
void draw(bool, bool, bool, bool, bool, bool ThreeInOne);
void plot1DPulls(HRes1DHit * h1, HRes1DHit * h2, HRes1DHit * h3, bool ThreeInOne);
void plot1DPullsVsPos(HRes1DHit * h1, HRes1DHit * h2, HRes1DHit * h3, bool ThreeInOne);
bool setPreferences(bool& doPulls, bool& doPullsVsPos, bool& doPullsCloseWire, bool&, bool&, bool& ThreeIn1) ;

// This is the main function
void plotHit1DPull(){
  // Load needed macros and files
  gROOT->LoadMacro("macros.C");     // Load service macros
  gROOT->LoadMacro("../plugins/Histograms.h"); // Load definition of histograms

  // Get the style
  TStyle * style = getStyle("tdr");
  /// TStyle * style = getStyle();

  // //style->SetOptStat("RME");
  // style->SetOptStat(0);
  // style->SetOptFit(11);
  style->SetFitFormat("4.4g");
  // style->SetStatFontSize(0.03);
  // style->SetStatColor(0);
  style->SetStatY(0.99);
  style->SetStatX(0.99);
  style->SetTitleYOffset(1.6);
  style->SetLabelSize(0.035, "XYZ");
  // style->SetStatH(0.3);
  // style->SetStatW(0.25);
  // // style->SetStatW(0.25);
  // // style->SetStatH(0.35);
  // style->SetTitleFillColor(0);
  // style->SetLabelSize(0.03);

  style->cd();                      // Apply style 

  //Main switches
  bool ThreeInOne = true;  // Plot the 3 steps in a single canvas (where appl.)
  bool doPulls = false; 
  bool doPullsVsPos = false; 
  bool doPullsCloseWire = false; 
  bool doPullsVsAngle = false; 
  bool doPullsVsFE = false; 
  // Read user input, namely what plots should be produced
  
  while(!setPreferences(doPulls, doPullsVsPos, doPullsCloseWire, doPullsVsAngle, doPullsVsFE, ThreeInOne));

  draw(doPulls, doPullsVsPos, doPullsCloseWire, doPullsVsAngle, doPullsVsFE, ThreeInOne);
}

void draw(bool doPulls, bool doPullsVsPos, bool doPullsCloseWire, bool doPullsVsAngle, bool doPullsVsFE, bool ThreeInOne) {
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

  if (doPulls) {
    // Pull, Rphi
    plot1DPulls(h1RPhi,h2RPhi,h3RPhi,ThreeInOne);

    // Pull, RZ 
    plot1DPulls(h1RZ,h2RZ,h3RZ,ThreeInOne);

    // Pull, RZ, per wheel
    plot1DPulls(h1RZ_W0,h2RZ_W0,h3RZ_W0,ThreeInOne);
    plot1DPulls(h1RZ_W1,h2RZ_W1,h3RZ_W1,ThreeInOne);
    plot1DPulls(h1RZ_W2,h2RZ_W2,h3RZ_W2,ThreeInOne);
  }

  if (doPullsVsPos){
    // Pull, Rphi
    plot1DPullsVsPos(h1RPhi,h2RPhi,h3RPhi,ThreeInOne);

    // Pull, RZ 
    plot1DPullsVsPos(h1RZ,h2RZ,h3RZ,ThreeInOne);

    // // Pull, RZ, per wheel
    // plot1DPullsVsPos(h1RZ_W0,h2RZ_W0,h3RZ_W0,ThreeInOne);
    // plot1DPullsVsPos(h1RZ_W1,h2RZ_W1,h3RZ_W1,ThreeInOne);
    // plot1DPullsVsPos(h1RZ_W2,h2RZ_W2,h3RZ_W2,ThreeInOne);
  }

  if (doPullsCloseWire){
    // Pull, Rphi
    plot1DPullsCloseWire(h1RPhi,h2RPhi,h3RPhi,ThreeInOne);

    // Pull, RZ 
    plot1DPullsCloseWire(h1RZ,h2RZ,h3RZ,ThreeInOne);

    // Pull, RZ, per wheel
    plot1DPullsCloseWire(h1RZ_W0,h2RZ_W0,h3RZ_W0,ThreeInOne);
    plot1DPullsCloseWire(h1RZ_W1,h2RZ_W1,h3RZ_W1,ThreeInOne);
    plot1DPullsCloseWire(h1RZ_W2,h2RZ_W2,h3RZ_W2,ThreeInOne);
  }
  if (doPullsVsAngle){
    // Pull, Rphi
    plot1DPullsVsAngle(h1RPhi,h2RPhi,h3RPhi,ThreeInOne);

    // Pull, RZ 
    plot1DPullsVsAngle(h1RZ,h2RZ,h3RZ,ThreeInOne);

    // Pull, RZ, per wheel
    plot1DPullsVsAngle(h1RZ_W0,h2RZ_W0,h3RZ_W0,ThreeInOne);
    plot1DPullsVsAngle(h1RZ_W1,h2RZ_W1,h3RZ_W1,ThreeInOne);
    plot1DPullsVsAngle(h1RZ_W2,h2RZ_W2,h3RZ_W2,ThreeInOne);
  }

  if (doPullsVsFE){
    // Pull, Rphi
    plot1DPullsVsFE(h1RPhi,h2RPhi,h3RPhi,ThreeInOne);

    // Pull, RZ 
    plot1DPullsVsFE(h1RZ,h2RZ,h3RZ,ThreeInOne);

    // Pull, RZ, per wheel
    plot1DPullsVsFE(h1RZ_W0,h2RZ_W0,h3RZ_W0,ThreeInOne);
    plot1DPullsVsFE(h1RZ_W1,h2RZ_W1,h3RZ_W1,ThreeInOne);
    plot1DPullsVsFE(h1RZ_W2,h2RZ_W2,h3RZ_W2,ThreeInOne);
  }
}

void plot1DPulls(HRes1DHit* h1, HRes1DHit* h2, HRes1DHit* h3, bool ThreeInOne) {
  int i = 2;

  if(ThreeInOne)
    cout << "ThreeInOne = true!" << endl;
  else
    cout << "ThreeInOne = false!" << endl;

  TString N1 = "c_1D_" + h1->name;
  TString N2 = "c_1D_" + h2->name;
  TString N3 = "c_1D_" + h3->name;
  int form = 2;
  if (ThreeInOne) c1 = newCanvas(N1.Remove(5,2)+"_hPull",3,1,800,400);
  else newCanvas(N1+"_hPull",form);
  h1->hPull->SetXTitle("(d_{reco}-d_{sim})/#sigma_{reco}");
  h1->hPull->SetYTitle("# events");
  drawGFit(h1->hPull, -5.,5.,-5.,5.);

  if (ThreeInOne) c1->cd(i++);
  else c1 = newCanvas(N2+"_hPull",form);
  h2->hPull->SetXTitle("(d_{reco}-d_{sim})/#sigma_{reco}");
  h2->hPull->SetYTitle("# events");
  drawGFit(h2->hPull,  -5.,5.,-5.,5.);

  if (ThreeInOne) c1->cd(i++);
  else c1 = newCanvas(N3+"_hPull",form);
  h3->hPull->SetXTitle("(d_{reco}-d_{sim})/#sigma_{reco}");
  h3->hPull->SetYTitle("# events");
  drawGFit(h3->hPull,  -5.,5.,-5.,5.);

}

void plot1DPullsVsPos(HRes1DHit* h1, HRes1DHit* h2, HRes1DHit* h3, bool ThreeInOne) {
  int i = 2;

  if(ThreeInOne)
    cout << "ThreeInOne = true!" << endl;
  else
    cout << "ThreeInOne = false!" << endl;

  TString N1 = "c_1D_" + h1->name;
  TString N2 = "c_1D_" + h2->name;
  TString N3 = "c_1D_" + h3->name;
  int form = 2;
  if (ThreeInOne) c1 = newCanvas(N1.Remove(5,2)+"_hPullVsPos",3,1,800,400);
  else newCanvas(N1+"_hPullVsPos",form);
  plotAndProfileXSpread(h1->hPullVsPos,  0.,2.1 ,true);

  if (ThreeInOne) c1->cd(i++);
  else c1 = newCanvas(N2+"_hPullVsPos",form);
  plotAndProfileXSpread(h2->hPullVsPos, 0.,2.1 ,true);

  if (ThreeInOne) c1->cd(i++);
  else c1 = newCanvas(N3+"_hPullVsPos",form);
  plotAndProfileXSpread(h3->hPullVsPos, 0.,2.1 ,true);

}

void plot1DPullsCloseWire(HRes1DHit* h1, HRes1DHit* h2, HRes1DHit* h3, bool ThreeInOne) {
  int i = 2;

  if(ThreeInOne)
    cout << "ThreeInOne = true!" << endl;
  else
    cout << "ThreeInOne = false!" << endl;

  TString N1 = "c_1D_" + h1->name;
  TString N2 = "c_1D_" + h2->name;
  TString N3 = "c_1D_" + h3->name;
  int form = 2;
  if (ThreeInOne) c1 = newCanvas(N1.Remove(5,2)+"_hPullVsPos",3,1,800,400);
  else newCanvas(N1+"_hPullCloseWire",form);
  drawCloseWire(h1->hPullVsPos, 10);

  if (ThreeInOne) c1->cd(i++);
  else c1 = newCanvas(N2+"_hPullCloseWire",form);
  drawCloseWire(h2->hPullVsPos, 10);

  if (ThreeInOne) c1->cd(i++);
  else c1 = newCanvas(N3+"_hPullCloseWire",form);
  drawCloseWire(h3->hPullVsPos, 10);

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

void plot1DPullsVsAngle(HRes1DHit* h1, HRes1DHit* h2, HRes1DHit* h3, bool ThreeInOne) {
  int i = 2;

  TString N1 = "c_1D_" + h1->name;
  TString N2 = "c_1D_" + h2->name;
  TString N3 = "c_1D_" + h3->name;
  int form = 2;
  if (ThreeInOne) c1 = newCanvas(N1.Remove(5,2)+"_hPullVsAngle",3,1,800,400);
  else newCanvas(N1+"_hPullVsAngle",form);
  plotAndProfileXSpread(h1->hPullVsAngle,  0.,2. ,true);

  if (ThreeInOne) c1->cd(i++);
  else c1 = newCanvas(N2+"_hPullVsAngle",form);
  plotAndProfileXSpread(h2->hPullVsAngle, 0.,2. ,true);

  if (ThreeInOne) c1->cd(i++);
  else c1 = newCanvas(N3+"_hPullVsAngle",form);
  plotAndProfileXSpread(h3->hPullVsAngle, 0.,2. ,true);

}

void plot1DPullsVsFE(HRes1DHit* h1, HRes1DHit* h2, HRes1DHit* h3, bool ThreeInOne) {
  int i = 2;

  TString N1 = "c_1D_" + h1->name;
  TString N2 = "c_1D_" + h2->name;
  TString N3 = "c_1D_" + h3->name;
  int form = 2;
  if (ThreeInOne) c1 = newCanvas(N1.Remove(5,2)+"_hPullVsDistFE",3,1,800,400);
  else newCanvas(N1+"_hPullVsDistFE",form);
  plotAndProfileXSpread(h1->hPullVsDistFE,  0.,300. ,true, -5.,5.);

  if (ThreeInOne) c1->cd(i++);
  else c1 = newCanvas(N2+"_hPullVsDistFE",form);
  plotAndProfileXSpread(h2->hPullVsDistFE, 0.,300. ,true, -5.,5.);

  if (ThreeInOne) c1->cd(i++);
  else c1 = newCanvas(N3+"_hPullVsDistFE",form);
  plotAndProfileXSpread(h3->hPullVsDistFE, 0.,300. ,true, -5.,5.);

}

bool setPreferences(bool& doPulls,
                    bool& doPullsVsPos,
                    bool& doPullsCloseWire,
                    bool& doPullsVsAngle,
                    bool& doPullsVsFE,
                    bool& ThreeIn1) {

  int dimension = 0;

  cout << "===================================================" << endl;
  cout << "==== plotHitReso User Menu =====================" << endl;
  cout << "Chose the plot you want to produce:" << endl;
  cout << "1 - 1D RecHit Pulls" << endl;
  cout << "2 - 1D RecHit Pulls vs Pos" << endl;
  cout << "3 - 1D RecHit Pulls close wire" << endl;
  cout << "4 - 1D RecHit Pulls vs angle" << endl;
  cout << "5 - 1D RecHit Pulls vs FE distance" << endl;
  cout << "-->";
  cin >> dimension;

  switch(dimension) 
  {
    case 1:
      {
        doPulls = true;
        break;
      }
    case 2:
      {
        doPullsVsPos = true;
        break;
      }
    case 3:
      {
        doPullsCloseWire = true;
        break;
      }
    case 4:
      {
        doPullsVsAngle = true;
        break;
      }
    case 5:
      {
        doPullsVsFE = true;
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
