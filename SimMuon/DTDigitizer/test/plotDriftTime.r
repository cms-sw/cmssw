
/********************************************************************** 
 * Various tests of the parametrization funtions.
 * Execute with: .x plotDriftTime.r
 *
 * Author: G. Bevilacqua, N. Amapane
 * 
 **********************************************************************/

#include <iostream>
#include "TGraph.h"
#include "TString.h"
#include "TPostScript.h"
#include "TCanvas.h"

// in cm!
#define DT_Cell_HalfWidth 2.1  

typedef struct {
  double v_drift, t_drift, delta_t, t_width_m, t_width_p ;
} drift_time ;

typedef struct {
  double v_drift, x_drift, delta_x, x_width_m, x_width_p ;
} drift_distance;


// Create graphs for parametrized drift time as a function one of the parameters.
TGraphAsymmErrors * doGraph(int funct,       // which of the pars varies (0->3)
			    Double_t pars[3],// The other pars
			    int Nx, Double_t * xv, // x array
			    bool old,        // use the old parametrization
			    short interpol   // use interpolation
			    ){
  drift_time * DT;
  
  static const int maxsize = 200;
  Double_t tv[maxsize];
  Double_t stm[maxsize];
  Double_t stp[maxsize];
  
  if (Nx>maxsize) return 0;

  Double_t par[4]; //   x,alpha,bWire,bNorm,interpol

  // Assign fixed parameters
  int idum = 0;
  for (int i=0; i<4; i++) {
    if (i != funct) {
      par[i] = pars[idum];
      idum++;
    }
  }

  for(int i=0; i<Nx; i++) {
    par[funct] = xv[i];
    if(old){
      tv[i] = oldParametrization(par[0], par[1], par[2], par[3]);
    } else {
      DT = driftTime( par[0], par[1], par[2], par[3], interpol);
      tv[i] = DT->t_drift;
      stm[i] = DT->t_width_m;
      stp[i] = DT->t_width_p;
    }
  }

  TGraphAsymmErrors  *grx;
  if(old){
    grx = new TGraphAsymmErrors(Nx,xv,tv,0,0,0,0);
  } else {
    grx = new TGraphAsymmErrors(Nx,xv,tv,0,0,stm,stp);
  }
  return grx;
}

// Plot graphs of the drift time (created with doGraph)
void plotGraph(TString &title,
	       TGraphAsymmErrors *grx1,
	       TGraphAsymmErrors *grx2,
	       TGraphAsymmErrors  *grx3) 
{
  grx1->GetYaxis()->SetTitle("time (ns)");
  grx1->SetTitle(title.Data());
  grx1->SetLineColor(1);
  grx1->SetMarkerColor(1);
  grx1->SetMarkerStyle(2);
  grx1->SetMarkerSize(.4);
  grx1->SetMinimum(0);
  grx1->SetMaximum(500);
  grx1->Draw("AP");

  grx2->SetLineColor(4);
  grx2->SetMarkerColor(4);
  grx2->SetMarkerStyle(22);
  grx2->SetMarkerSize(.7); 
  grx2->Draw("P");

  grx3->SetLineColor(2);
  grx3->SetMarkerColor(2);
  grx3->SetMarkerStyle(21);
  grx3->SetMarkerSize(.4); 
  grx3->Draw("P");
}


// Option 1: Plot drift time as a function of x
void DtVSx(Double_t alpha, Double_t bWire, Double_t bNorm, Int_t ifl, Int_t flagCanvas, Int_t flagFile)
{
  static const Int_t N_x = 84;  //passo x = 0.5 mm
  Double_t xarray[N_x];

  float max = DT_Cell_HalfWidth+0.1;

  float step = max*2./N_x;

  for(Int_t i=0; i<N_x; i++) {
    xarray[i] = i*step - max;
  }

  Double_t par[3];
  par[0] = alpha;
  par[1] = bWire;
  par[2] = bNorm;

  TGraphAsymmErrors *gOld  = doGraph(0,par,N_x,xarray,true,0);
  TGraphAsymmErrors *gNew  = doGraph(0,par,N_x,xarray,false,0);
  TGraphAsymmErrors *gNewI = doGraph(0,par,N_x,xarray,false,1);

  TString title = "Drift time for #alpha=" + dToTString(alpha) + 
    ", Bnorm =" + dToTString(bNorm) + 
    ", Bwire =" + dToTString(bWire);
  TCanvas *cx=0;
  if(flagCanvas==1) {
    cx = new TCanvas(title,"Drift time as a function of x", 600,400);
  }
  
  plotGraph(title, gOld,gNew,gNewI);
  
  if(flagFile==1) {
    TString path ="Plots/PlotDtVSx/"+title+".ps"; 
    path.ReplaceAll(" ","_");
    path.ReplaceAll("#","");
    cx->Print(path.Data());
  }
}

//  Option 2: Plot drift time as a function of alpha
void DtVSalpha(Double_t x, Double_t bWire, Double_t bNorm, Int_t ifl, Int_t flagCanvas, Int_t flagFile)
{

  static const Int_t N_alpha = 120;
  float max = 60.;

  //---
  float step = max*2./N_alpha; 
  Double_t xarray[N_alpha];
 
  for(Int_t i=0; i<N_alpha; i++) {
    xarray[i] = i*step - max;
  }

  Double_t par[3];
  par[0] = x;
  par[1] = bWire;
  par[2] = bNorm;

  TGraphAsymmErrors *gNew  = doGraph(1,par,N_alpha,xarray,false,0);
  TGraphAsymmErrors *gNewI = doGraph(1,par,N_alpha,xarray,false,1);
  TGraphAsymmErrors *gOld  = doGraph(1,par,N_alpha,xarray,true,0);


  TString title = "Drift time for x=" + dToTString(x) + 
    ", Bnorm =" + dToTString(bNorm) + 
    ", Bwire =" + dToTString(bWire);
  TCanvas *ca=0;
  if(flagCanvas==1) {
    ca = new TCanvas(title,"Drift time as a function of alpha", 600,400);
  }

  plotGraph(title, gOld,gNew,gNewI);

  if(flagFile==1) {
    TString path ="Plots/PlotDtVSalpha/"+title+".ps"; 
    path.ReplaceAll(" ","_");
    path.ReplaceAll("#","");
    ca->Print(path.Data());
  }
}

//  Option 3: Plot drift time as a function of Bwire
void DtVSbWire(Double_t x, Double_t alpha, Double_t bNorm, Int_t ifl, Int_t flagCanvas, Int_t flagFile)
{
  static const Int_t Nx = 50;
  float max = 0.5;

  //---
  float step = max*2./Nx;
  Double_t xarray[Nx];
 
  for(Int_t i=0; i<Nx; i++) {
    xarray[i] = i*step - max;
  }

  Double_t par[3];
  par[0] = x;
  par[1] = alpha;
  par[2] = bNorm;

  TGraphAsymmErrors *gOld  = doGraph(2,par,Nx,xarray,true,0);
  TGraphAsymmErrors *gNew  = doGraph(2,par,Nx,xarray,false,0);
  TGraphAsymmErrors *gNewI = doGraph(2,par,Nx,xarray,false,1);

  TString title = "Drift time for x=" + dToTString(x) +
    ", #alpha=" + dToTString(alpha) +
    ", Bnorm =" + dToTString(bNorm);
  TCanvas *cbWire=0;
  if (flagCanvas==1) {
    cbWire = new TCanvas(title,"Drift time as a function of bWire", 600,400);
  }

  plotGraph(title, gOld,gNew,gNewI);

  if(flagFile==1)
    {
      TString path ="Plots/PlotDtVSbWire/"+title+".ps"; 
      path.ReplaceAll(" ","_");
      path.ReplaceAll("#","");
      cbWire->Print(path.Data());
    }
      
}

//  Option 4: Plot drift time as a function of Bnorm
void DtVSbNorm(Double_t x, Double_t alpha, Double_t bWire, Int_t ifl, Int_t flagCanvas, Int_t flagFile)
{

  static const Int_t Nx = 85;
  float max = 0.85;

  //---
  float step = max/Nx;
  Double_t xarray[Nx];

  for(Int_t i=0; i<Nx; i++) {
    xarray[i] = i*step;
  }

  Double_t par[3];
  par[0] = x;
  par[1] = alpha;
  par[2] = bWire;

  TGraphAsymmErrors *gOld  = doGraph(3,par,Nx,xarray,true,0);
  TGraphAsymmErrors *gNew  = doGraph(3,par,Nx,xarray,false,0);
  TGraphAsymmErrors *gNewI = doGraph(3,par,Nx,xarray,false,1);

  TString title = "Drift time for x=" + dToTString(x) +
    ", #alpha=" + dToTString(alpha) +
    ", Bwire =" + dToTString(bWire);
  TCanvas *cbNorm=0;
  if(flagCanvas==1) {
    cbNorm = new TCanvas(title,"Drift time as a function of bNorm", 600,400);
  }

  plotGraph(title, gOld,gNew,gNewI);

  if(flagFile==1)
    {
      TString path ="Plots/PlotDtVSbNorm/"+title+".ps";
      path.ReplaceAll(" ","_");
      path.ReplaceAll("#","");
      cbNorm->Print(path.Data());
    }

}



//  Option 6: Simulate a time box
void timeBox(double alpha, double bWire, double bNorm, bool old){
  short interpol = 1;
  
  int smear = 1;
  if (!old) {
    cout << "Use smearing of times? [0/1]" <<endl;
    cin >> smear;
  }

  int ntracks = 50000;

  cout << "Simulation of " << ntracks << " tracks with " << (old?"old":"new") 
       << " parametrization " << endl
       << " Smearing of times " << ((smear&&(!old))?"ON":"OFF") << endl;
  

  TH1D *hTBox = new TH1D("hTBox","Time box",275,-50,500);
  TRandom rnd;
  drift_time * DT;
  for (int i = 0; i<ntracks; i++) {
    float x = rnd.Uniform(-2.1,2.1);
    if (old) {
      hTBox->Fill(oldParametrization(x,alpha,bWire,bNorm));
    } else {
      float dt;
      if (smear) {
	dt = smearedTime(x,alpha,bWire,bNorm, interpol);
      } else {
	DT = driftTime(x,alpha,bWire,bNorm, interpol);	
	dt = DT->t_drift;
      }
      hTBox->Fill(dt);
    }
  }
  TCanvas *cx=new TCanvas("c_hTBox","Time box");
  hTBox->Draw();
}

// Option 5: Print value computed by the parametrization 
void printDt(double x, double alpha, double bWire, double bNorm, int ifl)
{
  short interpol = 0;

  drift_time * DT;
  DT = driftTime(x,alpha,bWire,bNorm, interpol);

  cout<<endl<<"driftTime(x="<<x<<",alpha="<<alpha<<",bWire="<<bWire<<",bNorm="<<bNorm<<",ifl="<<ifl<<") = "<<DT->t_drift<<" ns"<<endl;

  cout<<"sigma_r = "<<DT->t_width_p<<" ns"<<endl;
  cout<<"sigma_l = "<<DT->t_width_m<<" ns"<<endl;

}


// Used by option 9, 
float plotNoSmearing(float alpha, float bWire, float bNorm, short interpol) {  
  drift_time * DT;
  drift_distance * DX;

  TMarker * m = new TMarker();
  m->SetMarkerColor(kBlue);
  m->SetMarkerStyle(2);
  for (int i =0; i < 210; i++){
    float x = i/100.;
    DT = driftTime(x, alpha, bWire, bNorm, interpol);
    float dt = DT->t_drift;
    DX = trackDistance(dt,alpha,bWire,bNorm, interpol);
    m->DrawMarker(x,DX->x_drift/10.-x);
  }
}

// Used by option 9,
void resPullPlots(float alpha=0, float bWire=0, float bNorm=0, short interpol=1) {
  gROOT->LoadMacro("macros.C");

  TStyle * style = getStyle();
  style->SetOptStat("OURMEN");
  //  style->SetOptStat("RME");
  style->SetOptFit(101);
  style->cd();

  int form = 1;

  //  TFile * f = new TFile("MBHitAnalysis_NewDigis.root");
  //  TFile * f = new TFile("MBHitAnalysis_NewDigis_onlyMu.root");

  TCanvas * c1;  

  //  goto resvspos2;

  TCanvas *cx=new TCanvas("c_hTBox","Time box");
  hTBox->Draw();

  c1 = newCanvas("c_hPos",2,1,form);
  c1->cd(1);
  hPos->Draw();
  c1->cd(2);
  hPosC->Draw();

  c1 = newCanvas("c_hRes",2,1,form);
  c1->cd(1);
  drawGFit(&hRes, -0.2,0.2,-0.1,0.1);
  c1->cd(2);
  drawGFit(&hResC, -0.2,0.2,-0.1,0.1);


  c1 = newCanvas("c_hPull",2,1,form);
  c1->cd(1);
  drawGFit(&hPull, -5,5);
  c1->cd(2);
  drawGFit(&hPullC, -5,5);

 resvspos:

  c1 = newCanvas("c_hResVsPos");
  plotAndProfileX(&hResVsPos,-0.1,0.1,true);
  plotNoSmearing(alpha, bWire, bNorm, interpol);
  
  c1 = newCanvas("c_hResVsPosC");
  plotAndProfileX(&hResVsPosC,-0.1,0.1,true);
  plotNoSmearing(alpha, bWire, bNorm, interpol);  


 pullvspos:
  c1 = newCanvas("c_hPullVsPos");
  plotAndProfileX(&hPullVsPos,-5,8, true);
  c1 = newCanvas("c_hPullVsPosC");  
  plotAndProfileX(&hPullVsPosC,-5,8, true);

 end:
}

// Option 9: simulate resolution and pulls
void resPull(double alpha, double bWire, double bNorm){

  int ntracks = 50000;
  bool smear=true;
  //smear = false;
  short interpol = 1;
  bool secondLevelCorr = true;
  secondLevelCorr = false;

  TH1F * hTBox = new TH1F ("hTBox","Time box",275,-50,500);
  TH1F * hPos = new TH1F ("hPos", "RHit position", 100, 0,2.5);
  TH1F * hRes  = new TH1F ("hRes", "RHit residual", 1290, -4.3,4.3);
  TH1F * hPull = new TH1F ("hPull", "RHit pull", 100, -5, 5);
  TH2F * hResVsPos = new TH2F("hResVsPos", "RHit residual vs position",
			      100, 0,2.5, 1290, -4.3,4.3);    
  TH2F * hPullVsPos = new TH2F("hPullVsPos", "RHit pull vs position",
			       100, 0,2.5, 130, -5.,8.);    

  TH2F * hTPosVsPos = new TH2F("hTPosVsPos", "True pos vs rec pos",
			       220, 0,2.2, 220, 0,2.2);    
  TH2F * hratio = new TH2F("hratio", "RHit residual vs position",
 			   220, 0,2.2, 200, 0., 1.5);

  //   TH2F * hTPosVsPos1 = new TH2F("hTPosVsPos1", "True pos vs rec pos",
  // 			      220, 0,2.2, 220, 0,2.2);    
  //   TH2F * hResVsPos1 = new TH2F("hResVsPos1", "RHit residual vs position",
  // 			      100, 0,2.5, 1290, -4.3,4.3);    



  TH1F * hPosC = new TH1F ("hPosC", "RHit position", 100, 0,2.5);
  TH1F * hResC  = new TH1F ("hResC", "RHit residual", 1290, -4.3,4.3);
  TH1F * hPullC = new TH1F ("hPullC", "RHit pull", 100, -5, 5);
  TH1F * hPullC_fixed = new TH1F ("hPullC_fixed", "RHit pull", 100, -5, 5);
  TH2F * hResVsPosC = new TH2F("hResVsPosC", "RHit residual vs position",
			       100, 0,2.5, 860, -4.3,4.3);    
  TH2F * hPullVsPosC = new TH2F("hPullVsPosC", "RHit pull vs position",
				100, 0,2.5, 130, -5.,8.);    


  TRandom rnd;
  drift_time * DT;
  drift_distance * DX;
  for (int i = 0; i<ntracks; i++) {
    float x = rnd.Uniform(-2.1,2.1);
    float dt;
    if(smear) {
      dt = smearedTime(x,alpha,bWire,bNorm, interpol);
    } else {
      DT = driftTime(x, alpha, bWire, bNorm, interpol);
      dt = DT->t_drift;
    }
    DX = trackDistance(dt,alpha,bWire,bNorm, interpol);
    float rX =  DX->x_drift/10.;
    float res = rX - fabs(x);
    float pull = res*20./(DX->x_width_m + DX->x_width_p);
    hTBox->Fill(dt);
    hPos->Fill(rX);
    hRes->Fill(res);
    hPull->Fill(pull);
    hResVsPos->Fill(fabs(x),res);
    hPullVsPos->Fill(fabs(x),pull);


    hTPosVsPos->Fill(fabs(x),rX);
    hratio->Fill(rX,fabs(x)/rX);

    //     TF1 * fun = new TF1("fun", "pol2", 0, 0.2);
    //     fun->SetParameter(0,-3.80066e-01);
    //     fun->SetParameter(1, 1.36233e+01);
    //     fun->SetParameter(2,-3.49056e+01);

    //     float corX = rX;
    //     if (corX<0.2) corX *= fun->Eval(rX);
    
    //     hTPosVsPos1->Fill(fabs(x),corX);
    //     hResVsPos1->Fill(fabs(x),corX-fabs(x));


    // correct for difference mean - mean value    
    float dX = 0.;
    dX = (DX->x_width_p - DX->x_width_m)*sqrt(2/TMath::Pi())/10.;
    float rXC = TMath::Max(0,rX-dX);
    float sigma = (DX->x_width_m + DX->x_width_p)/20.;
    if (secondLevelCorr && fabs(x)<0.4) { //FIXME!!!
      DT = driftTime(rXC, alpha, bWire, bNorm, interpol); //SIGN
      float dX1 = (DT->t_width_p - DT->t_width_m)*sqrt(2/TMath::Pi())*DT->v_drift/10.;
      rXC = TMath::Max(0,rX-dX1);
      //      sigma = (DT->t_width_p * DT->t_width_m)*DT->v_drift/20.;
    }
    
    res = rXC - fabs(x);
    pull = res/sigma;
    hPosC->Fill(rXC);
    hResC->Fill(res);
    hPullC->Fill(pull);
    hPullC_fixed->Fill(res/0.01423);
    hResVsPosC->Fill(fabs(x),res);
    hPullVsPosC->Fill(fabs(x),pull);
  }

  resPullPlots(alpha, bWire, bNorm, interpol);

  cout << endl
       << "number of tracks:         " << ntracks << endl
       << "time smearing:            " << (smear?"ON":"OFF") << endl
       << "second-level correction:  " << (secondLevelCorr?"ON":"OFF") << endl
       << "interpolation:            " << (interpol?"ON":"OFF") << endl
       << endl
       << "Plots with a name ending by C include peak-mean correction." << endl
       << "red markers: profile of scatter plots" << endl
       << "blue markers: behaviour without smearing" << endl
       << endl;
}


// Option 10: Verify the peak-mean correction for the asymmetric arrival time distribution

void plotAsymDist(double mean, double sigma1, double sigma2) {
  int ntracks = 500000;

  TH1D *hParam = new TH1D("hParam","Double half-gaussian",500,0,500);
  
  for (int i = 0; i<ntracks; i++) {
    double x = asymGausSample(mean, sigma1, sigma2);
    hParam->Fill(x);
  }

  TCanvas *c1 = new TCanvas("c_Param","Double half-gaussian sampling");
  hParam->Draw();

  cout << endl << "The mean calculated from the parametrized t, sigma_L, sigma_R is:" << endl
       << "mean+(sigma_R-sigma_L)*sqrt(2./pi) = "
       << mean+(sigma2-sigma1)*sqrt(2./TMath::Pi()) <<endl;

}

//  Option 11: Simulate the mean timer distribution
void meanTimer(double alpha, double bWire, double bNorm, bool old){
  
  int smear = 1;
  if (!old) {
    cout << "Use smearing of times? [0/1]" <<endl;
    cin >> smear;
  }

  int ntracks = 50000;
  
  cout << "Simulation of " << ntracks << " tracks with " << (old?"old":"new") 
       << " parametrization " << endl
       << " Smearing of times " << ((smear&&(!old))?"ON":"OFF") << endl;
  

  TH1D *hMeanTimer = new TH1D("hMeanTimer","Mean timer",275,340,420);
  TRandom rnd;
  
  for (int i = 0; i<ntracks; i++) {
    float x = rnd.Uniform(0,2.1);
    float t1=DriftTimeChoosing(x,alpha,bWire,bNorm,old,smear);
    float t2=DriftTimeChoosing(2.1-x,alpha,bWire,bNorm,old,smear);
    float t3=DriftTimeChoosing(x,alpha,bWire,bNorm,old,smear);
    
    hMeanTimer->Fill( (t1+t3)/2 + t2 );
  }
  TCanvas *cx=new TCanvas("c_hMeanTimer","Mean Timer");
  hMeanTimer->Draw();
}

double DriftTimeChoosing(double x,double alpha, double bWire, double bNorm, bool old, bool smear){
  short interpol = 1;
  drift_time * DT;
  if (old)
    return oldParametrization(x,alpha,bWire,bNorm);
  else {
    float dt;
    if (smear) 
      dt = smearedTime(x,alpha,bWire,bNorm, interpol);
    else {
      DT = driftTime(x,alpha,bWire,bNorm, interpol);	
      dt = DT->t_drift;
    }
    return dt;
  }
}



int plot()
{
  cout<<endl<<"Drift time in a muon chamber"<<endl;
  cout<<endl<<"[x] = cm , -2.1 cm < x < 2.1 cm"<<endl;
  cout<<"[dt] = ns"<<endl; 
  cout<<"[alpha] = degrees , -30 < alpha < 30"<<endl;
  cout<<"[bWire] = T, |bWire| < 0.4"<<endl;
  cout<<"[bNorm] = T, |bNorm| < 0.75 (symmetric)"<<endl;

  Double_t dt, x, alpha, bWire, bNorm;
  Int_t ifl, flc, flf, fli;
  ifl = 0;
  flc = 1; //flagCanvas (for DtVS<...>): 0 = no creating canvas
  flf = 0; //flagFile (for DtVS<...>): 0 = no saving; 1 = save to file
  fli = 0; //flagIsto (for ShowGaussian): 0 = curve; 1 = curve + 500ev sampling

  cout << endl
       << "Choose: " << endl
       << " 1 Plot drift time as a function of x"<<endl
       << " 2 Plot drift time as a function of alpha"<<endl
       << " 3 Plot drift time as a function of bWire"<<endl
       << " 4 Plot drift time as a function of bNorm"<<endl
       << " 5 Predefined collection of the above plots"<<endl
       << " 6 Compute drift time for the given parameters"<<endl
       << " 7 Simulated time box (new parametrization)"<<endl
       << " 8 Simulated time box (old parametrization)"<<endl
       << " 9 Simulated resolution and pulls"<<endl
       << " 10 Verify the peak-mean correction for the arrival time distribution"<<endl  
       << " 11 Mean Timer distribution"<<endl;  

       Int_t n1 = 0;
  cin>>n1;

  if(n1==1) {
    cout<<"Insert alpha value [deg]: ";
    cin>>alpha;
    cout<<"Insert bWire value [T]: ";
    cin>>bWire;
    cout<<"Insert bNorm value [T]: ";
    cin>>bNorm;
      
    cout<<endl<<"  alpha = "<<alpha<<endl;
    cout<<"  bWire = "<<bWire<<endl;
    cout<<"  bNorm = "<<bNorm<<endl;
    cout<<"  ifl = "<<ifl<<endl;
      
    DtVSx(alpha,bWire,bNorm,ifl,flc,flf);

    helpDt();

  } 

  else if(n1==2) {
    cout<<"Insert x value [cm from wire]: ";
    cin>>x;
    cout<<"Insert bWire value: [T]";
    cin>>bWire;
    cout<<"Insert bNorm value: [T]";
    cin>>bNorm;

    cout<<endl<<"  x = "<<x<<endl;
    cout<<"  bWire = "<<bWire<<endl;
    cout<<"  bNorm = "<<bNorm<<endl;
    cout<<"  ifl = "<<ifl<<endl;

    DtVSalpha(x,bWire,bNorm,ifl,flc,flf);
    helpDt();

  } 

  else if(n1==3) {
    cout<<"Insert x value [cm from wire]: ";
    cin>>x;
    cout<<"Insert alpha value [deg]: ";
    cin>>alpha;
    cout<<"Insert bNorm value [T]: ";
    cin>>bNorm;

    cout<<endl<<"  x = "<<x<<endl;
    cout<<"  alpha = "<<alpha<<endl;
    cout<<"  bNorm = "<<bNorm<<endl;
    cout<<"  ifl = "<<ifl<<endl;

    DtVSbWire(x,alpha,bNorm,ifl,flc,flf);
    helpDt();

  } 

  else if(n1==4) {
    cout<<"Insert x value [cm from wire]: ";
    cin>>x;
    cout<<"Insert alpha value [deg]: ";
    cin>>alpha;
    cout<<"Insert bWire value [T]: ";
    cin>>bWire;

    cout<<endl<<"  x = "<<x<<endl;
    cout<<"  alpha = "<<alpha<<endl;
    cout<<"  bWire = "<<bWire<<endl;
    cout<<"  ifl = "<<ifl<<endl;

    DtVSbNorm(x,alpha,bWire,ifl,flc,flf);
    helpDt();

  } 

  else if(n1==5) {
    Double_t x, alpha, bWire, bNorm;
    

    cX = new TCanvas("DTvsX","Drift time as a function of x", 600,400);
    int i=1;
    cX->Divide(4,3);    
    cX->cd(i++);
    alpha=0; bWire=0; bNorm=0;
    DtVSx(alpha,bWire,bNorm,0,0,0);
    cX->cd(i++);
    alpha=0; bWire=0.4; bNorm=0.7;
    DtVSx(alpha,bWire,bNorm,0,0,0);
    cX->cd(i++);
    alpha=0; bWire=0; bNorm=0;
    DtVSx(alpha,bWire,bNorm,0,0,0);
    cX->cd(i++);
    alpha=0; bWire=0.4; bNorm=0.7;
    DtVSx(alpha,bWire,bNorm,0,0,0);

    cX->cd(i++);
    alpha=30; bWire=0; bNorm=0;
    DtVSx(alpha,bWire,bNorm,0,0,0);
    cX->cd(i++);
    alpha=30; bWire=0.4; bNorm=0.7;
    DtVSx(alpha,bWire,bNorm,0,0,0);
    cX->cd(i++);
    alpha=30; bWire=0; bNorm=0;
    DtVSx(alpha,bWire,bNorm,0,0,0);
    cX->cd(i++);
    alpha=30; bWire=0.4; bNorm=0.7;
    DtVSx(alpha,bWire,bNorm,0,0,0);

    cX->cd(i++);
    alpha=55; bWire=0; bNorm=0;
    DtVSx(alpha,bWire,bNorm,0,0,0);
    cX->cd(i++);
    alpha=55; bWire=0.4; bNorm=0.7;
    DtVSx(alpha,bWire,bNorm,0,0,0);
    cX->cd(i++);
    alpha=55; bWire=0; bNorm=0;
    DtVSx(alpha,bWire,bNorm,0,0,0);
    cX->cd(i++);
    alpha=55; bWire=0.4; bNorm=0.7;
    DtVSx(alpha,bWire,bNorm,0,0,0);

    cX = new TCanvas("DTvsalpha","Drift time as a function of alpha", 600,400);
    i=1;
    cX->Divide(4,3);    
    cX->cd(i++);
    x=0.5; bWire=0; bNorm=0;
    DtVSalpha(x,bWire,bNorm,0,0,0);
    cX->cd(i++);
    x=0.5; bWire=0.4; bNorm=0;
    DtVSalpha(x,bWire,bNorm,0,0,0);
    cX->cd(i++);
    x=0.5; bWire=0; bNorm=0.75;
    DtVSalpha(x,bWire,bNorm,0,0,0);
    cX->cd(i++);
    x=0.5; bWire=0.4; bNorm=0.75;
    DtVSalpha(x,bWire,bNorm,0,0,0);
    
    cX->cd(i++);
    x=1; bWire=0; bNorm=0;
    DtVSalpha(x,bWire,bNorm,0,0,0);
    cX->cd(i++);
    x=1; bWire=0.4; bNorm=0;
    DtVSalpha(x,bWire,bNorm,0,0,0);
    cX->cd(i++);
    x=1; bWire=0; bNorm=0.75;
    DtVSalpha(x,bWire,bNorm,0,0,0);
    cX->cd(i++);
    x=1; bWire=0.4; bNorm=0.75;
    DtVSalpha(x,bWire,bNorm,0,0,0);

    cX->cd(i++);
    x=1.5; bWire=0; bNorm=0;
    DtVSalpha(x,bWire,bNorm,0,0,0);
    cX->cd(i++);
    x=1.5; bWire=0.4; bNorm=0;
    DtVSalpha(x,bWire,bNorm,0,0,0);
    cX->cd(i++);
    x=1.5; bWire=0; bNorm=0.75;
    DtVSalpha(x,bWire,bNorm,0,0,0);
    cX->cd(i++);
    x=1.5; bWire=0.4; bNorm=0.75;
    DtVSalpha(x,bWire,bNorm,0,0,0);


    cX = new TCanvas("DTvsbWire","Drift time as a function of bWire", 600,400);
    i=1;
    cX->Divide(4,3);    
    cX->cd(i++);
    x=0.5; alpha=0; bNorm=0;
    DtVSbWire(x,alpha,bNorm,0,0,0);
    cX->cd(i++);
    x=0.5; alpha=30; bNorm=0;
    DtVSbWire(x,alpha,bNorm,0,0,0);
    cX->cd(i++);
    x=0.5; alpha=0; bNorm=0.4;
    DtVSbWire(x,alpha,bNorm,0,0,0);
    cX->cd(i++);
    x=0.5; alpha=30; bNorm=0.4;
    DtVSbWire(x,alpha,bNorm,0,0,0);

    cX->cd(i++);
    x=1; alpha=0; bNorm=0;
    DtVSbWire(x,alpha,bNorm,0,0,0);
    cX->cd(i++);
    x=1; alpha=30; bNorm=0;
    DtVSbWire(x,alpha,bNorm,0,0,0);
    cX->cd(i++);
    x=1; alpha=0; bNorm=0.4;
    DtVSbWire(x,alpha,bNorm,0,0,0);
    cX->cd(i++);
    x=1; alpha=30; bNorm=0.4;
    DtVSbWire(x,alpha,bNorm,0,0,0);

    cX->cd(i++);
    x=1.5; alpha=0; bNorm=0;
    DtVSbWire(x,alpha,bNorm,0,0,0);
    cX->cd(i++);
    x=1.5; alpha=30; bNorm=0;
    DtVSbWire(x,alpha,bNorm,0,0,0);
    cX->cd(i++);
    x=1.5; alpha=0; bNorm=0.4;
    DtVSbWire(x,alpha,bNorm,0,0,0);
    cX->cd(i++);
    x=1.5; alpha=30; bNorm=0.4;
    DtVSbWire(x,alpha,bNorm,0,0,0);


    cX = new TCanvas("DTvsbNorm","Drift time as a function of bNorm", 600,400);
    i=1;
    cX->Divide(4,3);
    cX->cd(i++);
    x=0.5; alpha=0; bWire=0;
    DtVSbNorm(x,alpha,bWire,0,0,0);
    cX->cd(i++);
    alpha=30; bWire=0;
    DtVSbNorm(x,alpha,bWire,0,0,0);
    cX->cd(i++);
    alpha=0; bWire=0.4;
    DtVSbNorm(x,alpha,bWire,0,0,0);
    cX->cd(i++);
    alpha=30; bWire=0.4;
    DtVSbNorm(x,alpha,bWire,0,0,0);
    
    cX->cd(i++);
    x=1; alpha=0; bWire=0;
    DtVSbNorm(x,alpha,bWire,0,0,0);
    cX->cd(i++);
    alpha=30; bWire=0;
    DtVSbNorm(x,alpha,bWire,0,0,0);
    cX->cd(i++);
    alpha=0; bWire=0.4;
    DtVSbNorm(x,alpha,bWire,0,0,0);
    cX->cd(i++);
    alpha=30; bWire=0.4;
    DtVSbNorm(x,alpha,bWire,0,0,0);

    cX->cd(i++);
    x=1.5; alpha=0; bWire=0;
    DtVSbNorm(x,alpha,bWire,0,0,0);
    cX->cd(i++);
    alpha=30; bWire=0;
    DtVSbNorm(x,alpha,bWire,0,0,0);
    cX->cd(i++);
    alpha=0; bWire=0.4;
    DtVSbNorm(x,alpha,bWire,0,0,0);
    cX->cd(i++);
    alpha=30; bWire=0.4;
    DtVSbNorm(x,alpha,bWire,0,0,0);
  } 

  else if(n1==6) {
    cout<<"Insert x value [cm from wire]: ";
    cin>>x;
    cout<<"Insert alpha value [deg]: ";
    cin>>alpha;
    cout<<"Insert bWire value [T]: ";
    cin>>bWire;
    cout<<"Insert bNorm value [T]: ";
    cin>>bNorm;

    cout<<endl<<"  x = "<<x<<endl;
    cout<<"  alpha = "<<alpha<<endl;
    cout<<"  bWire = "<<bWire<<endl;
    cout<<"  bNorm = "<<bNorm<<endl;
    cout<<"  ifl = "<<ifl<<endl;

    // ... To be implemented: draw the reconstructed arrival time distribution.

    printDt(x,alpha,bWire,bNorm,ifl);

  }

  else if(n1==7||n1==8) {
    cout<<"Insert alpha value [deg]: ";
    cin>>alpha;
    cout<<"Insert bWire value [T]: ";
    cin>>bWire;
    cout<<"Insert bNorm value [T]: ";
    cin>>bNorm;

    cout<<endl<<"  alpha = "<<alpha<<endl;
    cout<<"  bWire = "<<bWire<<endl;
    cout<<"  bNorm = "<<bNorm<<endl;

    bool old = false;
    if (n1==8) old = true;
    timeBox(alpha,bWire,bNorm,old);

  }

  else if(n1==9) {
    cout<<"Insert alpha value [deg]: ";
    cin>>alpha;
    cout<<"Insert bWire value [T]: ";
    cin>>bWire;
    cout<<"Insert bNorm value [T]: ";
    cin>>bNorm;

    cout<<endl<<"  alpha = "<<alpha<<endl;
    cout<<"  bWire = "<<bWire<<endl;
    cout<<"  bNorm = "<<bNorm<<endl;

    resPull(alpha, bWire, bNorm);  
  }

  else if(n1==10) {
    
    double mean = 200;
    double sigma1 = 10;
    double sigma2 = 100;

    plotAsymDist(mean, sigma1, sigma2);
  }

 else if(n1==11) {
   int old= 0;
    cout<<"Insert alpha value [deg]: ";
    cin>>alpha;
    cout<<"Insert bWire value [T]: ";
    cin>>bWire;
    cout<<"Insert bNorm value [T]: ";
    cin>>bNorm;
    cout<<"Old[1] or New[0] parametrization? ";
    cin >> old;
    cout<<endl;
    cout<<"  alpha = "<<alpha<<endl;
    cout<<"  bWire = "<<bWire<<endl;
    cout<<"  bNorm = "<<bNorm<<endl;
    cout<<"  ifl = "<<ifl<<endl;

    meanTimer(alpha,bWire,bNorm,old);

  }





 
  else {
    cout<<"There's something wrong. n1= " << n1 << endl;
    n1 = 0;
    return 1;
  }
  

  cout<<endl<<"Execution successful"<<endl;
  return 0;
}


// Print a simple legend for plots...
void helpDt() {
  cout << endl << "Legend:" <<endl
       << "Red:   new parametrization (Puerta, Garcia-Abia), with interpolation" << endl
       << "Blue:  new parametrization (Puerta, Garcia-Abia), no interpolation" << endl
       << "Black: old parametrization (Gresele, Rovelli)" <<endl;
}


// The entry point.
// just call .x plotDriftTime.r
void plotDriftTime() {
  //   gROOT->GetList()->Delete();
  //   gROOT->GetListOfCanvases()->Delete();

  gSystem->Load("libCLHEP");
  gSystem->Load("libtestSimMuonDTDigitizer.so");
  plot();
}




