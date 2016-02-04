#define SimpleVertexAnalysis_cxx
#include "SimpleVertexAnalysis.h"
#include "TH2.h"
#include "TF1.h"
#include "TStyle.h"
#include "TPostScript.h"
#include <iostream>
#include <fstream>
#include <iomanip>

using std::cout;
using std::endl;

// .L SimpleVertexAnalysis.C++
// t = new SimpleVertexAnalysis ("simpleVertexTree.root","VertexFitter")
// t->vertexLoop()
// t->trackLoop()


void SimpleVertexAnalysis::vertexLoop()
{
   if (fChain == 0) return;

   Int_t nentries = Int_t(fChain->GetEntriesFast());
   bookVertexHisto();
   Int_t nbytes = 0, nb = 0;
   total=0; failure=0;
   for (Int_t jentry=0; jentry<nentries;jentry++) {
      Int_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      // if (Cut(ientry) < 0) continue;
      ++total;
      if ((vertex==3)&&(nbrTrk_Rec>0)) {
	resX->Fill((simPos_X-recPos_X)*10000.);
	resY->Fill((simPos_Y-recPos_Y)*10000.);
	resZ->Fill((simPos_Z-recPos_Z)*10000.);
	pullX->Fill((simPos_X-recPos_X)/recErr_X);
	pullY->Fill((simPos_Y-recPos_Y)/recErr_Y);
	pullZ->Fill((simPos_Z-recPos_Z)/recErr_Z);
	chiNorm->Fill(chiTot/ndf);
	chiProbability->Fill(chiProb);
	weight->Fill((ndf+3.)/2.);
	int tt = ((recTracks==0) ? nbrTrk_Rec : recTracks);
	normWeight->Fill((ndf+3.)/2. / tt);
	downWeight->Fill(((ndf+3.)/2.) - tt);
	numberUsedRecTracks->Fill(nbrTrk_Rec);
	numberRawRecTracks->Fill(recTracks);
// 	numberSimTracks->Fill(nbrTrk_Sim);
	sharedTracks->Fill(nbrTrk_Shared);
// 	ratioSharedTracks->Fill(nbrTrk_Shared/ nbrTrk_Rec/*(int)( (recTracks==0) ? nbrTrk_Rec : recTracks */);
	timing->Fill(time);
   } else {//if ((vertex==1)||(nbrTrk_Rec<1.1)) {
     ++failure;
// 	timing->Fill(time);
	numberSimTracks->Fill(nbrTrk_Sim);
	numberRawRecTracks->Fill(recTracks);
// 	cout << vertex<<" "<<nbrTrk_Rec<<endl;
   }
  }
//   plotVertexResult();
}


void SimpleVertexAnalysis::vertexCoverage(ostream &out)
{
   if (fChain == 0) return;

   Int_t nentries = Int_t(fChain->GetEntriesFast());
   Int_t nbytes = 0, nb = 0;
   vector<float> xResiduals, yResiduals, zResiduals;
   for (Int_t jentry=0; jentry<nentries;jentry++) {
      Int_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      if (vertex==3) {
	xResiduals.push_back(fabs(simPos_X-recPos_X)*10000.);
	yResiduals.push_back(fabs(simPos_Y-recPos_Y)*10000.);
	zResiduals.push_back(fabs(simPos_Z-recPos_Z)*10000.);
     }
  }
  out << "Coverage:      50%       90%        95% \n";
  out.width(10);out << "X-coord.:"; x_coverage = getCoverage(xResiduals, out);
  out.width(10);out << "Y-coord.:"; y_coverage = getCoverage(yResiduals, out);
  out.width(10);out << "Z-coord.:"; z_coverage = getCoverage(zResiduals, out);
}


float SimpleVertexAnalysis::getCoverage(vector<float> &residuals, ostream &out)
{
  float retVal;
  sort(residuals.begin(),residuals.end());
  out.width(10);out << residuals[(int)(residuals.size()*0.5)];
  out.width(10);out << residuals[(int)(residuals.size()*0.9)];
  out.width(10);out << (retVal = residuals[(int)(residuals.size()*0.95)]);
  out << endl;
  return retVal;
}




void SimpleVertexAnalysis::vertexHistoLimits(float aMaxTrans, float aMaxZ, 
	float aMaxTracks,  float aMaxWeights, float aMaxTime)
{
  maxTrans = aMaxTrans;
  maxZ = aMaxZ;
  maxTracks = aMaxTracks;
  maxWeights = aMaxWeights;
  maxTime = aMaxTime;
}


void SimpleVertexAnalysis::bookVertexHisto()
{
  if (bookedVertex) deleteVertexHisto();
  bookedVertex=true;
  resX = new TH1F("ResX"+theTreeName, "Residual x coordinate: "+theTreeName, 50, -maxTrans, maxTrans);
  resY = new TH1F("ResY"+theTreeName, "Residual y coordinate: "+theTreeName, 50, -maxTrans, maxTrans);
  resZ = new TH1F("ResZ"+theTreeName, "Residual z coordinate: "+theTreeName, 50, -maxZ, maxZ);
  pullX = new TH1F("PullX"+theTreeName, "Pull x coordinate: "+theTreeName, 100, -10., 10.);
  pullY = new TH1F("PullY"+theTreeName, "Pull y coordinate: "+theTreeName, 100, -10., 10.);
  pullZ = new TH1F("PullZ"+theTreeName, "Pull z coordinate: "+theTreeName, 100, -10., 10.);
  chiNorm = new TH1F("ChiNorm"+theTreeName, "Normalized chi-square: " +theTreeName, 100, 0., 10.);
  chiProbability = new TH1F("ChiProb"+theTreeName, "Chi-square probability: "+theTreeName, 100, 0., 1.);
  weight = new TH1F("weight"+theTreeName, "Sum of track weights: " +theTreeName, 100, 0., maxWeights);
  normWeight = new TH1F("normWeight"+theTreeName, "Normalized Sum of track weights: " +theTreeName, 100, -0.1, 1.1);
  downWeight = new TH1F("downWeight"+theTreeName, "negative of track weights: " +theTreeName, 100, -20., 0.1);
  numberUsedRecTracks = new TH1F("usedRecTrk"+theTreeName, "Number of RecTracks used: " +theTreeName, 21, -0.5, maxTracks);
  numberRawRecTracks  = new TH1F("rawRecTrk"+theTreeName, "Number of RecTracks given: " +theTreeName, 21, -0.5, maxTracks);
  numberSimTracks = new TH1F("SimTrk"+theTreeName, "Number of SimTracks: " +theTreeName, 21, -0.5, maxTracks);
  sharedTracks = new TH1F("Shared"+theTreeName, "Number of shared tracks: " +theTreeName, 21, -0.5, maxTracks);
  ratioSharedTracks = new TH1F("Ratio"+theTreeName, "Ratio of shared tracks: " +theTreeName, 100, 0., 2.);
  timing = new TH1F("Timing"+theTreeName, "Time per fit: " +theTreeName, 100, 0., maxTime);
  resX->SetXTitle("Res. x coord. [#mum] "/*+theTreeName*/);
  resY->SetXTitle("Res. y coord. [#mum] "/*+theTreeName*/);
  resZ->SetXTitle("Res. z coord. [#mum] "/*+theTreeName*/);
  pullX->SetXTitle("Pull x coord. "/*+theTreeName*/);
  pullY->SetXTitle("Pull y coord. "/*+theTreeName*/);
  pullZ->SetXTitle("Pull z coord. "/*+theTreeName*/);
  chiNorm->SetXTitle("Normalized chi-square: " /*+theTreeName*/);
  chiProbability->SetXTitle("Chi-square probability"/*+theTreeName*/);
  weight->SetXTitle("Sum of track weights: " +theTreeName);
  normWeight->SetXTitle("Normalized Sum of track weights: " +theTreeName);
  numberUsedRecTracks->SetXTitle("Number of RecTracks used: " +theTreeName);
  numberRawRecTracks ->SetXTitle("Number of RecTracks given: " +theTreeName);
  numberSimTracks->SetXTitle("Number of SimTracks: " +theTreeName);
  sharedTracks->SetXTitle("Number of shared tracks: " +theTreeName);
  ratioSharedTracks->SetXTitle("Ratio of shared tracks: " +theTreeName);
  timing->SetXTitle("Time per fit [#mus] " +theTreeName);
}

void SimpleVertexAnalysis::trackLoop()
{
   if (fChain == 0) return;

   Int_t nentries = Int_t(fChain->GetEntriesFast());
   bookTrackHisto();
   Int_t nbytes = 0, nb = 0;
    for (Int_t jentry=0; jentry<nentries;jentry++) {
//   for (Int_t jentry=0; jentry<10;jentry++) {
//cout << "Event "<<jentry<<endl;
      Int_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      if (recTracks>MAXTRACK) {
        cout << "Number of Tracks in event exceeds Maximum."<<endl;
	cout << "Increase MAXTRACK to at least " << recTracks<< " (line 12 in SimpleVertexAnalysis.h)."<<endl;
      }
      for (int iTrack=0; ((iTrack<recTracks) && (iTrack<MAXTRACK)); ++iTrack) {
	pTRec->Fill(1/recPar_ptinv[iTrack]);
	etaRec->Fill(-log(tan(recPar_theta[iTrack]/2)));


        if (recTrack_simIndex[iTrack] != -1) {
	
// cout << " ptin "<<	  simPar_ptinv[recTrack_simIndex[iTrack]]<< " rec: " <<recPar_ptinv[iTrack]<< " err: " <<recErr_ptinv[iTrack];
// cout << " phi  "<<	  simPar_phi[recTrack_simIndex[iTrack]]<< " rec: " <<recPar_phi[iTrack]<< " err: " <<recErr_phi[iTrack];
// cout << " thet "<<	  simPar_theta[recTrack_simIndex[iTrack]]<< " rec: " <<recPar_theta[iTrack]<< " err: " <<recErr_theta[iTrack];
// cout << " timp "<<	  simPar_timp[recTrack_simIndex[iTrack]]<< " rec: " <<recPar_timp[iTrack]<< " err: " <<recErr_timp[iTrack];
// cout << " limp "<<	  simPar_limp[recTrack_simIndex[iTrack]]<< " rec: " <<recPar_limp[iTrack]<< " err: " <<recErr_limp[iTrack];
// cout <<endl;
	  resRecPt->Fill(simPar_ptinv[recTrack_simIndex[iTrack]]-recPar_ptinv[iTrack]);
	  resRecPhi->Fill(simPar_phi[recTrack_simIndex[iTrack]]-recPar_phi[iTrack]);
	  resRecTheta->Fill(simPar_theta[recTrack_simIndex[iTrack]]-recPar_theta[iTrack]);
	  resRecTimp->Fill(simPar_timp[recTrack_simIndex[iTrack]]-recPar_timp[iTrack]);
	  resRecLimp->Fill(simPar_limp[recTrack_simIndex[iTrack]]-recPar_limp[iTrack]);

	  pullRecPt->Fill((simPar_ptinv[recTrack_simIndex[iTrack]]-recPar_ptinv[iTrack])/recErr_ptinv[iTrack]);
	  pullRecPhi->Fill((simPar_phi[recTrack_simIndex[iTrack]]-recPar_phi[iTrack])/recErr_phi[iTrack]);
	  pullRecTheta->Fill((simPar_theta[recTrack_simIndex[iTrack]]-recPar_theta[iTrack])/recErr_theta[iTrack]);
	  pullRecTimp->Fill((simPar_timp[recTrack_simIndex[iTrack]]-recPar_timp[iTrack])/recErr_timp[iTrack]);
	  pullRecLimp->Fill((simPar_limp[recTrack_simIndex[iTrack]]-recPar_limp[iTrack])/recErr_limp[iTrack]);

	  pTSim->Fill(1/simPar_ptinv[recTrack_simIndex[iTrack]]);
	  etaSim->Fill(-log(tan(simPar_theta[recTrack_simIndex[iTrack]]/2)));

	  if (refPar_ptinv[iTrack]!=0.) {
	    resRefPt->Fill(simPar_ptinv[recTrack_simIndex[iTrack]]-refPar_ptinv[iTrack]);
	    resRefPhi->Fill(simPar_phi[recTrack_simIndex[iTrack]]-refPar_phi[iTrack]);
	    resRefTheta->Fill(simPar_theta[recTrack_simIndex[iTrack]]-refPar_theta[iTrack]);
	    resRefTimp->Fill(simPar_timp[recTrack_simIndex[iTrack]]-refPar_timp[iTrack]);
	    resRefLimp->Fill(simPar_limp[recTrack_simIndex[iTrack]]-refPar_limp[iTrack]);

	    pullRefPt->Fill((simPar_ptinv[recTrack_simIndex[iTrack]]-refPar_ptinv[iTrack])/refErr_ptinv[iTrack]);
	    pullRefPhi->Fill((simPar_phi[recTrack_simIndex[iTrack]]-refPar_phi[iTrack])/refErr_phi[iTrack]);
	    pullRefTheta->Fill((simPar_theta[recTrack_simIndex[iTrack]]-refPar_theta[iTrack])/refErr_theta[iTrack]);
	    pullRefTimp->Fill((simPar_timp[recTrack_simIndex[iTrack]]-refPar_timp[iTrack])/refErr_timp[iTrack]);
	    pullRefLimp->Fill((simPar_limp[recTrack_simIndex[iTrack]]-refPar_limp[iTrack])/refErr_limp[iTrack]);
	    pTRef->Fill(1/refPar_ptinv[iTrack]);
	    etaRef->Fill(-log(tan(refPar_theta[iTrack]/2)));
	  }
	}
     }
   }
//    plotTrackResult();
}

void SimpleVertexAnalysis::bookTrackHisto()
{
  if (bookedTrack) deleteTrackHisto();
  bookedTrack=true;
  resRecPt    = new TH1F("ResRecPtInv"+theTreeName, "Residual Reco 1/p_{T}: "+theTreeName, 100, -0.03, 0.03);
  resRecPhi   = new TH1F("ResRecPhi"+theTreeName, "Residual Reco #phi: "+theTreeName, 100, -0.003, 0.003);
  resRecTheta = new TH1F("ResRecTheta"+theTreeName, "Residual Reco #theta: "+theTreeName, 100, -0.003, 0.003);
  resRecTimp  = new TH1F("ResRecTimp"+theTreeName, "Residual Reco Transverse IP: "+theTreeName, 100, -0.03, 0.03);
  resRecLimp  = new TH1F("ResRecLimp"+theTreeName, "Residual Reco Longitudinal IP: "+theTreeName, 100, -0.04, 0.04);

  pullRecPt    = new TH1F("PullRecPtInv"+theTreeName, "Pull Reco 1/p_{T}: "+theTreeName, 100, -5., 5.);
  pullRecPhi   = new TH1F("PullRecPhi"+theTreeName, "Pull Reco #phi: "+theTreeName, 100, -5., 5.);
  pullRecTheta = new TH1F("PullRecTheta"+theTreeName, "Pull Reco #theta: "+theTreeName, 100, -5., 5.);
  pullRecTimp  = new TH1F("PullRecTimp"+theTreeName, "Pull Reco Transverse IP: "+theTreeName, 100, -5., 5.);
  pullRecLimp  = new TH1F("PullRecLimp"+theTreeName, "Pull Reco Longitudinal IP: "+theTreeName, 100, -5., 5.);

  resRefPt    = new TH1F("ResRefPtInv"+theTreeName, "Residual Refitted 1/p_{T}: "+theTreeName, 100, -0.03, 0.03);
  resRefPhi   = new TH1F("ResRefPhi"+theTreeName, "Residual Refitted #phi: "+theTreeName, 100, -0.003, 0.003);
  resRefTheta = new TH1F("ResRefTheta"+theTreeName, "Residual Refitted #theta: "+theTreeName, 100, -0.003, 0.003);
  resRefTimp  = new TH1F("ResRefTimp"+theTreeName, "Residual Refitted Transverse IP: "+theTreeName, 100, -0.02, 0.02);
  resRefLimp  = new TH1F("ResRefLimp"+theTreeName, "Residual Refitted Longitudinal IP: "+theTreeName, 100, -0.02, 0.02);

  pullRefPt    = new TH1F("PullRefPtInv"+theTreeName, "Pull Refitted 1/p_{T}: "+theTreeName, 100, -5., 5.);
  pullRefPhi   = new TH1F("PullRefPhi"+theTreeName, "Pull Refitted #phi: "+theTreeName, 100, -5., 5.);
  pullRefTheta = new TH1F("PullRefTheta"+theTreeName, "Pull Refitted #theta: "+theTreeName, 100, -5., 5.);
  pullRefTimp  = new TH1F("PullRefTimp"+theTreeName, "Pull Refitted Transverse IP: "+theTreeName, 100, -5., 5.);
  pullRefLimp  = new TH1F("PullRefLimp"+theTreeName, "Pull Refitted Longitudinal IP: "+theTreeName, 100, -5., 5.);

  pTSim   = new TH1F("pTSim"+theTreeName, "Simulated p_{T} distribution : "+theTreeName, 100, 0., 50.);
  etaSim  = new TH1F("etaSim"+theTreeName, "Simulated #eta distribution : "+theTreeName, 100, 0., 3.);
  pTRec   = new TH1F("pTRec"+theTreeName, "Reconstructed p_{T} distribution : "+theTreeName, 100, 0., 50.);
  etaRec  = new TH1F("etaRec"+theTreeName, "Reconstructed #eta distribution : "+theTreeName, 100, 0., 2.6);
  pTRef   = new TH1F("pTRef"+theTreeName, "Refitted p_{T} distribution : "+theTreeName, 100, 0., 50.);
  etaRef  = new TH1F("etaRef"+theTreeName, "Refitted #eta distribution : "+theTreeName, 100, 0., 3.);
  pTRec->SetXTitle("p_{T}[GeV/c]");

}

void SimpleVertexAnalysis::deleteTrackHisto()
{
  bookedTrack=false;
  delete resRecPt; delete resRecPhi; delete resRecTheta; delete resRecTimp; delete resRecLimp;
  delete pullRecPt; delete pullRecPhi; delete pullRecTheta; delete pullRecTimp; delete pullRecLimp;
  delete resRefPt; delete resRefPhi; delete resRefTheta; delete resRefTimp; delete resRefLimp;
  delete pullRefPt; delete pullRefPhi; delete pullRefTheta; delete pullRefTimp; delete pullRefLimp;
  delete pTSim; delete etaSim; delete pTRec; delete etaRec; delete pTRef; delete etaRef;
}

void SimpleVertexAnalysis::deleteVertexHisto()
{
  delete resX; delete resY; delete resZ;
  delete pullX; delete pullY; delete pullZ;
  delete chiNorm; delete chiProbability; delete weight; delete normWeight;delete downWeight;
  delete numberUsedRecTracks; delete numberRawRecTracks; delete numberSimTracks; delete sharedTracks; delete ratioSharedTracks;
  delete timing;
  bookedVertex=false;
};


SimpleVertexAnalysis::SimpleVertexAnalysis(TString filename, TString treeName)
{
  bookedVertex=false;bookedTrack=false;
  bookedVertexC=false;bookedTrackC=false;
  TFile *f = new TFile(filename);
  if (f) {
    TTree* tree = (TTree*)gDirectory->Get(treeName);
    if (tree) {
      Init(tree);
      nentries = Int_t(fChain->GetEntries());
      myChain = 0;
    } else {
      cout <<"Tree "<< treeName <<" does not exist \n";
    }
  } else {
    cout <<"File "<< filename <<" does not exist \n";
  }
  theTreeName = treeName;
  vertexHistoLimits();
}


SimpleVertexAnalysis::SimpleVertexAnalysis(TString base, int start, int end, 
		TString treeName)
{
  bookedVertex=false;bookedTrack=false;
  bookedVertexC=false;bookedTrackC=false;
  myChain = new TChain(treeName);
  char nr[4];
  for (int i = start; i<=end ; i++) {
    if (i<10) sprintf(nr,"%1.1d",i);
    else if (i<100) sprintf(nr,"%2.2d",i);
    else sprintf(nr,"%3.3d",i);
    TString name = base+"_"+nr+".root";
    myChain->Add(name);
  }
  Init(myChain);
  nentries = Int_t(fChain->GetEntries());
  theTreeName = treeName;
  vertexHistoLimits();
}

void SimpleVertexAnalysis::psVertexResult(TString name)
{
  if (!bookedVertexC) plotVertexResult();
  resCanvas->Print(name+".ps[");
  resCanvas->Draw();
  resCanvas->Print(name+".ps");
  statCanvas->Draw();
  statCanvas->Print(name+".ps");
  statCanvas->Print(name+".ps]");
}

void SimpleVertexAnalysis::singleGaussianVertexResult(ostream &out)
{
   resX->Fit("gaus","QO");
   resY->Fit("gaus","QO");
   resZ->Fit("gaus","QO");
   pullX->Fit("gaus","QO");
   pullY->Fit("gaus","QO");
   pullZ->Fit("gaus","QO");

   out << "Main results on "<<resX->GetEntries()<<" Events:\n";
   out << "Resolutions, X: "<< resX->GetFunction("gaus")->GetParameter(2)<<endl;
   out << "Resolutions, Y: "<< resY->GetFunction("gaus")->GetParameter(2)<<endl;
   out << "Resolutions, Z: "<< resZ->GetFunction("gaus")->GetParameter(2)<<endl;
   out << "Pull, X: "<< pullX->GetFunction("gaus")->GetParameter(2)<<endl;
   out << "Pull, Y: "<< pullY->GetFunction("gaus")->GetParameter(2)<<endl;
   out << "Pull, Z: "<< pullZ->GetFunction("gaus")->GetParameter(2)<<endl;
   out << "Mean nomalised chi**2: "<<chiNorm->GetMean()<<endl;
   out << "Mean chi**2-Probability: "<<chiProbability->GetMean()<<endl;
   out << "Mean CPU time: "<<(timing->GetMean())*1000.<<" microseconds\n";
   out << "Failure rate: "<<failure<<" - "<<failure/total<<endl;
}
void SimpleVertexAnalysis::statTeXResult(ostream &out)
{
   out <<  theTreeName<< " \t& " ;
   out.precision(3); out <<chiNorm->GetMean()<< " \t& " ;
   out.precision(3); out <<chiProbability->GetMean()<< " \t& " ;
   out.precision(3); out <<normWeight->GetMean()<< " \t& " ;
//   out.precision(3); out <<numberRawRecTracks->GetMean()<< " \t& " ;
//   out.precision(3); out <<numberSimTracks->GetMean()<< " \t& " ;
   out.precision(3); out <<(timing->GetMean())*1000.<< " \t& " ;
//   if (failure!=0) {  else  out <<total;
float rare = (float)failure/(float)total*100.;
  out.precision(2); out <<rare<< " \t\\\\ \n" ;
}
void SimpleVertexAnalysis::xTeXResult(ostream &out)
{
   out <<  theTreeName<< " \t& " ;
   out.precision(3); out <<resX->GetFunction("gaus")->GetParameter(2)<< " \t& " ;
   out.precision(3); out <<resX->GetFunction("gaus")->GetParameter(1)<< " \t& " ;
   out.precision(3); out <<x_coverage<< " \t& " ;
   out.precision(3); out <<pullX->GetFunction("gaus")->GetParameter(2);
   out<< " \t\\\\ \n" ;
}
void SimpleVertexAnalysis::zTeXResult(ostream &out)
{
   out <<  theTreeName<< " \t& " ;
   out.precision(3); out <<resZ->GetFunction("gaus")->GetParameter(2)<< " \t& " ;
   out.precision(3); out <<resZ->GetFunction("gaus")->GetParameter(1)<< " \t& " ;
   out.precision(3); out <<z_coverage<< " \t& " ;
   out.precision(3); out <<pullZ->GetFunction("gaus")->GetParameter(2);
   out<< " \t\\\\ \n" ;
}
void SimpleVertexAnalysis::resolutionTeXResult(ostream &out)
{
   out <<  theTreeName<< " \t& " ;
   out.precision(3); out <<resX->GetFunction("gaus")->GetParameter(2)<< " \t& " ;
   out.precision(3); out <<resX->GetFunction("gaus")->GetParameter(1)<< " \t& " ;
   out.precision(3); out <<x_coverage<< " \t& " ;
   out.precision(3); out <<pullX->GetFunction("gaus")->GetParameter(2)<< " \t& " ;
   out.precision(3); out <<resZ->GetFunction("gaus")->GetParameter(2)<< " \t& " ;
   out.precision(3); out <<resZ->GetFunction("gaus")->GetParameter(1)<< " \t& " ;
   out.precision(3); out <<z_coverage<< " \t& " ;
   out.precision(3); out <<pullZ->GetFunction("gaus")->GetParameter(2);
   out<< " \t\\\\ \n" ;
}


void SimpleVertexAnalysis::doubleGaussianVertexResult(ostream &out)
{
   out << "Main results\n";
   out << "Resolutions, X: ";doubleGaussianFit(resX);
   out << "Resolutions, Y: ";doubleGaussianFit(resY);
   out << "Resolutions, Z: ";doubleGaussianFit(resZ);
   out << "Pull, X: ";doubleGaussianFit(pullX);
   out << "Pull, Y: ";doubleGaussianFit(pullY);
   out << "Pull, Z: ";doubleGaussianFit(pullZ);
   out << "Mean nomalised chi**2: "<<chiNorm->GetMean()<<endl;
   out << "Mean chi**2-Probability: "<<chiProbability->GetMean()<<endl;
   out << "Mean CPU time: "<<(timing->GetMean())*1000.<<" microseconds\n";
   out << "Failure rate: "<<failure/total<<endl;
}


void SimpleVertexAnalysis::doubleGaussianFit(TH1F *plot, ostream &out)
{
  TF1 *myfit = new TF1("myfit","[0]*exp(-0.5*((x-[1])/[2])^2)+[3]*exp(-0.5*((x-[4])/[5])^2)");
  myfit->SetParameter(0, 1);
  myfit->SetParameter(1, plot->GetMean());
  myfit->SetParameter(2, plot->GetRMS()/2);
  myfit->SetParameter(3, 1);
  myfit->SetParameter(4, plot->GetMean());
  myfit->SetParameter(5, plot->GetRMS()*1.5);

  plot->Fit(myfit,"QO");
  out << myfit->GetParameter(2)<<" "<<myfit->GetParameter(5)<<endl;
}


void SimpleVertexAnalysis::plotVertexResult()
{
   if (bookedTrackC) {
     delete resCanvas; delete statCanvas;
   }
   resCanvas = new TCanvas(theTreeName+"Res",theTreeName+"Res", 600, 800);
   resCanvas->Divide(2,3);
   
   resCanvas->cd(1);
   resX->Draw();
   resCanvas->cd(2);
   pullX->Draw();
   resCanvas->cd(3);
   resY->Draw();
   resCanvas->cd(4);
   pullY->Draw();
   resCanvas->cd(5);
   resZ->Draw();
   resCanvas->cd(6);
   pullZ->Draw();
   resCanvas->Update();

   statCanvas = new TCanvas(theTreeName+"Stat",theTreeName+"Stat", 600, 800);
   statCanvas->Divide(2,3);
   statCanvas->cd(1);
   chiNorm->Draw();
   statCanvas->cd(2);
   chiProbability->Draw();
   statCanvas->cd(3);
   weight->Draw();
   statCanvas->cd(4);
   numberUsedRecTracks->SetLineColor(2);
   numberUsedRecTracks->Draw();
   numberRawRecTracks->SetLineColor(3);
   numberRawRecTracks->Draw("same");
   numberSimTracks->SetLineColor(4);
   numberSimTracks->Draw("same");
   statCanvas->cd(5);
   normWeight->Draw();
   statCanvas->cd(6);
   timing->Draw();
   statCanvas->Update();
   bookedTrackC=true;
}


void SimpleVertexAnalysis::epsVertexResult(TString name)
{
   epsPlot(resX, name);
   epsPlot(pullX, name);
   epsPlot(resY, name);
   epsPlot(pullY, name);
   epsPlot(resZ, name);
   epsPlot(pullZ, name);

   epsPlot(chiNorm, name);
   epsPlot(chiProbability, name);
   epsPlot(weight, name);
   epsPlot(numberUsedRecTracks, name);
   epsPlot(numberRawRecTracks, name);
   epsPlot(numberSimTracks, name);
   epsPlot(ratioSharedTracks, name);
   epsPlot(timing, name);
}


void SimpleVertexAnalysis::epsPlot(TH1F *plot, TString name)
{
   TCanvas *tc = new TCanvas();
   plot->Draw();
   tc->Print(TString(name+plot->GetName())+".eps");
   delete tc;
}



void SimpleVertexAnalysis::plotTrackResult()
{
   if (bookedVertexC) {
       delete resRecCanvas;delete pullRecCanvas; delete resRefCanvas; delete pullRefCanvas; delete distCanvas;
     }
   resRecCanvas = new TCanvas(theTreeName+"ResRec",theTreeName+" Reconstrcuted Track Parameter Residuals", 900, 600);
   resRecCanvas->Divide(3,2);

   TStyle *fitStyle = new TStyle("Default","Fit Style");
   fitStyle->SetLineWidth(1);
   fitStyle->SetFuncColor(2);

//    gStyle->SetFuncStyle(fitStyle);
   resRecCanvas->cd(1);
   resRecPt->Fit("gaus");
   resRecCanvas->cd(2);
   resRecPhi->Fit("gaus");
   resRecCanvas->cd(3);
   resRecTheta->Fit("gaus");
   resRecCanvas->cd(4);
   resRecTimp->Fit("gaus");
   resRecCanvas->cd(5);
   resRecLimp->Fit("gaus");
   resRecCanvas->Update();

   pullRecCanvas = new TCanvas(theTreeName+"PullRec",theTreeName+" Reconstrcuted Track Parameter Pulls", 900, 600);
   pullRecCanvas->Divide(3,2);

   pullRecCanvas->cd(1);
   pullRecPt->Fit("gaus");
   pullRecCanvas->cd(2);
   pullRecPhi->Fit("gaus");
   pullRecCanvas->cd(3);
   pullRecTheta->Fit("gaus");
   pullRecCanvas->cd(4);
   pullRecTimp->Fit("gaus");
   pullRecCanvas->cd(5);
   pullRecLimp->Fit("gaus");
   pullRecCanvas->Update();

   resRefCanvas = new TCanvas(theTreeName+"ResRef",theTreeName+" Refitted Track Parameter Residuals", 900, 600);
   resRefCanvas->Divide(3,2);

   resRefCanvas->cd(1);
   resRefPt->Fit("gaus");
   resRefCanvas->cd(2);
   resRefPhi->Fit("gaus");
   resRefCanvas->cd(3);
   resRefTheta->Fit("gaus");
   resRefCanvas->cd(4);
   resRefTimp->Fit("gaus");
   resRefCanvas->cd(5);
   resRefLimp->Fit("gaus");
   resRefCanvas->Update();

   pullRefCanvas = new TCanvas(theTreeName+"PullRef",theTreeName+" Refitted Track Parameter Pulls", 900, 600);
   pullRefCanvas->Divide(3,2);

   pullRefCanvas->cd(1);
   pullRefPt->Fit("gaus");
   pullRefCanvas->cd(2);
   pullRefPhi->Fit("gaus");
   pullRefCanvas->cd(3);
   pullRefTheta->Fit("gaus");
   pullRefCanvas->cd(4);
   pullRefTimp->Fit("gaus");
   pullRefCanvas->cd(5);
   pullRefLimp->Fit("gaus");
   pullRefCanvas->Update();

   distCanvas = new TCanvas(theTreeName+"TrkDistr",theTreeName+" Track Distributions", 900, 600);
   distCanvas->Divide(3,2);
   distCanvas->cd(1);
   pTSim->Draw();
   distCanvas->cd(2);
   etaSim->Draw();
   distCanvas->cd(3);
   pTRec->Draw();
   distCanvas->cd(4);
   etaRec->Draw();
   distCanvas->cd(5);
   pTRef->Draw();
   distCanvas->cd(6);
   etaRef->Draw();
   distCanvas->Update();

   bookedVertexC = true;
}


void SimpleVertexAnalysis::psTrackResult(TString name)
{
  if (!bookedVertexC) plotTrackResult();
  resRecCanvas->Print(name+".ps[");
  resRecCanvas->Draw();
  resRecCanvas->Print(name+".ps");
  pullRecCanvas->Draw();
  pullRecCanvas->Print(name+".ps");
  resRefCanvas->Draw();
  resRefCanvas->Print(name+".ps");
  pullRefCanvas->Draw();
  pullRefCanvas->Print(name+".ps");
  distCanvas->Draw();
  distCanvas->Print(name+".ps");
  distCanvas->Print(name+".ps]");
//   resRecCanvas->Draw();
//   resRecCanvas->Print(name+"_resRec.eps");
//   pullRecCanvas->Draw();
//   pullRecCanvas->Print(name+"_pullRec.eps");
//   resRefCanvas->Draw();
//   resRefCanvas->Print(name+"_resRef.eps");
//   pullRefCanvas->Draw();
//   pullRefCanvas->Print(name+"_pullRef.eps");
}
