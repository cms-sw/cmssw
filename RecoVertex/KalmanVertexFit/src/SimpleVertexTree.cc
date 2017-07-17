#include "RecoVertex/KalmanVertexFit/interface/SimpleVertexTree.h"

#include "TROOT.h"
#include "TTree.h"
#include "TH1F.h"
#include <iostream>

using namespace std;

SimpleVertexTree::SimpleVertexTree(const char * filterName,
	const MagneticField * magField) :
     theFitterName(filterName)
{

  vertexTree = new TTree(filterName, "Vertex fit results");
//   trackTest 
//     = SimpleConfigurable<bool> (false, "SimpleVertexTree:trackTest").value();
//   if (trackTest) {
//     maxTrack = SimpleConfigurable<int> (100, "SimpleVertexTree:maximumTracksToStore").value();
//   } else 
  maxTrack = 0;
  result = new VertexFitterResult(maxTrack, magField);

  vertexTree->Branch("vertex",(void *)result->vertexPresent(),"vertex/I");
  vertexTree->Branch("simPos",(void *)result->simVertexPos(),"X/F:Y/F:Z/F");
  vertexTree->Branch("recPos",(void *)result->recVertexPos(),"X/F:Y/F:Z/F");
  vertexTree->Branch("recErr",(void *)result->recVertexErr(),"X/F:Y/F:Z/F");
  vertexTree->Branch("nbrTrk",(void *)result->trackInformation(),"Sim/I:Rec/I:Shared/I");
  vertexTree->Branch("chiTot",(void *)result->chi2Information (),"chiTot/F");
  vertexTree->Branch("ndf",(void *)(result->chi2Information ()+1),"ndf/F");
  vertexTree->Branch("chiProb",(void *)(result->chi2Information ()+2),"chiProb/F");
  vertexTree->Branch("time",(void *)result->time(),"time/F");


  parameterNames[0] = new TString("ptinv");
  parameterNames[1] = new TString("theta");
  parameterNames[2] = new TString("phi");
  parameterNames[3] = new TString("timp");
  parameterNames[4] = new TString("limp");

  vertexTree->Branch("simTracks",(void *)result->numberSimTracks(),"simTracks/I");
  vertexTree->Branch("simTrack_recIndex",(void *)result->simTrack_recIndex(),"simTrack_recIndex[simTracks]/I");
  defineTrackBranch("sim", "Par", &VertexFitterResult::simParameters, "simTracks");

  vertexTree->Branch("recTracks",(void *)result->numberRecTracks(),"recTracks/I");
  vertexTree->Branch("recTrack_simIndex",(void *)result->recTrack_simIndex(),"recTrack_simIndex[recTracks]/I");
  vertexTree->Branch("recTrack_weight",(void *)result->recTrackWeight(),"recTrack_weight[recTracks]/F");
  defineTrackBranch("rec", "Par", &VertexFitterResult::recParameters, "recTracks");
  defineTrackBranch("ref", "Par", &VertexFitterResult::refParameters, "recTracks");
  defineTrackBranch("rec", "Err", &VertexFitterResult::recErrors, "recTracks");
  defineTrackBranch("ref", "Err", &VertexFitterResult::refErrors, "recTracks");

  numberOfVertices = 0;
}


void SimpleVertexTree::defineTrackBranch(const TString& prefix, const TString& type,
			const float* (VertexFitterResult::*pfunc)(const int) const,
			const TString& index)
{
    TString branchName, branchVariables;
    for ( int i=0; i<5; i++ ) {
      branchName = prefix + type + '_' + *parameterNames[i];
      branchVariables = branchName + '[' + index + "]/F";
      vertexTree->Branch(branchName,(void *)(result->*pfunc)(i),branchVariables);
    }
}


SimpleVertexTree::~SimpleVertexTree()
{
  std::cout << std::endl<< "End of SimpleVertexTree for "<< theFitterName << std::endl;
  std::cout << std::endl<< "Number of vertices fit: "<< numberOfVertices<<std::endl;

  //
  // save current root directory
  //
  TDirectory* rootDir = gDirectory;
  //
  // close files
  //
  vertexTree->GetDirectory()->cd();
  vertexTree->Write();
  if (numberOfVertices>0) {
    TH1F *resX = new TH1F(theFitterName + "_ResX","Residual x coordinate: "+theFitterName, 100, -0.03, 0.03);
    TH1F *resY = new TH1F(theFitterName + "_ResY","Residual y coordinate: "+theFitterName, 100, -0.03, 0.03);
    TH1F *resZ = new TH1F(theFitterName + "_ResZ","Residual z coordinate: "+theFitterName, 100, -0.03, 0.03);
    TH1F *pullX = new TH1F(theFitterName + "_PullX","Pull x coordinate: "+theFitterName, 100, -10., 10.);
    TH1F *pullY = new TH1F(theFitterName + "_PullY","Pull y coordinate: "+theFitterName, 100, -10., 10.);
    TH1F *pullZ = new TH1F(theFitterName + "_PullZ","Pull z coordinate: "+theFitterName, 100, -10., 10.);
    TH1F *chiNorm = new TH1F(theFitterName + "_ChiNorm","Normalized chi-square: " +theFitterName, 100, 0., 10.);
    TH1F *chiProb = new TH1F(theFitterName + "_ChiProb","Chi-square probability: "+theFitterName, 100, 0., 1.);
    vertexTree->Project(theFitterName + "_ResX", "(simPos.X-recPos.X)");
    vertexTree->Project(theFitterName + "_ResY", "(simPos.Y-recPos.Y)");
    vertexTree->Project(theFitterName + "_ResZ", "(simPos.Z-recPos.Z)");
    vertexTree->Project(theFitterName + "_PullX", "(simPos.X-recPos.X)/recErr.X");
    vertexTree->Project(theFitterName + "_PullY", "(simPos.Y-recPos.Y)/recErr.Y");
    vertexTree->Project(theFitterName + "_PullZ", "(simPos.Z-recPos.Z)/recErr.Z");
    vertexTree->Project(theFitterName + "_ChiNorm", "chiTot/ndf");
    vertexTree->Project(theFitterName + "_ChiProb", "chiProb");
    std::cout << "Mean of Residual distribution X: "<< resX->GetMean()<<std::endl;
    std::cout << "Mean of Residual distribution Y: "<< resY->GetMean()<<std::endl;
    std::cout << "Mean of Residual distribution Z: "<< resZ->GetMean()<<std::endl;
    std::cout << "RMS of Residual distribution X:  "<< resX->GetRMS()<<std::endl;
    std::cout << "RMS of Residual distribution Y:  "<< resY->GetRMS()<<std::endl;
    std::cout << "RMS of Residual distribution Z:  "<< resZ->GetRMS()<<std::endl;
    std::cout << "Mean of Pull distribution X: "<< pullX->GetMean()<<std::endl;
    std::cout << "Mean of Pull distribution Y: "<< pullY->GetMean()<<std::endl;
    std::cout << "Mean of Pull distribution Z: "<< pullZ->GetMean()<<std::endl;
    std::cout << "RMS of Pull distribution X:  "<< pullX->GetRMS()<<std::endl;
    std::cout << "RMS of Pull distribution Y:  "<< pullY->GetRMS()<<std::endl;
    std::cout << "RMS of Pull distribution Z:  "<< pullZ->GetRMS()<<std::endl;
    std::cout << "Average chi-square probability: "<< chiProb->GetMean()<<std::endl;
    std::cout << "Average normalized chi-square : "<< chiNorm->GetMean()<<std::endl;
    resX->Write();
    resY->Write();
    resZ->Write();
    pullX->Write();
    pullY->Write();
    pullZ->Write();
    chiNorm->Write();
    chiProb->Write();
  }
  delete vertexTree;
  //
  // restore directory
  //
  rootDir->cd();
  std::cout << std::endl;

}

void SimpleVertexTree::fill(const TransientVertex & recv, const TrackingVertex * simv, 
  	    reco::RecoToSimCollection *recSimColl, const float &time) 
{
  result->fill(recv, simv, recSimColl, time);
  fill();
}

void SimpleVertexTree::fill(const TransientVertex & recv, const float &time) 
{
  result->fill(recv, 0, 0, time);
  fill();
}

void SimpleVertexTree::fill(const TrackingVertex * simv) 
{
  result->fill(TransientVertex(), simv, 0);
  fill();
}

// void SimpleVertexTree::fill(const RecVertex & recVertex, 
// 			const std::vector < RecTrack > & recTrackV,
// 			const TkSimVertex * simv, const float &time)
// {
//   result->fill(recVertex, recTrackV, simv, time);
//   fill();
// }
// 
// void SimpleVertexTree::fill(const std::vector < RecTrack > & recTrackV,
// 			const TkSimVertex * simv, const float &time)
// {
//   result->fill(RecVertex(), recTrackV, simv, time);
//   fill();
// }

void SimpleVertexTree::fill() 
{
  ++numberOfVertices;
  static std::atomic<int> nFill{0};
  TDirectory* rootDir = gDirectory;
  //
  // fill entry
  //
  vertexTree->GetDirectory()->cd();
  vertexTree->Fill();
  if ( (++nFill)%1000==0 )  vertexTree->AutoSave();
  //
  // restore directory
  //
  rootDir->cd();
  result->reset();


}
