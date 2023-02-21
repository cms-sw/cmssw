#define NtupleHelper_cc
#include "NtupleHelper.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TMath.h>

ClassImp(NtupleHelper);

void NtupleHelper::Book() {
  hvxy = new TH2F("vxy", "x vers. y vertex", 100, -0.01, 0.01, 100, -0.01, 0.01);
  hvxyz = new TH3F("vxyz", "x y z vertex", 100, -0.01, 0.01, 200, -40., 40., 100, -0.01, 0.01);
  hdphi = new TH2F("dphi", "d vers phi", 100, -0., 6.283, 100, -0.2, 0.2);
  hd = new TH2F("d", "d vers. z ", 100, -40., 40., 100, -0.01, 0.01);
  hvxz = new TH2F("vxz", "x vers. z vertex", 100, -40., 40., 100, -0.01, 0.01);
  hsigma = new TH1F("sigma", "sigma d ", 100, 0., 0.01);
  hsx = new TH1F("sx", "sigma vers. z", 50, -40., 40.);
  hpt = new TH1F("pt", "pt", 100, 0., 20.);
  hsx->Sumw2();
  hsd = new TH1F("sd", "d0 vers. z", 50, -40., 40.);
  hsd->Sumw2();
  hsw = new TH1F("sw", "sigma weight vers. z", 50, -40., 40.);
  hsw->Sumw2();
}
zData NtupleHelper::Loop(int maxEvents) {
  //   In a ROOT session, you can do:
  //      Root > .L NtupleHelper.C
  //      Root > NtupleHelper t
  //      Root > t.GetEntry(12); // Fill t data members with entry number 12
  //      Root > t.Show();       // Show values of entry 12
  //      Root > t.Show(16);     // Read and show values of entry 16
  //      Root > t.Loop();       // Loop on all entries
  //

  //     This is the loop skeleton where:
  //    jentry is the global entry number in the chain
  //    ientry is the entry number in the current Tree
  //  Note that the argument to GetEntry must be:
  //    jentry for TChain::GetEntry
  //    ientry for TTree::GetEntry and TBranch::GetEntry
  //
  //       To read only selected branches, Insert statements like:
  // METHOD1:
  //    fChain->SetBranchStatus("*",0);  // disable all branches
  //    fChain->SetBranchStatus("branchname",1);  // activate branchname
  // METHOD2: replace line
  //    fChain->GetEntry(jentry);       //read all branches
  //by  b_branchname->GetEntry(ientry); //read only this branch
  //   TH2F *hvxy = new TH2F("vxy","x vers. y vertex",100,-0.4,0.4,100,-0.4,0.4);

  std::cout << "  loop over entries" << std::endl;
  std::cout << "  maximum number of entries: " << maxEvents << std::endl;
  zData zvector;
  if (fChain == 0)
    return zvector;
  zvector.erase(zvector.begin(), zvector.end());
  Long64_t nentries = fChain->GetEntriesFast();

  std::cout << " total number of entries: " << fChain->GetEntries() << std::endl;

  int theevent = 0;
  for (Long64_t jentry = 0; jentry < nentries; jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0)
      break;
    fChain->GetEntry(jentry);
    //      if (sigmaD <0.05&&pt>4.0)
    //	{

    //if (z0>-20. && z0<20. && pt>4 ) {
    //if (pt>2 ) {
    //if (pt>5 && sigmaD<0.02) { //it was pt>5
    if (true) {
      if (maxEvents > 0 && maxEvents == theevent) {
        std::cout << " reached maximum number of tracks, continue" << std::endl;
        break;
      }
      hvxy->Fill(x, y);
      hvxyz->Fill(x, z0, y);
      hdphi->Fill(phi, d0);
      hvxz->Fill(z0, x);
      hsd->Fill(z0, fabs(d0));
      hd->Fill(z0, d0);
      hsx->Fill(z0, fabs(x));
      hpt->Fill(pt, 1.0);
      hsw->Fill(z0, 1.0);
      hsigma->Fill(sigmaD);

      // for reco ntuples:
      if (pt > 1.2 && TMath::Abs(eta) < 2.4 && TMath::Abs(d0) < 5 && TMath::Abs(z0) < 60 && nPixelLayerMeas >= 3 &&
          nTotLayerMeas >= 11 && normchi2 < 2 && quality && algo) {
        //TMath::Abs(d0)<0.06 ) {
        //(chi2/ndof)<5 && TMath::Prob(chi2, (int)ndof)>0.02 && TMath::Abs(eta)<2.2 ) {
        zvector.push_back(data(z0, sigmaz0, d0, sigmaD, phi, pt, 1.));
        theevent++;
      }
    }
    // if (Cut(ientry) < 0) continue;
    //	}
  }
  std::cout << zvector.size() << " selected tracks read in.\n";
  hsx->Divide(hsw);
  hsd->Divide(hsw);
  return zvector;
}
