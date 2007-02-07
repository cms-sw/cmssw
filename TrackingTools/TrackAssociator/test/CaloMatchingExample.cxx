void makePlots(TString filename)
{
   TFile* file = new TFile(filename);
   TTree* tree = (TTree*)file->Get("calomatch");
   TCanvas* c = new TCanvas("matches","matches",600,900);
   TH1F* h1 = new TH1F("he1","Matched ECAL crystal energy",100,0,5);
   h1->SetFillColor(7);
   TH1F* h1_1 = new TH1F("he11","Matched ECAL crystal energy, pt<10",100,0,5);
   TH1F* h1_2 = new TH1F("he12","Matched ECAL crystal energy, 10<pt<30",100,0,5);
   TH1F* h1_3 = new TH1F("he13","Matched ECAL crystal energy, 30<pt",100,0,5);
   TH1F* h2 = new TH1F("he2","3x3 ECAL crystal energy",100,0,5);
   h2->SetFillColor(7);
   TH1F* h3 = new TH1F("he3","5x5 ECAL crystal energy",100,0,5);
   h3->SetFillColor(7);
   
   TH1F* h4 = new TH1F("he4","Matched HCAL crystal energy",100,0,25);
   h4->SetFillColor(3);
   TH1F* h5 = new TH1F("he5","3x3 HCAL crystal energy",100,0,25);
   h5->SetFillColor(3);
   TH1F* h6 = new TH1F("he6","5x5 HCAL crystal energy",100,0,25);
   h6->SetFillColor(3);
   tree->Draw("ecalCrossedEnergy>>he1");
   tree->Draw("ecal3x3Energy>>he2");
   tree->Draw("ecal5x5Energy>>he3");
   tree->Draw("hcalCrossedEnergy>>he4");
   tree->Draw("hcal3x3Energy>>he5");
   tree->Draw("hcal5x5Energy>>he6");
   tree->Draw("ecalCrossedEnergy>>he11","trackPt<10");
   tree->Draw("ecalCrossedEnergy>>he12","trackPt<30&&trackPt>10");
   tree->Draw("ecalCrossedEnergy>>he13","trackPt>30");
   c->Clear();
   c->Divide(2,3);
   c->cd(1); h1->Draw();
   c->cd(2); h4->Draw();
   c->cd(3); h2->Draw();
   c->cd(4); h5->Draw();
   c->cd(5); h3->Draw();
   c->cd(6); h6->Draw();
   
    
   TCanvas* c2 = new TCanvas("details","details",600,900);

   gROOT->LoadMacro("resolution_fit.cxx");
   
   c2->Clear();
   c2->Divide(2,3);
   
   c2->cd(1);
   TH1F* hh1 = new TH1F("hh1","ECAL dEta, energy deposition vs track propagatation",100,-0.1,0.1);
   tree->Draw("trkPosAtEcal[][0]-ecalMaxPos[][0]>>hh1");
   // resolution_fit(hh1,"ECAL dEta, energy deposition vs track propagatation");
   
   c2->cd(2);
   TH1F* hh2 = new TH1F("hh2","ECAL dPhi, energy deposition vs track propagatation",100,-0.1,0.1);
   tree->Draw("trkPosAtEcal[][1]-ecalMaxPos[][1]>>hh2");

   c2->cd(3);
   TH1F* hh3 = new TH1F("hh3","HCAL dEta, energy deposition vs track propagatation",100,-0.1,0.1);
   tree->Draw("trkPosAtHcal[][0]-hcalMaxPos[][0]>>hh3");
   
   c2->cd(4);
   TH1F* hh4 = new TH1F("hh4","HCAL dPhi, energy deposition vs track propagatation",100,-0.1,0.1);
   tree->Draw("trkPosAtHcal[][1]-hcalMaxPos[][1]>>hh4");

   c2->cd(5); 
   TH1F* hh5 = new TH1F("hh5","ECAL crystal Eta for failed matches",100,-3.5,3.5);
   tree->Draw("trkPosAtEcal[][0]>>hh5","fabs(ecalCrossedEnergy)<0.001");

   c2->cd(6); 
   TH1F* hh6 = new TH1F("hh6","ECAL crystal Phi for failed matches",100,-3.5,3.5);
   tree->Draw("trkPosAtEcal[][1]>>hh6","fabs(ecalCrossedEnergy)<0.001");
}
