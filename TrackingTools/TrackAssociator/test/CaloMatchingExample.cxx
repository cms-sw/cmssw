double scale( double energy, double eta = 0 )
{
   if (fabs(eta)<1.3){ 
      // barrel
      return energy*113.2;
   } else {
      // endcap
      if (fabs(eta) <3)	{
	 return 167.2*energy;
      }
   }
   // for the rest return uncorrected energy
   return energy;
}

void makePlots(TString filename, bool fit = false)
{
   TFile* file = new TFile(filename);
   TTree* tree = (TTree*)file->Get("calomatch");
   if (fit) gROOT->LoadMacro("CruijffPdf.cc++");
   if (fit) gROOT->LoadMacro("resolution_fit.cxx++");
   gROOT->SetStyle("Plain");
   gStyle->SetPalette(1);

   TCanvas* c = new TCanvas("matches","matches",600,900);
   // c->Print("calo_match.pdf[","pdf");
   // TPostScript *ps = new TPostScript("calo_match.ps",111);
   // ps->NewPage();
   
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
   // tree->Draw("ecalCrossedEnergy>>he11","trackPt<10");
   // tree->Draw("ecalCrossedEnergy>>he12","trackPt<30&&trackPt>10");
   // tree->Draw("ecalCrossedEnergy>>he13","trackPt>30");
   c->Clear();
   c->Divide(2,3);
   c->cd(1); h1->Draw();
   c->cd(2); h4->Draw();
   c->cd(3); h2->Draw();
   c->cd(4); h5->Draw();
   c->cd(5); h3->Draw();
   c->cd(6); h6->Draw();
   c->Update();
   c->Print("calo_match.pdf(","pdf");
   //ps->NewPage();
   std::cout << "NextPage" << std::endl;
   ////////////////////////////////////////////////////////////////////////////////////
    
   // TCanvas* c2 = new TCanvas("details","details",600,900);
   c->Clear();
   c->Divide(2,3);
   
   c->cd(1);
   TH1F* hh1 = new TH1F("hh1","ECAL dEta, energy deposition vs track propagatation",100,-0.1,0.1);
   tree->Draw("trkPosAtEcal[][0]-ecalMaxPos[][0]>>hh1");
   // resolution_fit(hh1,"ECAL dEta, energy deposition vs track propagatation");
   
   c->cd(2);
   TH1F* hh2 = new TH1F("hh2","ECAL dPhi, energy deposition vs track propagatation",100,-0.1,0.1);
   tree->Draw("trkPosAtEcal[][1]-ecalMaxPos[][1]>>hh2");

   c->cd(3);
   TH1F* hh3 = new TH1F("hh3","HCAL dEta, energy deposition vs track propagatation",100,-0.1,0.1);
   tree->Draw("trkPosAtHcal[][0]-hcalMaxPos[][0]>>hh3");
   
   c->cd(4);
   TH1F* hh4 = new TH1F("hh4","HCAL dPhi, energy deposition vs track propagatation",100,-0.1,0.1);
   tree->Draw("trkPosAtHcal[][1]-hcalMaxPos[][1]>>hh4");

   c->cd(5); 
   TH1F* hh5 = new TH1F("hh5","ECAL crystal Eta for failed matches",100,-3.5,3.5);
   tree->Draw("trkPosAtEcal[][0]>>hh5","fabs(ecalCrossedEnergy)<0.001");

   c->cd(6); 
   TH1F* hh6 = new TH1F("hh6","ECAL crystal Phi for failed matches",100,-3.5,3.5);
   tree->Draw("trkPosAtEcal[][1]>>hh6","fabs(ecalCrossedEnergy)<0.001");
   
   c->Update();
   c->Print("calo_match.pdf","pdf");
   // ps->NewPage();
   std::cout << "NextPage" << std::endl;
   ////////////////////////////////////////////////////////////////////////////////////
   
   // c = new TCanvas("ecal","ecal",600,900);
   c->Clear();
   c->Divide(2,3);
   c->cd(1);
   calomatch->Draw("ecal5x5Energy/ecalTrueEnergy:trkPosAtEcal[0][0]","ecal5x5Energy>0&&ecalTrueEnergy>0&&ecal5x5Energy/ecalTrueEnergy>0&&ecal5x5Energy/ecalTrueEnergy<5&&nTracks==1","colz");
   c->cd(2);
   calomatch->Draw("ecal5x5Energy/ecalTrueEnergy:trkPosAtEcal[0][1]","ecal5x5Energy>0&&ecalTrueEnergy>0&&ecal5x5Energy/ecalTrueEnergy>0&&ecal5x5Energy/ecalTrueEnergy<5&&nTracks==1","colz");
   c->cd(3);
   calomatch->Draw("ecal3x3Energy/ecalTrueEnergy:trkPosAtEcal[0][0]","ecal3x3Energy>0&&ecalTrueEnergy>0&&ecal3x3Energy/ecalTrueEnergy>0&&ecal3x3Energy/ecalTrueEnergy<5&&nTracks==1","colz");
   c->cd(4);
   calomatch->Draw("ecal3x3Energy/ecalTrueEnergy:trkPosAtEcal[0][1]","ecal3x3Energy>0&&ecalTrueEnergy>0&&ecal3x3Energy/ecalTrueEnergy>0&&ecal3x3Energy/ecalTrueEnergy<5&&nTracks==1","colz");
   c->cd(5);
   calomatch->Draw("ecalCrossedEnergy/ecalTrueEnergy:trkPosAtEcal[0][0]","ecalCrossedEnergy>0&&ecalTrueEnergy>0&&ecalCrossedEnergy/ecalTrueEnergy>0&&ecalCrossedEnergy/ecalTrueEnergy<5&&nTracks==1","colz");
   c->cd(6);
   calomatch->Draw("ecalCrossedEnergy/ecalTrueEnergy:trkPosAtEcal[0][1]","ecalCrossedEnergy>0&&ecalTrueEnergy>0&&ecalCrossedEnergy/ecalTrueEnergy>0&&ecalCrossedEnergy/ecalTrueEnergy<5&&nTracks==1","colz");
   
   c->Update();
   c->Print("calo_match.pdf","pdf");
   // ps->NewPage();
   std::cout << "NextPage" << std::endl;
   ////////////////////////////////////////////////////////////////////////////////////

   // c = new TCanvas("hcal","hcal",600,900);
   c->Clear();
   c->Divide(2,3);
   c->cd(1);
   calomatch->Draw("hcal5x5Energy/hcalTrueEnergyCorrected:trkPosAtHcal[0][0]",
		   "hcalTrueEnergy>0&&hcal5x5Energy/hcalTrueEnergy>0&&hcal5x5Energy/hcalTrueEnergyCorrected<5&&nTracks==1","colz");
   std::cout << "NextCD" << std::endl;
   c->cd(2);
   calomatch->Draw("hcal5x5Energy/hcalTrueEnergyCorrected:trkPosAtHcal[0][1]",
		   "hcalTrueEnergy>0&&hcal5x5Energy/hcalTrueEnergy>0&&hcal5x5Energy/hcalTrueEnergyCorrected<5&&nTracks==1","colz");
   std::cout << "NextCD" << std::endl;
   c->cd(3);
   calomatch->Draw("hcal3x3Energy/hcalTrueEnergyCorrected:trkPosAtHcal[0][0]",
		   "hcal3x3Energy>0&&hcalTrueEnergy>0&&hcal3x3Energy/hcalTrueEnergy>0&&hcal3x3Energy/hcalTrueEnergyCorrected<5&&nTracks==1","colz");
   std::cout << "NextCD" << std::endl;
   c->cd(4);
   calomatch->Draw("hcal3x3Energy/hcalTrueEnergyCorrected:trkPosAtHcal[0][1]",
		   "hcal3x3Energy>0&&hcalTrueEnergy>0&&hcal3x3Energy/hcalTrueEnergy>0&&hcal3x3Energy/hcalTrueEnergyCorrected<5&&nTracks==1","colz");
   std::cout << "NextCD" << std::endl;
   c->cd(5);
   calomatch->Draw("hcalCrossedEnergy/hcalTrueEnergyCorrected:trkPosAtHcal[0][0]",
		   "hcalCrossedEnergy>0&&hcalTrueEnergy>0&&hcalCrossedEnergy/hcalTrueEnergy>0&&hcalCrossedEnergy/hcalTrueEnergyCorrected<5&&nTracks==1","colz");
   std::cout << "NextCD" << std::endl;
   c->cd(6);
   calomatch->Draw("hcalCrossedEnergy/hcalTrueEnergyCorrected:trkPosAtHcal[0][1]",
		   "hcalCrossedEnergy>0&&hcalTrueEnergy>0&&hcalCrossedEnergy/hcalTrueEnergy>0&&hcalCrossedEnergy/hcalTrueEnergyCorrected<5&&nTracks==1","colz");

   c->Update();
   // ps->NewPage();
   c->Print("calo_match.pdf","pdf");
   std::cout << "NextPage" << std::endl;
   ////////////////////////////////////////////////////////////////////////////////////

   // c = new TCanvas("energy_scale","energy scale",600,900);
   c->Clear();
   c->Divide(2,3);
   c->cd(1);
   calomatch->Draw("ecal5x5Energy/ecalTrueEnergy>>h11(50,0.2,3)","fabs(trkPosAtEcal[0][0])<1.48");
   if (fit) cruijff_fit(h11,"ecal5x5Energy/ecalTrueEnergy Barrel",0.5,1.5,0.1,10,0.1,10);
   c->cd(3);
   calomatch->Draw("ecal3x3Energy/ecalTrueEnergy>>h21(50,0.2,3)","fabs(trkPosAtEcal[0][0])<1.48");
   if (fit) cruijff_fit(h21,"ecal3x3Energy/ecalTrueEnergy Barrel",0.5,1.5,0.1,10,0.1,10);
   c->cd(5);
   calomatch->Draw("ecalCrossedEnergy/ecalTrueEnergy>>h31(50,0.2,3)","fabs(trkPosAtEcal[0][0])<1.48");
   if (fit) cruijff_fit(h31,"ecalCrossedEnergy/ecalTrueEnergy Barrel",0.5,1.5,0.1,10,0.1,10);

   c->cd(2);
   calomatch->Draw("hcal5x5Energy/hcalTrueEnergyCorrected>>h12(50,0.2,3.2)");
   if (fit) cruijff_fit(h12,"HCAL 5x5Energy/expectedEnergy",0.5,1.5,0.1,10,0.1,10);
   c->cd(4);
   calomatch->Draw("hcal3x3Energy/hcalTrueEnergyCorrected>>h22(50,0.2,3.2)");
   if (fit) cruijff_fit(h22,"HCAL 3x3Energy/expectedEnergy",0.5,1.5,0.1,10,0.1,10);
   c->cd(6);
   calomatch->Draw("hcalCrossedEnergy/hcalTrueEnergyCorrected>>h32(50,0.2,3.2)");
   if (fit) cruijff_fit(h32,"HCAL crossedEnergy/expectedEnergy",0.5,1.5,0.1,10,0.1,10);
   c->Print("calo_match.pdf)","pdf");
   // ps->Close();
   std::cout << "Done" << std::endl;

}
