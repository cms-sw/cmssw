#include <vector>


void NewResolution() { 

  vector<float> caloJetBarrel;
  vector<float> caloJetEndcaps;
  vector<float> jptJetBarrel;
  vector<float> jptJetEndcaps;
  vector<float> pfJetBarrel;
  vector<float> pfJetEndcaps;
  vector<float> pfNewJetBarrel;
  vector<float> pfNewJetEndcaps;
  vector<float> jetEnergy;
  
  jptJetEndcaps.push_back((0.414+0.416)/2.);
  jptJetEndcaps.push_back((0.361+0.356)/2.);
  jptJetEndcaps.push_back((0.334+0.338)/2.);
  jptJetEndcaps.push_back((0.281+0.281)/2.);
  jptJetEndcaps.push_back((0.207+0.203)/2.);
  jptJetEndcaps.push_back((0.162+0.160)/2.);
  jptJetEndcaps.push_back((0.144+0.145)/2.);
  jptJetEndcaps.push_back((0.129+0.131)/2.);
  jptJetEndcaps.push_back((0.117+0.118)/2.);
  jptJetEndcaps.push_back((0.103+0.105)/2.);
  jptJetEndcaps.push_back((0.093+0.089)/2.);
  jptJetEndcaps.push_back((0.080+0.083)/2.);
  jptJetEndcaps.push_back((0.071+0.068)/2.);
  jptJetEndcaps.push_back((0.063+0.063)/2.);
  jptJetEndcaps.push_back((0.057+0.057)/2.);
  jptJetEndcaps.push_back((0.048+0.050)/2.);
  jptJetEndcaps.push_back((0.049+0.052)/2.);
  jptJetEndcaps.push_back((0.046+0.044)/2.);
  jptJetEndcaps.push_back((0.047+0.042)/2.);
  jptJetEndcaps.push_back((0.010-8.268)/2.);
  jptJetEndcaps.push_back(-1.);
  jptJetEndcaps.push_back(-1.);
  jptJetEndcaps.push_back(-1.);
 
  jptJetBarrel.push_back((0.305+0.307)/2.);
  jptJetBarrel.push_back((0.289+0.282)/2.);
  jptJetBarrel.push_back((0.268+0.271)/2.);
  jptJetBarrel.push_back((0.253+0.250)/2.);
  jptJetBarrel.push_back((0.207+0.209)/2.);
  jptJetBarrel.push_back((0.168+0.166)/2.);
  jptJetBarrel.push_back((0.152+0.150)/2.);
  jptJetBarrel.push_back((0.137+0.138)/2.);
  jptJetBarrel.push_back((0.121+0.121)/2.);
  jptJetBarrel.push_back((0.109+0.107)/2.);
  jptJetBarrel.push_back((0.098+0.096)/2.);
  jptJetBarrel.push_back((0.087+0.087)/2.);
  jptJetBarrel.push_back((0.078+0.078)/2.);
  jptJetBarrel.push_back((0.067+0.067)/2.);
  jptJetBarrel.push_back((0.060+0.061)/2.);
  jptJetBarrel.push_back((0.055+0.055)/2.);
  jptJetBarrel.push_back((0.049+0.051)/2.);
  jptJetBarrel.push_back((0.044+0.046)/2.);
  jptJetBarrel.push_back((0.042+0.040)/2.);
  jptJetBarrel.push_back((0.039+0.038)/2.);
  jptJetBarrel.push_back((0.035+0.036)/2.);
  jptJetBarrel.push_back((0.035+0.035)/2.);
  jptJetBarrel.push_back((0.034+0.035)/2.);

  caloJetEndcaps.push_back((0.527+0.536)/2.);
  caloJetEndcaps.push_back((0.446+0.455)/2.);
  caloJetEndcaps.push_back((0.398+0.403)/2.);
  caloJetEndcaps.push_back((0.318+0.320)/2.);
  caloJetEndcaps.push_back((0.245+0.240)/2.);
  caloJetEndcaps.push_back((0.188+0.188)/2.);
  caloJetEndcaps.push_back((0.155+0.156)/2.);
  caloJetEndcaps.push_back((0.131+0.135)/2.);
  caloJetEndcaps.push_back((0.121+0.122)/2.);
  caloJetEndcaps.push_back((0.112+0.106)/2.);
  caloJetEndcaps.push_back((0.097+0.098)/2.);
  caloJetEndcaps.push_back((0.085+0.087)/2.);
  caloJetEndcaps.push_back((0.077+0.074)/2.);
  caloJetEndcaps.push_back((0.071+0.069)/2.);
  caloJetEndcaps.push_back((0.061+0.063)/2.);
  caloJetEndcaps.push_back((0.054+0.054)/2.);
  caloJetEndcaps.push_back((0.050+0.050)/2.);
  caloJetEndcaps.push_back((0.048+0.047)/2.);
  caloJetEndcaps.push_back((0.064+0.049)/2.);
  caloJetEndcaps.push_back((0.014+0.000)/2.);
  caloJetEndcaps.push_back(-1.);
  caloJetEndcaps.push_back(-1.);
  caloJetEndcaps.push_back(-1.);
			
  caloJetBarrel.push_back((0.612+0.614)/2.);
  caloJetBarrel.push_back((0.559+0.549)/2.);
  caloJetBarrel.push_back((0.484+0.481)/2.);
  caloJetBarrel.push_back((0.403+0.405)/2.);
  caloJetBarrel.push_back((0.314+0.312)/2.);
  caloJetBarrel.push_back((0.250+0.241)/2.);
  caloJetBarrel.push_back((0.199+0.204)/2.);
  caloJetBarrel.push_back((0.177+0.178)/2.);
  caloJetBarrel.push_back((0.157+0.159)/2.);
  caloJetBarrel.push_back((0.145+0.136)/2.);
  caloJetBarrel.push_back((0.127+0.127)/2.);
  caloJetBarrel.push_back((0.116+0.119)/2.);
  caloJetBarrel.push_back((0.102+0.101)/2.);
  caloJetBarrel.push_back((0.086+0.086)/2.);
  caloJetBarrel.push_back((0.074+0.075)/2.);
  caloJetBarrel.push_back((0.064+0.065)/2.);
  caloJetBarrel.push_back((0.057+0.060)/2.);
  caloJetBarrel.push_back((0.052+0.054)/2.);
  caloJetBarrel.push_back((0.047+0.048)/2.);
  caloJetBarrel.push_back((0.044+0.043)/2.);
  caloJetBarrel.push_back((0.039+0.040)/2.);
  caloJetBarrel.push_back((0.038+0.038)/2.);
  caloJetBarrel.push_back((0.036+0.036)/2.);


  pfJetEndcaps.push_back((0.310+0.311)/2.);
  pfJetEndcaps.push_back((0.276+0.279)/2.);
  pfJetEndcaps.push_back((0.254+0.255)/2.);
  pfJetEndcaps.push_back((0.216+0.218)/2.);
  pfJetEndcaps.push_back((0.178+0.179)/2.);
  pfJetEndcaps.push_back((0.153+0.152)/2.);
  pfJetEndcaps.push_back((0.135+0.136)/2.);
  pfJetEndcaps.push_back((0.119+0.121)/2.);
  pfJetEndcaps.push_back((0.108+0.113)/2.);
  pfJetEndcaps.push_back((0.100+0.100)/2.);
  pfJetEndcaps.push_back((0.092+0.088)/2.);
  pfJetEndcaps.push_back((0.079+0.083)/2.);
  pfJetEndcaps.push_back((0.072+0.072)/2.);
  pfJetEndcaps.push_back((0.061+0.063)/2.);
  pfJetEndcaps.push_back((0.055+0.056)/2.);
  pfJetEndcaps.push_back((0.050+0.049)/2.);
  pfJetEndcaps.push_back((0.046+0.049)/2.);
  pfJetEndcaps.push_back((0.047+0.046)/2.);
  pfJetEndcaps.push_back((0.052+0.046)/2.);
  pfJetEndcaps.push_back((0.017+0.046)/2.);
  pfJetEndcaps.push_back(-1.);
  pfJetEndcaps.push_back(-1.);
  pfJetEndcaps.push_back(-1.);
			       
		       
  pfJetBarrel.push_back((0.236+0.236)/2.);
  pfJetBarrel.push_back((0.217+0.213)/2.);			       
  pfJetBarrel.push_back((0.199+0.199)/2.);			       
  pfJetBarrel.push_back((0.180+0.179)/2.);			       
  pfJetBarrel.push_back((0.155+0.154)/2.);			       
  pfJetBarrel.push_back((0.135+0.138)/2.);			       
  pfJetBarrel.push_back((0.124+0.131)/2.);
  pfJetBarrel.push_back((0.117+0.121)/2.);
  pfJetBarrel.push_back((0.115+0.113)/2.);
  pfJetBarrel.push_back((0.109+0.110)/2.);
  pfJetBarrel.push_back((0.102+0.104)/2.);
  pfJetBarrel.push_back((0.096+0.097)/2.);
  pfJetBarrel.push_back((0.088+0.088)/2.);
  pfJetBarrel.push_back((0.077+0.077)/2.);
  pfJetBarrel.push_back((0.067+0.064)/2.);
  pfJetBarrel.push_back((0.056+0.057)/2.);
  pfJetBarrel.push_back((0.053+0.053)/2.);
  pfJetBarrel.push_back((0.047+0.048)/2.);
  pfJetBarrel.push_back((0.043+0.042)/2.);
  pfJetBarrel.push_back((0.039+0.040)/2.);
  pfJetBarrel.push_back((0.037+0.038)/2.);
  pfJetBarrel.push_back((0.036+0.036)/2.);
  pfJetBarrel.push_back((0.035+0.035)/2.);

  pfNewJetEndcaps.push_back(-1.);
  pfNewJetEndcaps.push_back(-1.);
  pfNewJetEndcaps.push_back(-1.);
  pfNewJetEndcaps.push_back(-1.);
  pfNewJetEndcaps.push_back(0.160);
  pfNewJetEndcaps.push_back(0.137);
  pfNewJetEndcaps.push_back(0.124);
  pfNewJetEndcaps.push_back(0.114);
  pfNewJetEndcaps.push_back(0.103);
  pfNewJetEndcaps.push_back(0.090);
  pfNewJetEndcaps.push_back(0.078);
  pfNewJetEndcaps.push_back(0.071);
  pfNewJetEndcaps.push_back(0.067);
  pfNewJetEndcaps.push_back(0.060);
  pfNewJetEndcaps.push_back(0.045);
  pfNewJetEndcaps.push_back(0.038);
  pfNewJetEndcaps.push_back(0.041);
  pfNewJetEndcaps.push_back(-1.);
  pfNewJetEndcaps.push_back(-1.);
  pfNewJetEndcaps.push_back(-1.);
  pfNewJetEndcaps.push_back(-1.);
  pfNewJetEndcaps.push_back(-1.);
  pfNewJetEndcaps.push_back(-1.);
			       
		       
  pfNewJetBarrel.push_back(-1.);
  pfNewJetBarrel.push_back(-1.);			       
  pfNewJetBarrel.push_back(-1.);			       
  pfNewJetBarrel.push_back(-1.);			       
  pfNewJetBarrel.push_back(0.146);			       
  pfNewJetBarrel.push_back(0.129);			       
  pfNewJetBarrel.push_back(0.120);
  pfNewJetBarrel.push_back(0.111);
  pfNewJetBarrel.push_back(0.105);
  pfNewJetBarrel.push_back(0.102);
  pfNewJetBarrel.push_back(0.095);
  pfNewJetBarrel.push_back(0.088);
  pfNewJetBarrel.push_back(0.076);
  pfNewJetBarrel.push_back(0.071);
  pfNewJetBarrel.push_back(0.058);
  pfNewJetBarrel.push_back(0.051);
  pfNewJetBarrel.push_back(0.050);
  pfNewJetBarrel.push_back(-1.);
  pfNewJetBarrel.push_back(-1.);
  pfNewJetBarrel.push_back(-1.);
  pfNewJetBarrel.push_back(-1.);
  pfNewJetBarrel.push_back(-1.);
  pfNewJetBarrel.push_back(-1.);

  jetEnergy.push_back(7.5);
  jetEnergy.push_back(11);
  jetEnergy.push_back(13.5);
  jetEnergy.push_back(17.5);
  jetEnergy.push_back(23.5);
  jetEnergy.push_back(31.);
  jetEnergy.push_back(40.);
  jetEnergy.push_back(51.);
  jetEnergy.push_back(64.5);
  jetEnergy.push_back(81.);
  jetEnergy.push_back(105.);
  jetEnergy.push_back(135.);
  jetEnergy.push_back(175.);
  jetEnergy.push_back(250.);
  jetEnergy.push_back(350.);
  jetEnergy.push_back(475.);
  jetEnergy.push_back(650.);
  jetEnergy.push_back(875.);
  jetEnergy.push_back(1250.);
  jetEnergy.push_back(1750.);
  jetEnergy.push_back(2250.);
  jetEnergy.push_back(2750.);
  jetEnergy.push_back(3250.);

  TGraph* grCaloBarrel = new TGraph ( 23, &jetEnergy[0], &caloJetBarrel[0] );
  TGraph* grCaloEndcap = new TGraph ( 23, &jetEnergy[0], &caloJetEndcaps[0] );
  TGraph* grJptBarrel = new TGraph ( 23, &jetEnergy[0], &jptJetBarrel[0] );
  TGraph* grJptEndcap = new TGraph ( 23, &jetEnergy[0], &jptJetEndcaps[0] );
  TGraph* grPfBarrel = new TGraph ( 23, &jetEnergy[0], &pfJetBarrel[0] );
  TGraph* grPfEndcap = new TGraph ( 23, &jetEnergy[0], &pfJetEndcaps[0] );
  TGraph* grPfNewBarrel = new TGraph ( 23, &jetEnergy[0], &pfNewJetBarrel[0] );
  TGraph* grPfNewEndcap = new TGraph ( 23, &jetEnergy[0], &pfNewJetEndcaps[0] );

  TCanvas *cBarrelOld = new TCanvas("cBarrelOld","",1000, 700);
  TCanvas *cBarrel = new TCanvas("cBarrel","",1000, 700);

  cBarrelOld->cd();

  TH2F *h = new TH2F("Barrel","", 
		     100, 20., 3500., 100, 0.0, 0.4 );
  h->SetTitle( "Jet Energy Resolution in the Barrel" );
  h->SetXTitle("p_{T} [GeV/c]" );
  h->SetYTitle("#s(p_{T} / <p_{T}>");
  gPad->SetLogx();
  gPad->SetGridx();
  gPad->SetGridy();
  h->SetStats(0);
  h->Draw();

  grCaloBarrel->SetMarkerColor(1);						
  grCaloBarrel->SetMarkerStyle(8);
  grCaloBarrel->Draw("P");

  grJptBarrel->SetMarkerColor(4);						
  grJptBarrel->SetMarkerStyle(23);
  grJptBarrel->Draw("P");

  grPfBarrel->SetMarkerColor(1);						
  grPfBarrel->SetMarkerStyle(22);
  grPfBarrel->Draw("P");

  //grPfNewBarrel->SetMarkerColor(2);						
  //grPfNewBarrel->SetMarkerStyle(22);
  //grPfNewBarrel->Draw("P");

  TLegend *leg=new TLegend(0.70,0.65,0.85,0.85);
  leg->AddEntry(grCaloBarrel, "Calo", "p");
  leg->AddEntry(grJptBarrel, "JPT", "p");
  leg->AddEntry(grPfNewBarrel, "PF 11/08", "p");
  leg->SetTextSize(0.03);
  leg->Draw();

  gPad->SaveAs("Barrel_223.png");

  cBarrel->cd();

  TH2F *h = new TH2F("Barrel","", 
		     100, 20., 3500., 100, 0.0, 0.4 );
  h->SetTitle( "Jet Energy Resolution in the Barrel" );
  h->SetXTitle("p_{T} [GeV/c]" );
  h->SetYTitle("#sigma (p_{T}) / <p_{T}>");
  gPad->SetLogx();
  gPad->SetGridx();
  gPad->SetGridy();
  h->SetStats(0);
  h->Draw();

  grCaloBarrel->SetMarkerColor(1);						
  grCaloBarrel->SetMarkerStyle(8);
  grCaloBarrel->SetMarkerSize(1.5);
  grCaloBarrel->Draw("P");

  grJptBarrel->SetMarkerColor(4);						
  grJptBarrel->SetMarkerStyle(23);
  grJptBarrel->SetMarkerSize(1.5);
  grJptBarrel->Draw("P");

  //grPfBarrel->SetMarkerColor(1);						
  //grPfBarrel->SetMarkerStyle(29);
  //grPfBarrel->Draw("P");

  grPfNewBarrel->SetMarkerColor(2);						
  grPfNewBarrel->SetMarkerStyle(22);
  grPfNewBarrel->SetMarkerSize(1.5);
  grPfNewBarrel->Draw("P");

  TLegend *leg=new TLegend(0.50,0.65,0.85,0.85);
  leg->AddEntry(grCaloBarrel, "Calorimeter Jets", "p");
  leg->AddEntry(grJptBarrel, "Jet-plus-Track Jets", "p");
  leg->AddEntry(grPfNewBarrel, "Particle-Flow Jets, 02/09", "p");
  leg->SetTextSize(0.03);
  leg->Draw();

  gPad->SaveAs("Barrel_310.png");

  TCanvas *cEndcapOld = new TCanvas("cEndcapOld","",1000, 700);
  TCanvas *cEndcap = new TCanvas("cEndcap","",1000, 700);

  cEndcapOld->cd();

  TH2F *h = new TH2F("Endcap","", 
		     100, 20., 3500., 100, 0.0, 0.4 );
  h->SetTitle( "Endcap" );
  h->SetXTitle("Ref pT [GeV/c]" );
  h->SetYTitle("Relative resolution");
  gPad->SetLogx();
  h->SetStats(0);
  h->Draw();

  grCaloEndcap->SetMarkerColor(1);						
  grCaloEndcap->SetMarkerStyle(8);
  grCaloEndcap->Draw("P");

  grJptEndcap->SetMarkerColor(4);						
  grJptEndcap->SetMarkerStyle(23);
  grJptEndcap->Draw("P");

  grPfEndcap->SetMarkerColor(2);						
  grPfEndcap->SetMarkerStyle(29);
  grPfEndcap->Draw("P");

  //grPfNewEndcap->SetMarkerColor(2);						
  //grPfNewEndcap->SetMarkerStyle(22);
  //grPfNewEndcap->Draw("P");

  TLegend *leg=new TLegend(0.70,0.65,0.85,0.85);
  leg->AddEntry(grCaloEndcap, "Calo", "p");
  leg->AddEntry(grJptEndcap, "JPT", "p");
  leg->AddEntry(grPfNewEndcap, "PF 11/08", "p");
  leg->SetTextSize(0.03);
  leg->Draw();

  gPad->SaveAs("Endcap_223.png");

  cEndcap->cd();

  TH2F *h = new TH2F("Endcap","", 
		     100, 20., 3500., 100, 0.0, 0.4 );
  h->SetTitle( "Endcap" );
  h->SetXTitle("Ref pT [GeV/c]" );
  h->SetYTitle("Relative resolution");
  gPad->SetLogx();
  h->SetStats(0);
  h->Draw();

  grCaloEndcap->SetMarkerColor(1);						
  grCaloEndcap->SetMarkerStyle(30);
  grCaloEndcap->SetMarkerSize(1.5);
  grCaloEndcap->Draw("P");

  grJptEndcap->SetMarkerColor(1);						
  grJptEndcap->SetMarkerStyle(3);
  grJptEndcap->SetMarkerSize(1.5);
  grJptEndcap->Draw("P");

  //grPfEndcap->SetMarkerColor(1);						
  //grPfEndcap->SetMarkerStyle(29);
  //grPfEndcap->Draw("P");

  grPfNewEndcap->SetMarkerColor(2);						
  grPfNewEndcap->SetMarkerStyle(22);
  grPfNewEndcap->SetMarkerSize(1.5);						
  grPfNewEndcap->Draw("P");

  TLegend *leg=new TLegend(0.70,0.65,0.85,0.85);
  leg->AddEntry(grCaloEndcap, "Calo", "p");
  leg->AddEntry(grJptEndcap, "JPT", "p");
  leg->AddEntry(grPfNewEndcap, "PF 02/09", "p");
  leg->SetTextSize(0.03);
  leg->Draw();

  gPad->SaveAs("Endcap_310.png");

  gPad->SaveAs("Tau.png");
}
