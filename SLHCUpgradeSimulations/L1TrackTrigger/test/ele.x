{

gStyle->SetOptStat(0);

int nbins = 50;
float x1 = 4.5 ;
float x2 = 54.5 ;

TH1F* hEG = new TH1F("hEG",";ET threshold (GeV); Rate (kHz)",nbins,x1,x2);

TH1F* hPho = new TH1F("hPho",";ET threshold (GeV); Rate (kHz)",nbins,x1,x2);
TH1F* hEle = new TH1F("hEle",";ET threshold (GeV); Rate (kHz)",nbins,x1,x2);
TH1F* hEleIso = new TH1F("hEleIso",";ET threshold (GeV); Rate (kHz)",nbins,x1,x2);
TH1F* hEleLoose = new TH1F("hEleLoose",";ET threshold (GeV); Rate (kHz)",nbins,x1,x2);	// PTtrack = 3 GeV
TH1F* hEleLooseV2 = new TH1F("hEleLooseV2",";ET threshold (GeV); Rate (kHz)",nbins,x1,x2);	// PTTrack cut = 2 GeV


Events -> Draw("Max$(l1extraL1EmParticles_SLHCL1ExtraParticlesNewClustering_EGamma_ALL.obj.pt_)>>hEG");
Events -> Draw("Max$(l1extraL1TkEmParticles_L1TkPhotons_EG_ALL.obj.pt_)>>hPho");

Events -> Draw("Max$(l1extraL1TkElectronParticles_L1TkElectrons_EG_ALL.obj.pt_)>>hEle");
Events -> Draw("Max$(l1extraL1TkElectronParticles_L1TkIsoElectrons_EG_ALL.obj.pt_)>>hEleIso");
Events -> Draw("Max$(l1extraL1TkElectronParticles_L1TkElectronsLoose_EG_ALL.obj.pt_)>>hEleLoose");
Events -> Draw("Max$(l1extraL1TkElectronParticles_L1TkElectronsLooseV2_EG_ALL.obj.pt_)>>hEleLooseV2");


for (int i=0; i<= nbins+1; i++) {
  hEG -> SetBinContent(i, hEG -> Integral(i, nbins+1) );
  hPho -> SetBinContent(i, hPho -> Integral(i, nbins+1) );
  hEle -> SetBinContent(i, hEle -> Integral(i, nbins+1) );
  hEleIso -> SetBinContent(i, hEleIso -> Integral(i, nbins+1) );
  hEleLoose -> SetBinContent(i, hEleLoose -> Integral(i, nbins+1) );
  hEleLooseV2 -> SetBinContent(i, hEleLooseV2 -> Integral(i, nbins+1) );

  hEG -> SetBinError(i, sqrt( hEG -> GetBinContent(i) ) );
  hPho -> SetBinError(i, sqrt( hPho -> GetBinContent(i) ) );
  hEle -> SetBinError(i, sqrt( hEle -> GetBinContent(i) ) );
  hEleIso -> SetBinError(i, sqrt( hEleIso -> GetBinContent(i) ) );
  hEleLoose -> SetBinError(i, sqrt( hEleLoose -> GetBinContent(i) ) );
  hEleLooseV2 -> SetBinError(i, sqrt( hEleLooseV2 -> GetBinContent(i) ) );

}

float nevts = Events->GetEntries();

hEG -> Scale(30000./nevts);
hPho -> Scale(30000./nevts);
hEle -> Scale(30000./nevts);
hEleIso -> Scale(30000./nevts);
hEleLoose -> Scale(30000./nevts);
hEleLooseV2 -> Scale(30000./nevts);

hEG -> SetMaximum(50000.);
hEG -> SetMinimum(10.);
hEG -> SetLineColor(2);
hEG -> Draw("hist");

//hPho -> SetLineColor(2);
//hPho -> Draw("same,hist");

hEle -> SetLineColor(1);
hEle -> SetMarkerColor(1);
hEle -> Draw("same,pe");

hEleIso -> SetLineColor(6);
hEleIso -> SetMarkerColor(6);
hEleIso -> Draw("same,hist");

hEleLoose -> SetLineColor(4);
hEleLoose -> SetMarkerColor(4);
hEleLoose -> Draw("same,pe");

hEleLooseV2 -> SetLineColor(7);
hEleLooseV2 -> Draw("same,hist");

gPad -> SetLogy(1);
gPad -> SetGridx(1);
gPad -> SetGridy(1);

 TLegend *leg = new TLegend(0.5,0.75,0.93,0.92,NULL,"brNDC");
        leg->SetBorderSize(0);
        leg->SetLineColor(1);
        leg->SetLineStyle(1);
        leg->SetLineWidth(1);
        leg->SetFillColor(0);
        leg->SetFillStyle(0);
        leg->SetTextFont(42);
 leg -> AddEntry(hEG,"L1EG","l");
 leg -> AddEntry(hEle,"L1TkElectrons","pe");
 leg -> AddEntry(hEleIso,"L1TkElectrons, TkIso","l");
 leg -> AddEntry(hEleLoose,"L1TkElectronsLoose, PT > 3 GeV","pe");
 leg -> AddEntry(hEleLooseV2,"L1TkElectronsLoose, PT > 2 GeV","l");

 leg -> Draw();


}


