void plotCompare(char type[2]="HB", char hist[5]="h0", int bin1=2, int bin2=2,
		 int logy=0, char header[20]="HCal Barrel", int save=0) {

  char file1[50], file2[50], title[100], fileps[50];
  sprintf (file1, "%s_new.root", type);
  sprintf (file2, "../data/%s_ref.root", type);
  sprintf (fileps, "%s%s.eps", type, hist);
  if (type == "HF") {
    switch (hist) {
    case "h0":   sprintf (title, "D(R) between Jet and its hits"); break;
    case "h1":   sprintf (title, "Time of hits in the Jets"); break;
    case "h2":   sprintf (title, "Energy of hits in the Jets"); break;
    case "h3":   sprintf (title, "Energy in ECal for the Jets"); break;
    case "h4":   sprintf (title, "Energy in HCal for the Jets"); break;
    case "h5":   sprintf (title, "Total energy   for the Jets"); break;
    case "h6":   sprintf (title, "Jet Energy"); break;
    case "h7":   sprintf (title, "Jet Eta"); break;
    case "h8":   sprintf (title, "Jet Phi"); break;
    case "h9":   sprintf (title, "Eta of hits"); break;          
    case "h10":  sprintf (title, "Phi of hits"); break;
    case "h11":  sprintf (title, "Energy of hits"); break;
    case "h12":  sprintf (title, "Time of hits"); break;
    case "h13":  sprintf (title, "Energy in Ecal"); break;
    case "h14":  sprintf (title, "Energy in Hcal"); break;
    case "h15":  sprintf (title, "Time of hits"); break;
    case "h16":  sprintf (title, "Time (energy weighted)"); break;
    case "h17":  sprintf (title, "Number of hits"); break;
    case "h18":  sprintf (title, "Energy in long fibre"); break;
    case "h19":  sprintf (title, "Energy in short fibre"); break;
    default:     sprintf (title, "Unknown"); break;
    }
  } else {
    switch (hist) {
    case "h0":   sprintf (title, "D(R) between Jet and its hits"); break;
    case "h1":   sprintf (title, "Time of hits in the Jets"); break;
    case "h2":   sprintf (title, "Energy of hits in the Jets"); break;
    case "h3":   sprintf (title, "Energy in ECal for the Jets"); break;
    case "h4":   sprintf (title, "Energy in HCal for the Jets"); break;
    case "h5":   sprintf (title, "Energy in HO   for the Jets"); break;
    case "h6":   sprintf (title, "Total energy   for the Jets"); break;
    case "h7":   sprintf (title, "D(eta) Jet"); break;
    case "h8":   sprintf (title, "D(phi) Jet"); break;
    case "h9":   sprintf (title, "D(r)   Jet"); break;
    case "h10":  sprintf (title, "Jet Energy"); break;
    case "h11":  sprintf (title, "Jet Eta"); break;
    case "h12":  sprintf (title, "Jet Phi"); break;
    case "h13":  sprintf (title, "Dijet Mass"); break;
    case "h14":  sprintf (title, "Energy in ECal NxN (DR Cone)"); break;
    case "h15":  sprintf (title, "Energy in HCal NxN (DR Cone)"); break;
    case "h16":  sprintf (title, "Energy in HO   NxN (DR Cone)"); break;
    case "h17":  sprintf (title, "Total energy   NxN (DR Cone)"); break;
    case "h18":  sprintf (title, "Energy in ECal NxN (eta/phi window)"); break;
    case "h19":  sprintf (title, "Energy in HCal NxN (eta/phi window)"); break;
    case "h20":  sprintf (title, "Energy in HO   NxN (eta/phi window)"); break;
    case "h21":  sprintf (title, "Total energy   NxN (eta/phi window)"); break;
    case "h22":  sprintf (title, "Layer number for hits"); break;
    case "h23":  sprintf (title, "Eta of hits"); break;          
    case "h24":  sprintf (title, "Phi of hits"); break;
    case "h25":  sprintf (title, "Energy of hits"); break;
    case "h26":  sprintf (title, "Time of hits"); break;
    case "h27":  sprintf (title, "ID of hits"); break;
    case "h28":  sprintf (title, "Jitter of hits"); break;
    case "h29":  sprintf (title, "Energy of hits in NxN"); break;
    case "h30":  sprintf (title, "Time   of hits in NxN"); break;
    case "h31":  sprintf (title, "Energy deposit in each layer"); break;
    case "h32":  sprintf (title, "Energy deposit in each depth"); break;
    case "h33":  sprintf (title, "Energy in HO"); break;
    case "h34":  sprintf (title, "Energy in HB/HE"); break;
    case "h35":  sprintf (title, "Energy in long HF fibre"); break;
    case "h36":  sprintf (title, "Energy in short HF fibre"); break;
    case "h37":  sprintf (title, "Energy in Ecal"); break;
    case "h38":  sprintf (title, "Energy in Hcal"); break;
    case "h39":  sprintf (title, "NxN trans fraction"); break; 
    case "h40":  sprintf (title, "Hit time 50ns"); break;   
    case "h41":  sprintf (title, "Hit time (energy weighted)"); break; 
    case "h42":  sprintf (title, "Number of hits in ECal"); break;
    case "h43":  sprintf (title, "Number of hits in HCal"); break;
    case "h44":  sprintf (title, "Number of hits"); break;
    case "h45":  sprintf (title, "Longitudinal Profile (E weighted)"); break;
    case "hl0":  sprintf (title, "Energy deposit in layer 0"); break; 
    case "hl1":  sprintf (title, "Energy deposit in layer 1"); break; 
    case "hl2":  sprintf (title, "Energy deposit in layer 2"); break; 
    case "hl3":  sprintf (title, "Energy deposit in layer 3"); break; 
    case "hl4":  sprintf (title, "Energy deposit in layer 4"); break; 
    case "hl5":  sprintf (title, "Energy deposit in layer 5"); break; 
    case "hl6":  sprintf (title, "Energy deposit in layer 6"); break; 
    case "hl7":  sprintf (title, "Energy deposit in layer 7"); break; 
    case "hl8":  sprintf (title, "Energy deposit in layer 8"); break; 
    case "hl9":  sprintf (title, "Energy deposit in layer 9"); break; 
    case "hl10": sprintf (title, "Energy deposit in layer 10"); break; 
    case "hl11": sprintf (title, "Energy deposit in layer 11"); break; 
    case "hl12": sprintf (title, "Energy deposit in layer 12"); break; 
    case "hl13": sprintf (title, "Energy deposit in layer 13"); break; 
    case "hl14": sprintf (title, "Energy deposit in layer 14"); break; 
    case "hl15": sprintf (title, "Energy deposit in layer 15"); break; 
    case "hl16": sprintf (title, "Energy deposit in layer 16"); break; 
    case "hl17": sprintf (title, "Energy deposit in layer 17"); break; 
    case "hl18": sprintf (title, "Energy deposit in layer 18"); break; 
    case "hl19": sprintf (title, "Energy deposit in layer 19"); break; 
    default:     sprintf (title, "Unknown"); break;
    }
  }

  cout << "New: " << file1 << " and Ref: " << file2 << " Title: " << title 
       << "\neps file: " << fileps << endl;

  gROOT->LoadMacro("mbtkStyle.C");
  setTDRStyle();

  TFile *File1 = new TFile(file1);
  TH1F* new_hist = (TH1F*) File1->Get(hist) ;

  TFile *File2 = new TFile(file2);
  TH1F* ref_hist = (TH1F*) File2->Get(hist) ;

  TCanvas *myc = new TCanvas("myc","",800,600);

  new_hist->SetLineStyle(1);
  new_hist->SetLineWidth(3);
  new_hist->SetLineColor(2);
  new_hist->Rebin(bin1);
  new_hist->GetXaxis()->SetTitle(title);
  //  new_hist->GetYaxis()->SetTitle("Frequency");
  if (logy > 0) gPad->SetLogy(1);
  new_hist->Draw("HIST");

  ref_hist->SetLineStyle(2);
  ref_hist->SetLineWidth(3);
  ref_hist->SetLineColor(4);
  ref_hist->Rebin(bin2);
  ref_hist->Draw("HIST same");

  leg1 = new TLegend(0.55,0.15,0.85,0.25);
  leg1->AddEntry(new_hist,"Current Release","F");
  leg1->AddEntry(ref_hist,"Reference Version","F");
  leg1->SetHeader(header);
  leg1->Draw();

  if (save > 0) myc->SaveAs(fileps);

  return;
}
