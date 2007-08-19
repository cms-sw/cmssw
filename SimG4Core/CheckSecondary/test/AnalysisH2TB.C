void analyse(char element[6], char list[10], char ene[6]) {

  char fname[40];
  sprintf (fname, "%s%s%sGeV.root", element, list, ene);
  double rhol = rhoL(element);
  double atwt = atomicWt(element);
  std::vector<double> masses = massScan();
  cout << fname << " rhoL " << rhol << " atomic weight " << atwt << "\n";

  TH1F *hiKE0[15], *hiKE1[15], *hiKE2[15], *hiCT0[15], *hiCT1[15], *hiCT2[15];
  char name[60], title[160], ctype[20], ytitle[20];
  double xbin;
  for (unsigned int ii=0; ii<=(masses.size())+1; ii++) {
    if      (ii == 0) sprintf (ctype, "All Particles");
    else if (ii == 1) sprintf (ctype, "Photons");
    else if (ii == 2) sprintf (ctype, "Electrons/Positrons");
    else if (ii == 3) sprintf (ctype, "Neutral Pions");
    else if (ii == 4) sprintf (ctype, "Charged Pions");
    else if (ii == 5) sprintf (ctype, "Charged Kaons");
    else if (ii == 6) sprintf (ctype, "Neutral Kaons");
    else if (ii == 7) sprintf (ctype, "Protons/Antiportons");
    else if (ii == 8) sprintf (ctype, "Neutrons");
    else if (ii == 9) sprintf (ctype, "Heavy hadrons");
    else              sprintf (ctype, "Ions");

    sprintf (title, "%s in %s at %s GeV (%s)", ctype, element, ene, list);
    sprintf (name, "KE0%s%s%sGeV(%s)", element, list, ene, ctype);
    hiKE0[ii] = new TH1F (name, title, 1000, 0., 100.);
    hiKE0[ii]->GetXaxis()->SetTitle("Kinetic Energy (GeV)");
    xbin = hiKE0[ii]->GetBinWidth(1);
    sprintf (ytitle, "Events/%6.3f GeV", xbin);
    hiKE0[ii]->GetYaxis()->SetTitle(ytitle);
    //std::cout << "hiKE0[" << ii << "] = " << hiKE0[ii] << " " <<  name << " KE Energy  " << title << "\n";
    sprintf (name, "CT0%s%s%sGeV(%s)", element, list, ene, ctype);
    hiCT0[ii] = new TH1F (name, title, 100, -1.0, 1.0.);
    hiCT0[ii]->GetXaxis()->SetTitle("cos (#theta)");
    xbin = hiCT0[ii]->GetBinWidth(1);
    sprintf (ytitle, "Events/%6.3f", xbin);
    hiCT0[ii]->GetYaxis()->SetTitle(ytitle);
    //std::cout << "hiCT0[" << ii << "] = " << hiCT0[ii] << " " <<  name << " cos(T#eta) " << title << "\n";

    sprintf (title, "%s (Elastic) in %s at %s GeV (%s)", ctype, element, ene, list);
    sprintf (name, "KE1%s%s%sGeV(%s)", element, list, ene, ctype);
    hiKE1[ii] = new TH1F (name, title, 1000, 0., 100.);
    hiKE1[ii]->GetXaxis()->SetTitle("Kinetic Energy (GeV)");
    xbin = hiKE1[ii]->GetBinWidth(1);
    sprintf (ytitle, "Events/%6.3f GeV", xbin);
    hiKE1[ii]->GetYaxis()->SetTitle(ytitle);
    //std::cout << "hiKE1[" << ii << "] = " << hiKE1[ii] << " " <<  name << " KE Energy  " << title << "\n";
    sprintf (name, "CT1%s%s%sGeV(%s)", element, list, ene, ctype);
    hiCT1[ii] = new TH1F (name, title, 100, -1.0, 1.0.);
    hiCT1[ii]->GetXaxis()->SetTitle("cos (#theta)");
    xbin = hiCT1[ii]->GetBinWidth(1);
    sprintf (ytitle, "Events/%6.3f", xbin);
    hiCT1[ii]->GetYaxis()->SetTitle(ytitle);
    //std::cout << "hiCT1[" << ii << "] = " << hiCT1[ii] << " " <<  name << " cos(T#eta) " << title << "\n";

    sprintf (title, "%s (InElastic) in %s at %s GeV (%s)", ctype, element, ene, list);
    sprintf (name, "KE2%s%s%sGeV(%s)", element, list, ene, ctype);
    hiKE2[ii] = new TH1F (name, title, 1000, 0., 100.);
    hiKE2[ii]->GetXaxis()->SetTitle("Kinetic Energy (GeV)");
    xbin = hiKE2[ii]->GetBinWidth(1);
    sprintf (ytitle, "Events/%6.3f GeV", xbin);
    hiKE2[ii]->GetYaxis()->SetTitle(ytitle);
    //std::cout << "hiKE2[" << ii << "] = " << hiKE2[ii] << " " <<  name << " KE Energy  " << title << "\n";
    sprintf (name, "CT2%s%s%sGeV(%s)", element, list, ene, ctype);
    hiCT2[ii] = new TH1F (name, title, 100, -1.0, 1.0.);
    hiCT2[ii]->GetXaxis()->SetTitle("cos (#theta)");
    xbin = hiCT2[ii]->GetBinWidth(1);
    sprintf (ytitle, "Events/%6.3f", xbin);
    hiCT2[ii]->GetYaxis()->SetTitle(ytitle);
    //std::cout << "hiCT2[" << ii << "] = " << hiCT2[ii] << " " <<  name << " cos(T#eta) " << title << "\n";
  }

  TFile *file = new TFile(fname);
  TTree *tree = (TTree *) file->Get("T1");
  
  if (!tree) {
    std::cout << "Cannot find Tree T1 in file " << fname << "\n";
  } else {
    std::cout << "Tree T1 found with " << tree->GetEntries() << " entries\n";
    int nentry = tree->GetEntries();
    int ninter=0, elastic=0, inelastic=0;
    for (int i=0; i<nentry; i++) {
      if (i%1000 == 0) std::cout << "Start processing event " << i << "\n";
      std::vector<int>                     *nsec, *procids;
      std::vector<double>                  *px, *py, *pz, *mass;
      std::vector<std::string>             *procs;
      tree->SetBranchAddress("NumberSecondaries", &nsec);
      tree->SetBranchAddress("ProcessID",         &procids);
      //      tree->SetBranchAddress("ProcessNames",      &procs);
      tree->SetBranchAddress("SecondaryPx",       &px);
      tree->SetBranchAddress("SecondaryPy",       &py);
      tree->SetBranchAddress("SecondaryPz",       &pz);
      tree->SetBranchAddress("SecondaryMass",     &mass);
      tree->GetEntry(i);
      if ((*nsec).size() > 0) {
	ninter++;
	bool isItElastic = false;
	if ((*procids)[0] == 17) {elastic++; isItElastic = true;}
	else                     inelastic++;
	
	if (ninter <3) {
	  std::cout << "Interaction " << ninter << "/" << i+1 << " Type "
		    << (*procids)[0]  << " with " << (*nsec)[0] << " secondaries\n";
	  for (int k=0; k<(*nsec)[0]; k++)
	    std::cout << " Secondary " << k << " Px " << (*px)[k] << " Py " << (*py)[k] << " Pz " << (*pz)[k] << " Mass " << (*mass)[k] << "\n";
	}
	
	for (int k=0; k<(*nsec)[0]; k++) {
	  int type = 0;
	  for (unsigned int it=0; it<masses.size(); it++) 
	    if (abs((*mass)[k]) > masses[it]) type = it+1;
	  double m  = abs((*mass)[k]);
	  double pl = (*py)[k];
	  double pt = ((*px)[k])*((*px)[k])+((*pz)[k])*((*pz)[k]);
	  double pp = (pt+pl*pl);
	  double ke = (sqrt (pp + m*m) - m)/1000.;
	  pp        = sqrt (pp);
	  double cth= (pp == 0. ? -2. : (pl/pp));
	  //std::cout << "Entry " << i << " Secondary " << k << " Mass " << (*mass)[k] << " Type " << type << " Cth " << cth << " KE " << ke << "\n";
	  hiKE0[0]->Fill(ke);
	  hiCT0[0]->Fill(cth);
	  hiKE0[type+1]->Fill(ke);
	  hiCT0[type+1]->Fill(cth);
	  if (isItElastic) {
	    hiKE1[0]->Fill(ke);
	    hiCT1[0]->Fill(cth);
	    hiKE1[type+1]->Fill(ke);
	    hiCT1[type+1]->Fill(cth);
	  } else {
	    hiKE2[0]->Fill(ke);
	    hiCT2[0]->Fill(cth);
	    hiKE2[type+1]->Fill(ke);
	    hiCT2[type+1]->Fill(cth);
	  }
	}
      }
    }
    
    std::cout << ninter << " interactions seen in " << nentry << " trials\n";
    double sigma = atwt*10000.*log((double)(nentry)/(double)(nentry-ninter))/(rhol*6.023);
    double dsigma    = sigma/sqrt(double(ninter));
    double sigmaEl   = sigma*((double)(elastic))/((double)(ninter));
    double dsigmaEl  = sigmaEl/sqrt(double(elastic));
    double sigmaInel = sigma*((double)(inelastic))/((double)(ninter));
    double dsigmaInel= sigmaInel/sqrt(double(inelastic));
    std::cout << "Total     " << sigma << " +- " << dsigma 
	      << " mb (" << ninter << " events)\n"
	      << "Elastic   " << sigmaEl<< " +- " << dsigmaEl
	      << " mb (" << ninter << " events)\n"
	      << "Inelasric " << sigmaInel << " +- " << dsigmaInel
	      << " mb (" << ninter << " events)\n";
  }

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFrameBorderMode(0);
  gStyle->SetFrameBorderSize(1);  gStyle->SetFrameFillColor(0);
  gStyle->SetFrameFillStyle(0);   gStyle->SetFrameLineColor(1);
  gStyle->SetFrameLineStyle(1);   gStyle->SetFrameLineWidth(1);
  gStyle->SetOptLogy(1);          gStyle->SetTitleOffset(1.2,"Y");
  TCanvas *cc1[15], *cc2[15];
  for (unsigned int iia=0; iia<=(masses.size())+1; iia++) {
    if      (iia == 0) sprintf (ctype, "All Particles");
    else if (iia == 1) sprintf (ctype, "Photons");
    else if (iia == 2) sprintf (ctype, "Electrons/Positrons");
    else if (iia == 3) sprintf (ctype, "Neutral Pions");
    else if (iia == 4) sprintf (ctype, "Charged Pions");
    else if (iia == 5) sprintf (ctype, "Charged Kaons");
    else if (iia == 6) sprintf (ctype, "Neutral Kaons");
    else if (iia == 7) sprintf (ctype, "Protons/Antiportons");
    else if (iia == 8) sprintf (ctype, "Neutrons");
    else if (iia == 9) sprintf (ctype, "Heavy hadrons");
    else               sprintf (ctype, "Ions");

    sprintf (title, "%s in %s at %s GeV (%s)", ctype, element, ene, list);
    sprintf(name, "C-KE%i", iia);
    cc1[iia] = new TCanvas(name,title,800,600); cc1[iia]->Divide(2,2);
    cc1[iia]->cd(1); if (hiKE0[iia]->GetEntries() > 0) hiKE0[iia]->Draw(); 
    cc1[iia]->cd(3); if (hiKE1[iia]->GetEntries() > 0) hiKE1[iia]->Draw();
    cc1[iia]->cd(4); if (hiKE2[iia]->GetEntries() > 0) hiKE2[iia]->Draw(); 
    sprintf(name, "C-CT%i", iia);
    cc2[iia] = new TCanvas(name,title,800,600); cc2[iia]->Divide(2,2);
    cc2[iia]->cd(1); if (hiCT0[iia]->GetEntries() > 0) hiCT0[iia]->Draw(); 
    cc2[iia]->cd(3); if (hiCT1[iia]->GetEntries() > 0) hiCT1[iia]->Draw();
    cc2[iia]->cd(4); if (hiCT2[iia]->GetEntries() > 0) hiCT2[iia]->Draw(); 
  }
}

double rhoL(char element[6]) {
  
  double tmp=0;
  if      (element == "Brass") tmp = 8.50 * 0.40.;
  else if (element == "PbWO4") tmp = 8.28 * 0.40.;
  return tmp;
}

double atomicWt(char element[6]) {
  
  double tmp=0;
  if      (element == "Brass") tmp = 64.228;
  else if (element == "PbWO4") tmp = 455.036;
  return tmp;
}

std::vector<double> massScan() {

  std::vector<double> tmp;
  tmp.push_back(0.01);
  tmp.push_back(1.00);
  tmp.push_back(135.0);
  tmp.push_back(140.0);
  tmp.push_back(495.0);
  tmp.push_back(500.0);
  tmp.push_back(938.5);
  tmp.push_back(940.0);
  tmp.push_back(1850.0);
  std::cout << tmp.size() << " Mass regions for prtaicles: ";
  for (unsigned int i=0; i<tmp.size(); i++) {
    std::cout << tmp[i];
    if (i == tmp.size()-1) std::cout << " MeV\n";
    else                   std::cout << ", ";
  }
  return tmp;
}
