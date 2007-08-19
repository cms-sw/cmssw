void analyse(char element[2], char list[10], char ene[6], int scan=1) {

  static double massP = 938.272;
  static double pi  = 3.1415926;
  static double deg = pi/180.;
  static double dcth= 0.05;
  char fname[40];
  sprintf (fname, "%s%s%sGeV.root", element, list, ene);

  double rhol = rhoL(element);
  double atwt = atomicWt(element);
  std::vector<double> angles = angleScan(scan);
  cout << fname << " rhoL " << rhol << " atomic weight " << atwt << "\n";

  TH1F *hiK0 = new TH1F ("hiK0", "All Protons",                  800,0.,8.);
  TH1F *hiK1 = new TH1F ("hiK1", "Elastic Scattered Protons",    800,0.,8.);
  TH1F *hiK2 = new TH1F ("hiK2", "Inelastic Scatteered Protons ",800,0.,8.);
  TH1F *hiC0 = new TH1F ("hiC0", "All Protons",                  100,-1.,1.);
  TH1F *hiC1 = new TH1F ("hiC1", "Elastic Scattered Protons",    100,-1.,1.);
  TH1F *hiC2 = new TH1F ("hiC2", "Inelastic Scattered Protons",  100,-1.,1.);
  std::vector<double> cthmin, cthmax;
  TH1F *hiKE1[30], *hiKE2[30];
  char name[60], title[160];
  for (unsigned int ii=0; ii<angles.size(); ii++) {
    double cth = cos(angles[ii]);
    cthmin.push_back(cth-0.5*dcth);
    cthmax.push_back(cth+0.5*dcth);
    sprintf (name, "KE1%s%s%sGeV%5.1f", element, list, ene, angles[ii]/deg);
    sprintf (title, "p+%s at %s GeV (%s) (#theta = %8.2f)", element, ene, list, angles[ii]/deg);
    hiKE1[ii] = new TH1F (name, title, 800, 0., 8.);
    //std::cout << "hiKE1[" << ii << "] = " << hiKE1[ii] << " " <<  name << "   " << title << "\n";
    sprintf (name, "KE2%s%s%sGeV%5.1f", element, list, ene, angles[ii]/deg);
    sprintf (title, "p+%s at %s GeV (%s) (#theta = %8.2f)", element, ene, list, angles[ii]/deg);
    hiKE2[ii] = new TH1F (name, title, 800, 0., 8.);
    //std::cout << "hiKE2[" << ii << "] = " << hiKE2[ii] << " " <<  name << "   " << title << "\n";
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
      if (i%10000 == 0) std:cout << "Started with event # " << i << "\n";
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
	  if (abs((*mass)[k]-massP) < 0.01) { // This is a proton
	    double pl = (*py)[k];
	    double pt = ((*px)[k])*((*px)[k])+((*pz)[k])*((*pz)[k]);
	    double pp = (pt+pl*pl);
	    double ke = (sqrt (pp + massP*massP) - massP)/1000.;
	    pp        = sqrt (pp);
	    double cth= (pp == 0. ? -2. : (pl/pp));
	    double wt = (pp == 0. ?  0. : (1000./pp));
	    // std::cout << "Entry " << i << " Secondary " << k << " Cth " << cth << " KE " << ke << " WT " << wt << "\n";
	    hiK0->Fill(ke);
	    hiC0->Fill(cth);
	    if (isItElastic) {
	      hiK1->Fill(ke);
	      hiC1->Fill(cth);
	    } else {
	      hiK2->Fill(ke);
	      hiC2->Fill(cth);
	      for (unsigned int ik=0; ik<angles.size(); ik++) {
		if (cth > cthmin[ik] && cth <= cthmax[ik]) {
		  // std::cout << " Loop " << ik << " Limit " << cthmin[ik] << " " << cthmax[ik] << " " << hiKE1[ik] << " " << hiKE2[ik] << "\n";
		  hiKE1[ik]->Fill(ke);
		  hiKE2[ik]->Fill(ke,wt);
		}
	      }
	    }
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

  TCanvas *c1 = new TCanvas("c1","K.E.",800,600); c1->Divide(2,2);
  hiK0->GetXaxis()->SetTitle("Kinetic Energy of proton (GeV)");
  hiK1->GetXaxis()->SetTitle("Kinetic Energy of proton (GeV)");
  hiK2->GetXaxis()->SetTitle("Kinetic Energy of proton (GeV)");
  c1->cd(1); hiK1->Draw(); c1->cd(2); hiK2->Draw(); c1->cd(3); hiK0->Draw();
  TCanvas *c2 = new TCanvas("c2","cos#theta",800,600); c2->Divide(2,2);
  hiC0->GetXaxis()->SetTitle("cos (#theta) of scattered protons");
  hiC1->GetXaxis()->SetTitle("cos (#theta) of scattered protons");
  hiC2->GetXaxis()->SetTitle("cos (#theta) of scattered protons");
  c2->cd(1); hiC1->Draw(); c2->cd(2); hiC2->Draw(); c2->cd(3); hiC0->Draw();
  TCanvas *cc[30];
  TH1F    *hiKE0[30];
  for (unsigned int iia=0; iia<angles.size(); iia++) {
    double xbin = hiKE2[iia]->GetBinWidth(1);
    sprintf (title, "Events/%6.3f GeV", xbin);
    hiKE1[iia]->GetXaxis()->SetTitle("Kinetic Energy of proton (GeV)");
    hiKE1[iia]->GetYaxis()->SetTitle(title);
    double xbin  = hiKE2[iia]->GetBinWidth(1);
    double scale = sigmaInel/(((double)(inelastic))*xbin*2.*pi*dcth);
    std::cout << "Bin " << iia << " Angle " << angles[iia]/deg << " Bin " << xbin << " Scale " << scale << " " << title << "\n";
    sprintf (title, "Events (scaled by #frac{1}{p})/%6.3f GeV", xbin);
    hiKE2[iia]->GetXaxis()->SetTitle("Kinetic Energy of proton (GeV)");
    hiKE2[iia]->GetYaxis()->SetTitle(title);
    hiKE0[iia] = (TH1F*)hiKE2[iia]->Clone();
    hiKE0[iia]->Scale(scale);
    hiKE0[iia]->GetYaxis()->SetTitle("E#frac{d^{3}#sigma}{dp^{3}} (mb/GeV^{2})");

    sprintf(name, "Canvas%i", iia);
    sprintf (title, "p+%s at %s GeV (%s) (#theta = %8.2f)", element, ene, list, angles[iia]/deg);
    cc[iia] = new TCanvas(name,title,800,600); cc[iia]->Divide(2,2);
    cc[iia]->cd(1); hiKE1[iia]->Draw(); cc[iia]->cd(2); hiKE0[iia]->Draw();
    cc[iia]->cd(3); hiKE2[iia]->Draw(); 
  }

  char ofile[40];
  sprintf (ofile, "%s%s%sGeV_%i.root", element, list, ene, scan);
  TFile f(ofile, "recreate");
  hiK0->Write(); hiK1->Write(); hiK2->Write();
  hiC0->Write(); hiC1->Write(); hiC2->Write();
  for (unsigned int iter=0; iter<angles.size(); iter++) {
    hiKE1[iter]->Write(); hiKE0[iter]->Write(); hiKE2[iter]->Write();
  }
  f.Close();
  std::cout << "o/p saved in file " << ofile << "\n";
  file->Close();
}

double rhoL(char element[2]) {

  double tmp=0;
  if      (element == "H")   tmp = 0.0708 * 800.;
  else if (element == "Be")  tmp = 1.848 * 80.;
  else if (element == "C")   tmp = 2.265 * 80.;
  else if (element == "Al")  tmp = 2.700 * 80.;
  else if (element == "Ti")  tmp = 4.530 * 40.;
  else if (element == "Fe")  tmp = 7.870 * 30.;
  else if (element == "Cu")  tmp = 8.960 * 30.;
  else if (element == "Nb")  tmp = 8.550 * 30.;
  else if (element == "Cd")  tmp = 8.630 * 30.;
  else if (element == "Sn")  tmp = 7.310 * 35.;
  else if (element == "Ta")  tmp = 16.65 * 20.;
  else if (element == "Pb")  tmp = 11.35 * 30.;
  else if (element == "U")   tmp = 18.95 * 20.;
  return tmp;
}

double atomicWt(char element[2]) {

  double tmp=0;
  if      (element == "H")   tmp = 1.00794;
  else if (element == "Be")  tmp = 9.0122;
  else if (element == "C")   tmp = 12.011;
  else if (element == "Al")  tmp = 26.98;
  else if (element == "Ti")  tmp = 47.88;
  else if (element == "Fe")  tmp = 55.85;
  else if (element == "Cu")  tmp = 63.546;
  else if (element == "Nb")  tmp = 92.906;
  else if (element == "Cd")  tmp = 112.41;
  else if (element == "Sn")  tmp = 118.69;
  else if (element == "Ta")  tmp = 180.9479;
  else if (element == "Pb")  tmp = 207.19;
  else if (element == "U")   tmp = 238.03;
  return tmp;
}

std::vector<double> angleScan(int scan) {

  static double deg = 3.1415926/180.;
  std::vector<double> tmp;
  if (scan <= 1) {
    tmp.push_back(59.1*deg);
    tmp.push_back(89.0*deg);
    tmp.push_back(119.0*deg);
    tmp.push_back(159.6*deg);
  } else {
    tmp.push_back(10.1*deg);
    tmp.push_back(15.0*deg);
    tmp.push_back(19.8*deg);
    tmp.push_back(24.8*deg);
    tmp.push_back(29.5*deg);
    tmp.push_back(34.6*deg);
    tmp.push_back(39.6*deg);
    tmp.push_back(49.3*deg);
    tmp.push_back(54.2*deg);
    tmp.push_back(59.1*deg);
    tmp.push_back(64.1*deg);
    tmp.push_back(69.1*deg);
    tmp.push_back(74.1*deg);
    tmp.push_back(79.1*deg);
    tmp.push_back(84.1*deg);
    tmp.push_back(89.0*deg);
    tmp.push_back(98.9*deg);
    tmp.push_back(108.9*deg);
    tmp.push_back(119.0*deg);
    tmp.push_back(129.1*deg);
    tmp.push_back(139.1*deg);
    tmp.push_back(149.3*deg);
    tmp.push_back(159.6*deg);
    tmp.push_back(161.4*deg);
    tmp.push_back(165.5*deg);
    tmp.push_back(169.5*deg);
    tmp.push_back(173.5*deg);
    tmp.push_back(177.0*deg);
  }
  std::cout << "Scan " << tmp.size() << " angular regions:\n";
  for (unsigned int i=0; i<tmp.size(); i++) {
    std::cout << tmp[i]/deg;
    if (i == tmp.size()-1) std::cout << " degrees\n";
    else                   std::cout << ", ";
  }
  return tmp;
}
