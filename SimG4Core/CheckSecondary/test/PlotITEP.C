void plotData(char element[2], char ene[6], char angle[6], int save=0) {

  char file1[50], file2[50];
  sprintf (file1, "itep/data/%s%sGeV%sdeg.dat",  element, ene, angle);
  sprintf (file2, "itep/data/%s%sGeV%sdeg.dat2", element, ene, angle);
  cout << file1 << "  " << file2 << "\n";
  ifstream infile1, infile2;
  infile1.open(file1);
  infile2.open(file2);
  
  Int_t   q1, i=0;
  Float_t m1, r1, x1[30], y1[30], stater1[30], syser1[30];
  infile1 >> m1 >> r1 >> q1;
  for (i=0; i<q1; i++) infile1 >> x1[i] >> y1[i] >> stater1[i] >> syser1[i];

  Int_t   q2, n=0;
  Float_t m2, r2, x2[30], y2[30], stater2[30], syser2[30];
  infile2 >> m2 >> r2 >> q2;
  for (i=0; i<q2; i++) infile2 >> x2[i] >> y2[i] >> stater2[i] >> syser2[i];

  Float_t x[30], chi[30], dif[30], edif[30], chi2=0.0, diff=0., dify=0. ;
  for (i=0; i<q1; i++) {
    for (int j=0; j<q2; j++) {
      double dx = ((x1[i]-x2[j]) >= 0 ? (x1[i]-x2[j]): -(x1[i]-x2[j]));
      if (dx < 0.0001) {
	double d1 = stater1[i]/y1[i];
	double d2 = stater2[j]/y2[j];
	double dd = sqrt(stater1[i]*stater1[i] + stater2[j]*stater2[j]);
	x[n]    = x1[i];
	dif[n]  = 200.*(y1[i]-y2[j])/(y1[i]+y2[j]);
	double da = (dif[n] > 0 ? dif[n]: -dif[n]);
	edif[n] = da*sqrt(d1*d1+d2*d2);
	chi[n]  = pow(((y1[i]-y2[j])/dd),2);
	diff   += da;
	chi2   += chi[n];
	dify   += dif[n];
	n++;
	break;
      }
    }
  }

  for (i=0; i<n; i++)
    std::cout << "Data " << i << " E " << x[i] << " Difference " << dif[i] << " +- " << edif[i] << " Chi2 " << chi[i] << "\n";

  dify /= n;
  std::cout << "Chi-Square = " << chi2 << "/" << n << " Mean Difference = " << diff/n << "\n";

  setStyle();
  char name[30], title[60];
  sprintf (name, "%s%sGeV%sdeg", element, ene, angle);
  sprintf (title, "p+%s at %s GeV (#theta = %s^{o})", element, ene, angle);
  TCanvas* c1  = new TCanvas("c1",name,400,300); c1->SetLeftMargin(0.15);
  TGraph*  gr1 = new TGraphErrors(q1,x1,y1,0,stater1);
  gr1->SetTitle(""); gr1->SetMarkerColor(4);  // blue
  gr1->SetMarkerStyle(22);  gr1->SetMarkerSize(1.4); gr1->Draw("ALP");
  gr1->GetXaxis()->SetTitle("Energy (GeV)"); 
  gr1->GetYaxis()->SetTitle("E#frac{d^{3}#sigma}{dp^{3}} (mb/GeV^{2})"); 

  TGraph* gr2 = new TGraphErrors(q2,x2,y2,0,stater2);
  gr2->SetMarkerColor(2);  // red
  gr2->SetMarkerStyle(23);  gr2->SetMarkerSize(1.4); gr2->Draw("LP");

  TLegend *leg1 = new TLegend(0.55,0.80,0.90,0.90);
  leg1->SetHeader(title); leg1->SetFillColor(0);
  leg1->SetTextSize(0.04);
  leg1->Draw();

  sprintf (name, "Chi%s%sGeV%sdeg", element, ene, angle);
  TCanvas *c2  = new TCanvas("c2",name,400,300); c2->SetLeftMargin(0.15);
  TGraph  *gr3 = new TGraph(n,x,chi);
  gr3->SetTitle("");  gr3->SetMarkerStyle(22);  gr3->SetMarkerColor(4);
  gr3->GetXaxis()->SetTitle("Energy (GeV)"); gr3->GetYaxis()->SetTitle("#chi^{2}");  
  gr3->Draw("ALP"); gr3->SetMarkerSize(1.4);
  leg1->Draw();

  TGraph*  gr4 = new TGraphErrors(n,x,dif,0,edif);
  gr4->SetTitle("");  gr4->SetMarkerStyle(20); gr4->SetMarkerSize(1.25);
  gr4->SetMarkerColor(6);
  gr4->GetYaxis()->SetRangeUser(-25.,25.);
  gr4->GetXaxis()->SetTitle("Energy (GeV)"); 
  gr4->GetYaxis()->SetTitle("Difference (%)");
  double xmin = gr4->GetXaxis()->GetXmin();
  double xmax = gr4->GetXaxis()->GetXmax();
  cout << " Xmin " << xmin << " " << xmax << "\n";
  sprintf (name, "Diff%s%sGeV%sdeg", element, ene, angle);
  TCanvas *c3  = new TCanvas("c3",name,400,300); c3->SetLeftMargin(0.15);
  TLine   *line = new TLine(xmin,dify,xmax,dify);
  line->SetLineStyle(2); line->SetLineColor(4);
  gr4->Draw("AP"); line->Draw();
  leg1->Draw();

  if (save > 0) {
    char fname[60];
    sprintf (fname, "%s%sGeV%sdeg_1.eps",  element, ene, angle);
    c1->SaveAs(fname);
    sprintf (fname, "%s%sGeV%sdeg_2.eps",  element, ene, angle);
    c2->SaveAs(fname);
    sprintf (fname, "%s%sGeV%sdeg_3.eps",  element, ene, angle);
    c3->SaveAs(fname);
  }
}

void plotKE4(char element[2], char ene[6], int first=0, int logy=0, int save=0) {

  setStyle();  
  TCanvas *myc = new TCanvas("myc","",800,600); myc->Divide(2,2);

  myc->cd(1); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotKE(element, ene, " 59.1", first, logy);
  myc->cd(2); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotKE(element, ene, " 89.0", first, logy);
  myc->cd(3); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotKE(element, ene, "119.0", first, logy);
  myc->cd(4); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotKE(element, ene, "159.6", first, logy);

  char fname[40];
  if (save > 0) {
    sprintf (fname, "%s%sGeV_1.eps",  element, ene);
    myc->SaveAs(fname);
  }

}

void plotKE1(char element[2], char ene[6], char angle[6], int first=0, int logy=0, int save=0) {

  setStyle();
  TCanvas *myc = new TCanvas("myc","",800,600); myc->SetLeftMargin(0.15);
  if (logy != 0) gPad->SetLogy(1);
  plotKE(element, ene, angle, first, logy);

  char anglx[6], fname[40];
  int nx = 0;
  for (int i=0; i<6; i++) {
    if (angle[i] != ' ') { anglx[nx] = angle[i]; nx++;}
  }
  if (save > 0) {
    sprintf (fname, "%s%sGeV%sdeg.eps",  element, ene, anglx);
    myc->SaveAs(fname);
  }
}

void plotKE(char element[2], char ene[6], char angle[6], int first=0, int logy=0) {

  char fname[40], list[10], hname[40];
  TH1F *hi[5];
  int i=0, icol=1;
  double  ymx0=1, ymi0=100., xlow=0.06, xhigh=0.26;
  for (i=0; i<4; i++) {
    if      (i == 0) {sprintf (list, "QGSP"); icol = 1;}
    else if (i == 1) {sprintf (list, "QGSC"); icol = 2;}
    //    else if (i == 3) {sprintf (list, "FTFP"); icol = 3;}
    else if (i == 2) {sprintf (list, "QGSP_BERT"); icol = 6;}
    else             {sprintf (list, "LHEP"); icol = 7;}
    sprintf (fname, "root/%s%s%sGeV_1.root", element, list, ene);
    sprintf (hname, "KE0%s%s%sGeV%s", element, list, ene, angle);
    TFile *file = new TFile(fname);
    hi[i] = (TH1F*) file->Get(hname);
    //    std::cout << "Get " << hname << " from " << fname <<" as " << hi[i] <<"\n";
    int nx = hi[i]->GetNbinsX();
    for (int k=1; k <= nx; k++) {
      double xx = hi[i]->GetBinCenter(k);
      double yy = hi[i]->GetBinContent(k);
      if (xx > xlow && xx < xhigh) {
	if (yy > ymx0) ymx0 = yy;
	if (yy < ymi0 && yy > 0) ymi0 = yy;
      }
    }
    hi[i]->GetXaxis()->SetRangeUser(xlow, xhigh); hi[i]->SetTitle("");
    hi[i]->SetLineStyle(1);  hi[i]->SetLineWidth(2); hi[i]->SetLineColor(icol);
    //    file->Close();
  }

  char anglx[6];
  int nx = 0;
  for (i=0; i<6; i++) {
    if (angle[i] != ' ') { anglx[nx] = angle[i]; nx++;}
  }
  sprintf (fname, "itep/data/%s%sGeV%sdeg.dat",  element, ene, anglx);
  //  std::cout << "Reads data from file " << fname << "\n";
  ifstream infile;
  infile.open(fname);
  
  int     q1;
  float   m1, r1, x1[30], y1[30], stater1[30], syser1[30];
  infile >> m1 >> r1 >> q1;
  for (i=0; i<q1; i++) {
    infile >> x1[i] >> y1[i] >> stater1[i] >> syser1[i];
    if (y1[i]+stater1[i] > ymx0) ymx0 = y1[i]+stater1[i];    if (y1[i]-stater1[i] < ymi0 && y1[i]-stater1[i] > 0) ymi0 = y1[i]-stater1[i];
  }
  TGraph*  gr1 = new TGraphErrors(q1,x1,y1,0,stater1);
  gr1->SetMarkerColor(4);  gr1->SetMarkerStyle(22);
  gr1->SetMarkerSize(1.6);

  if (logy == 0) {ymx0 *= 1.5; ymi0 *= 0.8;}
  else           {ymx0 *=10.0; ymi0 *= 0.2; }
  for (i = 0; i<4; i++)
    hi[i]->GetYaxis()->SetRangeUser(ymi0,ymx0);

  hi[first]->GetYaxis()->SetTitleOffset(1.6);
  hi[first]->Draw();
  for (i=0; i<4; i++) {
    if (i != first)  hi[i]->Draw("same");
  }
  gr1->Draw("p");

  TLegend *leg1 = new TLegend(0.50,0.60,0.90,0.90);
  for (i=0; i<4; i++) {
    if      (i == 0) sprintf (list, "QGSP"); 
    else if (i == 1) sprintf (list, "QGSC"); 
    else if (i == 2) sprintf (list, "QGSP_BERT"); 
    else             sprintf (list, "LHEP"); 
    leg1->AddEntry(hi[i],list,"F");
  }
  char header[50];
  sprintf (header, "p+%s at %s GeV (#theta = %s^{o})", element, ene, angle);
  leg1->SetHeader(header); leg1->SetFillColor(0);
  leg1->SetTextSize(0.04);
  leg1->Draw();

}

void plotCT4(char element[2], char ene[6], int first=0, int scan=1, int logy=0, int save=0) {

  setStyle();
  TCanvas *myc = new TCanvas("myc","",800,600); myc->Divide(2,2);

  myc->cd(1); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotCT(element, ene, 0.09, first, scan, logy); 
  myc->cd(2); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotCT(element, ene, 0.15, first, scan, logy); 
  myc->cd(3); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotCT(element, ene, 0.19, first, scan, logy); 
  myc->cd(4); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotCT(element, ene, 0.23, first, scan, logy); 

  char fname[40];
  if (save > 0) {
    sprintf (fname, "%s%sGeV_2.eps",  element, ene);
    myc->SaveAs(fname);
  }
}

void plotCT1(char element[2], char ene[6], double ke, int first=0, int scan=1, int logy=0, int save=0) {

  setStyle();
  TCanvas *myc = new TCanvas("myc","",800,600); myc->SetLeftMargin(0.15);
  if (logy != 0) gPad->SetLogy(1);
  plotCT(element, ene, ke, first, scan, logy);

  char fname[40];
  if (save > 0) {
    sprintf (fname, "%s%sGeV%4.2fGeV.eps", element, ene, ke);
    myc->SaveAs(fname);
  }
}

void plotCT(char element[2], char ene[6], double ke, int first=0, int scan=1, int logy=0) {

  static double pi  = 3.1415926;
  static double deg = pi/180.; 
  //  std::cout << "Scan " << scan;
  float  angles[30];
  int    nn=0;
  if (scan > 1) {
    angles[0]  = 10.1;
    angles[1]  = 15.0;
    angles[2]  = 19.8;
    angles[3]  = 24.8;
    angles[4]  = 29.5;
    angles[5]  = 34.6;
    angles[6]  = 39.6;
    angles[7]  = 44.3;
    angles[8]  = 49.3;
    angles[9]  = 54.2;
    angles[10] = 59.1;
    angles[11] = 64.1;
    angles[12] = 69.1;
    angles[13] = 74.1;
    angles[14] = 79.1;
    angles[15] = 84.1;
    angles[16] = 89.0;
    angles[17] = 98.9;
    angles[18] = 108.9;
    angles[19] = 119.0;
    angles[20] = 129.1;
    angles[21] = 139.1;
    angles[22] = 149.3;
    angles[23] = 159.6;
    angles[24] = 161.4;
    angles[25] = 165.5;
    angles[26] = 169.5;
    angles[27] = 173.5;
    angles[28] = 177.0;
    nn         = 29;
  } else {
    angles[0]  = 59.1;
    angles[1]  = 89.0;
    angles[2]  = 119.0;
    angles[3]  = 159.6;
    nn         = 4;
  }
  //  std::cout << " gives " << nn << " angles\n";

  char fname[40], list[10], hname[40];
  TH1F *hi[5];
  int i=0, icol=1;
  double  ymx0=1, ymi0=100., xlow=-1.0, xhigh=1.0;
  for (i=0; i<4; i++) {
    if      (i == 0) {sprintf (list, "QGSP"); icol = 1;}
    else if (i == 1) {sprintf (list, "QGSC"); icol = 2;}
    //    else if (i == 3) {sprintf (list, "FTFP"); icol = 3;}
    else if (i == 2) {sprintf (list, "QGSP_BERT"); icol = 6;}
    else             {sprintf (list, "LHEP"); icol = 7;}
    sprintf (fname, "root/%s%s%sGeV_1.root", element, list, ene);
    sprintf (hname, "CT0%s%s%sGeV%4.2f", element, list, ene, ke);
    TFile *file = new TFile(fname);
    hi[i] = (TH1F*) file->Get(hname);
    //    std::cout << "Get " << hname << " from " << fname <<" as " << hi[i] <<"\n";
    int nx = hi[i]->GetNbinsX();
    for (int k=1; k <= nx; k++) {
      double xx = hi[i]->GetBinCenter(k);
      double yy = hi[i]->GetBinContent(k);
      if (xx > xlow && xx < xhigh) {
	if (yy > ymx0)           ymx0 = yy;
	if (yy < ymi0 && yy > 0) ymi0 = yy;
      }
    }
    hi[i]->GetXaxis()->SetRangeUser(xlow, xhigh); hi[i]->SetTitle("");
    hi[i]->SetLineStyle(1);  hi[i]->SetLineWidth(2); hi[i]->SetLineColor(icol);
    //    file->Close();
  }

  int     q1;
  float   m1, r1, x1[30], y1[30], stater1[30], syser1[30];

  for (int kk=0; kk<nn; kk++) {
    char angle[6], anglx[6];
    sprintf (angle, "%5.1f", angles[kk]);
    int nx = 0;
    for (i=0; i<6; i++) {
      if (angle[i] != ' ') { anglx[nx] = angle[i]; nx++;}
    }
    sprintf (fname, "itep/data/%s%sGeV%sdeg.dat",  element, ene, anglx);
    ifstream infile;
    infile.open(fname);
  
    infile >> m1 >> r1 >> q1;
    for (i=0; i<q1; i++) {
      float xx1, yy1, stater, syser;
      infile >> xx1 >> yy1 >> stater >> syser;
      if (xx1 > ke-0.001 && xx1 < ke+0.001) {
	y1[kk] = yy1; stater1[kk] = stater; syser1[kk] = syser;
      }
    }
    infile.close();
    x1[kk] = cos(deg*angles[kk]);
    if (y1[kk]+stater1[kk] > ymx0) ymx0 = y1[kk]+stater1[kk];
    if (y1[kk]-stater1[kk] < ymi0 && y1[kk]-stater1[kk] > 0) ymi0 = y1[kk]-stater1[kk];
    //    std::cout << kk << " File " << fname << " X " << x1[kk] << " Y " << y1[kk] << " DY " << stater1[kk] << "\n";
  }

  TGraph*  gr1 = new TGraphErrors(nn,x1,y1,0,stater1);
  gr1->SetMarkerColor(4);  gr1->SetMarkerStyle(22);
  gr1->SetMarkerSize(1.6);

  if (logy == 0) {ymx0 *= 1.5; ymi0 *= 0.8;}
  else           {ymx0 *=10.0; ymi0 *= 0.2; }
  for (i = 0; i<4; i++)
    hi[i]->GetYaxis()->SetRangeUser(ymi0,ymx0);
  
  hi[first]->GetYaxis()->SetTitleOffset(1.6);
  hi[first]->Draw();
  for (i=0; i<4; i++) {
    if (i != first)  hi[i]->Draw("same");
  }
  gr1->Draw("p");

  TLegend *leg1 = new TLegend(0.15,0.60,0.60,0.90);
  for (i=0; i<4; i++) {
    if      (i == 0) sprintf (list, "QGSP"); 
    else if (i == 1) sprintf (list, "QGSC"); 
    else if (i == 2) sprintf (list, "QGSP_BERT"); 
    else             sprintf (list, "LHEP"); 
    leg1->AddEntry(hi[i],list,"F");
  }
  char header[50];
  sprintf (header, "p+%s at %s GeV (KE = %4.2f GeV)", element, ene, ke);
  leg1->SetHeader(header); leg1->SetFillColor(0);
  leg1->SetTextSize(0.04);
  leg1->Draw();

}

void plotBE4(char element[2], int logy=0, int scan=1, int save=0) {

  setStyle();
  TCanvas *myc = new TCanvas("myc","",800,600); myc->Divide(2,2);

  myc->cd(1); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotBE(element, " 59.1", 0.11, logy, scan);
  myc->cd(2); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotBE(element, " 59.1", 0.21, logy, scan);
  myc->cd(3); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotBE(element, "119.0", 0.11, logy, scan);
  myc->cd(4); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotBE(element, "119.0", 0.21, logy, scan);

  char fname[40];
  if (save > 0) {
    sprintf (fname, "%s_1.eps", element);
    myc->SaveAs(fname);
  }
}

void plotBE1(char element[2], char angle[6], double ke, int logy=0, int scan=1, int save=0) {

  setStyle();
  TCanvas *myc = new TCanvas("myc","",800,600); myc->SetLeftMargin(0.15);
  if (logy != 0) gPad->SetLogy(1);
  plotBE(element, angle, ke, logy, scan);

  char anglx[6], fname[40];
  int i=0, nx=0;
  for (i=0; i<6; i++) {
    if (angle[i] != ' ') { anglx[nx] = angle[i]; nx++;}
  }
  if (save > 0) {
    sprintf (fname, "%s%sdeg%4.2fGeV.eps",  element, anglx, ke);
    myc->SaveAs(fname);
  }
}

void plotBE(char element[2], char angle[6], double ke, int logy=0, int scan=1) {

  double ene[15];
  int    nene=0;
  if (scan <= 1) {
    ene[0] = 6.2; ene[1] = 6.5; ene[2] = 7.0; ene[3] = 7.5; 
    ene[4] = 8.2; ene[5] = 8.5; ene[6] = 9.0; nene   = 7;
  } else {
    ene[0] = 1.0; ene[1] = 1.4; ene[2] = 2.0; ene[3] = 3.0; 
    ene[4] = 5.0; ene[5] = 6.0; ene[6] = 6.2; ene[7] = 6.5;
    ene[8] = 7.0; ene[9] = 7.5; ene[10]= 8.2; ene[11]= 8.5; 
    ene[12]= 9.0; nene   = 13;
  }
 
  char anglx[6];
  int i=0, nx=0;
  for (i=0; i<6; i++) {
    if (angle[i] != ' ') { anglx[nx] = angle[i]; nx++;}
  }

  TGraph *gr[4];
  char fname[40], list[10], hname[40];
  int j=0, icol=1, ityp=20;
  double  ymx0=1, ymi0=100., xmi=5.0, xmx=10.0;
  for (i=0; i<4; i++) {
    if      (i == 0) {sprintf (list, "QGSP"); icol = 1; ityp = 24;}
    else if (i == 1) {sprintf (list, "QGSC"); icol = 2; ityp = 29;}
    //    else if (i == 3) {sprintf (list, "FTFP"); icol = 3; ityp = 27;}
    else if (i == 2) {sprintf (list, "QGSP_BERT"); icol = 6; ityp = 25;}
    else             {sprintf (list, "LHEP"); icol = 7; ityp = 26;}
    double yt[15];
    for (j=0; j<nene; j++) {
      sprintf (fname, "root/%s%s%3.1fGeV_1.root", element, list, ene[j]);
      sprintf (hname, "KE0%s%s%3.1fGeV%s", element, list, ene[j], angle);
      TFile *file = new TFile(fname);
      TH1F *hi = (TH1F*) file->Get(hname);
      //       std::cout << "Get " << hname << " from " << fname <<" as " << hi <<"\n";
      int    nk=0, nx = hi->GetNbinsX();
      double yy0=0;
      for (int k=1; k <= nx; k++) {
	double xx0 = hi->GetBinCenter(k);
	if (xx0 > ke-0.01 && xx0 < ke+0.01) {
	  yy0 += hi->GetBinContent(k);
	  nk++;
	}
      }
      if (nk > 0 ) yy0 /= nk;
      if (yy0 > ymx0)            ymx0 = yy0;
      if (yy0 < ymi0 && yy0 > 0) ymi0 = yy0;
      yt[j] = yy0;
      file->Close();
    }
    gr[i] = new TGraph(nene, ene, yt); gr[i]->SetMarkerSize(1.2);
    gr[i]->SetTitle(list); gr[i]->SetLineColor(icol); 
    gr[i]->SetLineStyle(i+1); gr[i]->SetLineWidth(2);
    gr[i]->SetMarkerColor(icol);  gr[i]->SetMarkerStyle(ityp); 
    gr[i]->GetXaxis()->SetTitle("Beam Energy (GeV)");
    //    cout << "Graph " << i << " with " << nene << " points\n";
    //    for (j=0; j<nene; j++)
    //      cout << j << " x " << ene[j] << " y " << yt[j] << "\n";
  }

  double ye[15], dy[15];
  for (j=0; j<nene; j++) {
    sprintf (fname, "itep/data/%s%3.1fGeV%sdeg.dat",  element, ene[j], anglx);
    //    cout << "Reads data from file " << fname << "\n";
    ifstream infile;
    infile.open(fname);
  
    int     q1;
    float   m1, r1, xx, yy, stater, syser;
    infile >> m1 >> r1 >> q1;
    for (i=0; i<q1; i++) {
      infile >> xx >> yy >> stater >> syser;
      if (xx > ke-0.01 && xx < ke+0.01) {
	ye[j] = yy;
	dy[j] = stater;
      }
    }
    infile.close();
    if (ye[j]+dy[j] > ymx0) ymx0 = ye[j]+dy[j];
    if (ye[j]-dy[j] < ymi0 && ye[j]-dy[j] > 0) ymi0 = ye[j]-dy[j];
  }
  //  cout << "Graph Data with " << nene << " points\n";
  //  for (j=0; j<nene; j++)
  //    cout << j << " x " << ene[j] << " y " << ye[j] << " +- " << dy[j] << "\n";
  TGraph*  gr1 = new TGraphErrors(nene,ene,ye,0,dy);
  gr1->SetMarkerColor(1);  gr1->SetMarkerStyle(22);
  gr1->SetMarkerSize(1.6);

  if (logy == 0) {ymx0 *= 1.5; ymi0 *= 0.8;}
  else           {ymx0 *= 10.0; ymi0 *= 0.2; }
  for (i = 0; i<4; i++) {
    gr[i]->GetYaxis()->SetRangeUser(ymi0,ymx0);
    gr[i]->GetXaxis()->SetRangeUser(xmi,xmx);
  }
  gr1->GetXaxis()->SetRangeUser(xmi,xmx);
  gr1->GetYaxis()->SetRangeUser(ymi0,ymx0);

  gr1->GetYaxis()->SetTitleOffset(1.6); gr1->SetTitle("");
  gr1->Draw("ap");
  for (i=0; i<4; i++)
    gr[i]->Draw("lp");
  
  TLegend *leg1 = new TLegend(0.45,0.60,0.90,0.90);
  for (i=0; i<4; i++) {
    if      (i == 0) sprintf (list, "QGSP"); 
    else if (i == 1) sprintf (list, "QGSC"); 
    else if (i == 2) sprintf (list, "QGSP_BERT"); 
    else             sprintf (list, "LHEP"); 
    leg1->AddEntry(gr[i],list,"LP");
  }
  char header[50];
  sprintf (header, "p+%s at (KE = %3.1f GeV, #theta = %s^{o})", element, ke, angle);
  leg1->SetHeader(header); leg1->SetFillColor(0);
  leg1->SetTextSize(0.04);
  leg1->Draw();

}

void setStyle() {

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFrameBorderMode(0);
  gStyle->SetFrameBorderSize(1);  gStyle->SetFrameFillColor(0);
  gStyle->SetFrameFillStyle(0);   gStyle->SetFrameLineColor(1);
  gStyle->SetFrameLineStyle(1);   gStyle->SetFrameLineWidth(1);
  gStyle->SetTitleOffset(1.6,"Y");  gStyle->SetOptStat(0);
  gStyle->SetLegendBorderSize(1);

}
