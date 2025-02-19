void NormHistos (TH1* h1, TH1* h2) {
  double integ1 = (double)h1->Integral();
  if (integ1>0.) h1->Scale(1./integ1);
  double integ2 = (double)h2->Integral();
  if (integ2>0.) h2->Scale(1./integ2);
  float max = TMath::Max(h1->GetMaximum(),h2->GetMaximum());
  h1->SetMaximum(1.1*max);
  //h1->SetMaximum(1.);
  h2->SetMinimum(1.E-6);
}


//___________________________________________________________
void fillPerfGraphEPreID(TGraph* gr, TH1* hl1,TH1* hl2) {

  int minbin = 0;
  int maxbin = hl1->GetNbinsX()+2;
  double effden = (double)hl1->Integral(minbin,maxbin);
  double rejden = (double)hl2->Integral(minbin,maxbin);

  const int n = 1;
  double lowcut[n] = { 0. };
  double upcut[n] = { 0.4 };
  double x[n], y[n];
  double xerr[n], yerr[n];
  for (int i=0;i<n;i++) {
    x[i] = (double)hl1->Integral(hl1->FindBin(lowcut[i]),hl1->FindBin(upcut[i]))/effden;
    xerr[i] = sqrt(x[i])/effden;
    y[i] = (double)hl2->Integral(hl2->FindBin(lowcut[i]),hl2->FindBin(upcut[i]))/rejden;
    yerr[i] = sqrt(y[i])/rejden;

    cout<<x[i]<<", "<<y[i]<<endl;
    gr->SetPoint(i,x[i],y[i]);
  }
}

//___________________________________________________________
void fillPerfGraphDiscr(TGraph* gr, TH1* hl1,TH1* hl2) {

  int minbin = 0;
  int maxbin = hl1->GetNbinsX()+2;
  double effden = (double)hl1->Integral(minbin,maxbin);
  double rejden = (double)hl2->Integral(minbin,maxbin);

  const int n = 1;
  double lowcut[n] = { 0.6 };
  double upcut[n] = { 1.01 };
  double x[n], y[n];
  double xerr[n], yerr[n];
  for (int i=0;i<n;i++) {
    x[i] = (double)hl1->Integral(hl1->FindBin(lowcut[i]),hl1->FindBin(upcut[i]))/effden;
    xerr[i] = sqrt(x[i])/effden;
    y[i] = (double)hl2->Integral(hl2->FindBin(lowcut[i]),hl2->FindBin(upcut[i]))/rejden;
    yerr[i] = sqrt(y[i])/rejden;

    cout<<x[i]<<", "<<y[i]<<endl;
    gr->SetPoint(i,x[i],y[i]);
  }
}



//___________________________________________________________
void fillPerfGraphEMF(TGraph* gr, TH1* hl1,TH1* hl2) {

  int minbin = 0;
  int maxbin = hl1->GetNbinsX()+2;
  double effden = (double)hl1->Integral(minbin,maxbin);
  double rejden = (double)hl2->Integral(minbin,maxbin);


  const int n = 5;
  double lowcut[n] = { 0., 0., 0., 0., 0. };
  double upcut[n] = { 0.75, 0.8, 0.85, 0.9, 0.95 };
  double x[n], y[n];
  double xerr[n], yerr[n];
  for (int i=0;i<n;i++) {
    x[i] = (double)hl1->Integral(minbin,hl1->FindBin(upcut[i]))/effden;
    xerr[i] = sqrt(x[i])/effden;
    y[i] = (double)hl2->Integral(minbin,hl2->FindBin(upcut[i]))/rejden;
    yerr[i] = sqrt(y[i])/rejden;

    gr->SetPoint(i,x[i],y[i]);
  }
}


//___________________________________________________________
void fillPerfGraphHoP(TGraph* gr, TH1* hl1,TH1* hl2) {

  int minbin = 0;
  int maxbin = hl1->GetNbinsX()+2;
  double effden = (double)hl1->Integral(minbin,maxbin);
  double rejden = (double)hl2->Integral(minbin,maxbin);


  const int n = 5;
  double upcut[n] = { 100., 100., 100., 100., 100. };
  double lowcut[n] = { 0.26, 0.21, 0.16, 0.11, 0.051 };
  double x[n], y[n];
  double xerr[n], yerr[n];
  for (int i=0;i<n;i++) {
    x[i] = (double)hl1->Integral(hl1->FindBin(lowcut[i]),maxbin)/effden;
    xerr[i] = sqrt(x[i])/effden;
    y[i] = (double)hl2->Integral(hl2->FindBin(lowcut[i]),maxbin)/rejden;
    yerr[i] = sqrt(y[i])/rejden;

    gr->SetPoint(i,x[i],y[i]);
  }
}

//___________________________________________________________
void fillPerfGraphESUM(TGraph* gr, TH1* hl1,TH1* hl2) {

  int minbin = 0;
  int maxbin = hl1->GetNbinsX()+2;
  double effden = (double)hl1->Integral(minbin,maxbin);
  double rejden = (double)hl2->Integral(minbin,maxbin);

  const int n = 5;
  double lowcut[n] = { 0., 0., 0., 0., 0. };
  double upcut[n] = { 0.7, 0.75, 0.8, 0.85, 0.9 };
  double x[n], y[n];
  double xerr[n], yerr[n];
  for (int i=0;i<n;i++) {
    x[i] = (double)hl1->Integral(minbin,hl1->FindBin(upcut[i]))/effden;
    xerr[i] = sqrt(x[i])/effden;
    y[i] = (double)hl2->Integral(minbin,hl2->FindBin(upcut[i]))/rejden;
    yerr[i] = sqrt(y[i])/rejden;

    gr->SetPoint(i,x[i],y[i]);
  }
}

//___________________________________________________________
void fillPerfGraphMVA(TGraph* gr, TH1* hl1,TH1* hl2) {

  int minbin = 0;
  int maxbin = hl1->GetNbinsX()+2;
  double effden = (double)hl1->Integral(minbin,maxbin);
  double rejden = (double)hl2->Integral(minbin,maxbin);


  const int n = 7;
  double lowcut[n] = { -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1 };
  double upcut[n] = { -0.6, -0.5, -0.4, -0.3, -0.2 , -0.1 , 0. };
  double x[n], y[n];
  double xerr[n], yerr[n];
  for (int i=0;i<n;i++) {
    x[i] = (double)hl1->Integral(minbin,hl1->FindBin(upcut[i]))/effden;
    xerr[i] = sqrt(x[i])/effden;
    y[i] = (double)hl2->Integral(minbin,hl2->FindBin(upcut[i]))/rejden;
    yerr[i] = sqrt(y[i])/rejden;

    gr->SetPoint(i,x[i],y[i]);
  }
}
