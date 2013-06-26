{
  
  gROOT->Reset();
  gROOT->Clear();
  
  gStyle->SetNdivisions(5);
  gStyle->SetCanvasBorderMode(0); 
  gStyle->SetPadBorderMode(1);
  gStyle->SetOptTitle(1);
  gStyle->SetStatFont(42);
  gStyle->SetCanvasColor(10);
  gStyle->SetPadColor(0);
  gStyle->SetTitleFont(62,"xy");
  gStyle->SetLabelFont(62,"xy");
  gStyle->SetTitleFontSize(0.07);
  gStyle->SetTitleSize(0.045,"xy");
  gStyle->SetLabelSize(0.05,"xy");
  gStyle->SetHistFillStyle(1001);
  gStyle->SetHistFillColor(0);
  gStyle->SetHistLineStyle(1);
  gStyle->SetHistLineWidth(1);
  gStyle->SetHistLineColor(1);
  gStyle->SetTitleXOffset(1.1);
  gStyle->SetTitleYOffset(1.15);
  //gStyle->SetOptStat(1110);
  gStyle->SetOptStat(kFALSE);
  gStyle->SetOptFit(0111);
  gStyle->SetStatH(0.1);

  //  string rdirName="general_AssociatorByHits/";

  //  const char * samples=getenv ("SCENARIOS");
  const char * samples="SCENARIOS";
  TString samplestring(samples);
  std::cout<<"Going to analyze the following scenarios:"<<samplestring<<endl;
  TString delim(" ");
  TObjArray *samplelist=samplestring.Tokenize(delim);
  int nscenarios=samplelist->GetEntries();
  TFile *files[nscenarios];
  TDirectory *dir[nscenarios];
  TH1F *fakeratevseta[nscenarios], *fakeratevsPt[nscenarios], *fakeratevshits[nscenarios];
  TH1F *efficvseta[nscenarios], *efficvsPt[nscenarios], *efficvshits[nscenarios];
  TH1F *sigmadxyvseta[nscenarios], *sigmadxyvspt[nscenarios], *pullDxy[nscenarios];
  TH1F *sigmadzvseta[nscenarios], *sigmadzvspt[nscenarios], *pullDz[nscenarios];
  TH1F *sigmaptvseta[nscenarios], *sigmaptvspt[nscenarios], *pullPt[nscenarios];
  for(int i=0;i<nscenarios;i++){
    TObjString * string =(TObjString*)samplelist->At(i);
    files[i]=new TFile("val.SAMPLE."+string->GetString()+".root");  
    files[i]->cd("DQMData/RecoTrackV/Track");
    TDirectory * dir[i]=gDirectory;
    TList *l= dir[i]->GetListOfKeys();
    string rdirName=l->At(0)->GetName();
    //fake
    dir[i]->GetObject((rdirName+"/fakerate").c_str(),fakeratevseta[i]);
    dir[i]->GetObject((rdirName+"/fakeratePt").c_str(),fakeratevsPt[i]);
    dir[i]->GetObject((rdirName+"/fakerate_vs_hit").c_str(),fakeratevshits[i]);
    //efficiency
    dir[i]->GetObject((rdirName+"/effic").c_str(),efficvseta[i]);
    dir[i]->GetObject((rdirName+"/efficPt").c_str(),efficvsPt[i]);
    dir[i]->GetObject((rdirName+"/effic_vs_hit").c_str(),efficvshits[i]);
    //dxy
    dir[i]->GetObject((rdirName+"/sigmadxy").c_str(),sigmadxyvseta[i]);
    dir[i]->GetObject((rdirName+"/sigmadxyPt").c_str(),sigmadxyvspt[i]);
    dir[i]->GetObject((rdirName+"/pullDxy").c_str(),pullDxy[i]);
    //dz
    dir[i]->GetObject((rdirName+"/sigmadz").c_str(),sigmadzvseta[i]);
    dir[i]->GetObject((rdirName+"/sigmadzPt").c_str(),sigmadzvspt[i]);
    dir[i]->GetObject((rdirName+"/pullDz").c_str(),pullDz[i]);
    //dpt
    dir[i]->GetObject((rdirName+"/sigmapt").c_str(),sigmaptvseta[i]);
    dir[i]->GetObject((rdirName+"/sigmaptPt").c_str(),sigmaptvspt[i]);
    dir[i]->GetObject((rdirName+"/pullPt").c_str(),pullPt[i]);
  }

  //&&&&&&&&&&&&&&&&&&&&&&&&&&&
  // fakerate vs eta
  //&&&&&&&&&&&&&&&&&&&&&&&&&&&
  
  TCanvas *c1 = new TCanvas("c1", "c1",129,17,926,703);
  // c1->SetBorderSize(2);
  // c1->SetFrameFillColor(0);
  // c1->SetFillColor(0);
  c1->SetGrid(1,1);
  for(int i=0;i<nscenarios;i++){
    
    fakeratevseta[i]->SetTitle("Fake rate vs #eta for SAMPLE events");
    fakeratevseta[i]->SetMarkerStyle(20+i);
    fakeratevseta[i]->SetMarkerColor(2+i);
    fakeratevseta[i]->SetMarkerSize(0.9);
    fakeratevseta[i]->SetLineColor(1);
    fakeratevseta[i]->SetLineWidth(1);
  }  

  TLegend *leg1 = new TLegend(0.1,0.75,0.35,0.9);
  leg1->SetTextAlign(32);
  leg1->SetTextColor(1);
  leg1->SetTextSize(0.04);

  for(int i=0;i<nscenarios;i++){
    fakeratevseta[i]->SetTitle("Fake rate vs #eta for SAMPLE events");
    fakeratevseta[i]->SetXTitle("#eta");
    fakeratevseta[i]->SetYTitle("Fake Rate");
    fakeratevseta[i]->SetAxisRange(-0.005,0.4,"Y");
    if(i==0)fakeratevseta[i]->Draw();
    else fakeratevseta[i]->Draw("same");
    leg1->AddEntry(fakeratevseta[i],((TObjString *)(samplelist->At(i)))->GetString(),"P");
  }

  leg1->Draw();
  
  c1->Update();
  c1->SaveAs("Fake_eta_SAMPLE.eps");
  c1->SaveAs("Fake_eta_SAMPLE.gif");
  c1->WaitPrimitive();

  delete c1;
  delete leg1;
  //&&&&&&&&&&&&&&&&&&&&&&&&&&&
  // fakerate vs Pt
  //&&&&&&&&&&&&&&&&&&&&&&&&&&&
  
  TCanvas *c1 = new TCanvas("c1", "c1",129,17,926,703);
  // c1->SetBorderSize(2);
  // c1->SetFrameFillColor(0);
  // c1->SetFillColor(0);
  c1->SetGrid(1,1);
  for(int i=0;i<nscenarios;i++){
    
    fakeratevsPt[i]->SetTitle("Fake rate vs Pt for SAMPLE events");
    fakeratevsPt[i]->SetMarkerStyle(20+i);
    fakeratevsPt[i]->SetMarkerColor(2+i);
    fakeratevsPt[i]->SetMarkerSize(0.9);
    fakeratevsPt[i]->SetLineColor(1);
    fakeratevsPt[i]->SetLineWidth(1);
  }  

  TLegend *leg1 = new TLegend(0.1,0.75,0.35,0.9);
  leg1->SetTextAlign(32);
  leg1->SetTextColor(1);
  leg1->SetTextSize(0.04);

  for(int i=0;i<nscenarios;i++){
    fakeratevsPt[i]->SetTitle("Fake rate vs Pt for SAMPLE events");
    fakeratevsPt[i]->SetXTitle("Pt");
    fakeratevsPt[i]->SetYTitle("Fake Rate");
    fakeratevsPt[i]->SetAxisRange(-0.005,0.5,"Y");
    if(i==0)fakeratevsPt[i]->Draw();
    else fakeratevsPt[i]->Draw("same");
    leg1->AddEntry(fakeratevsPt[i],((TObjString *)(samplelist->At(i)))->GetString(),"P");
  }

  leg1->Draw();
  
  c1->Update();
  c1->SaveAs("Fake_Pt_SAMPLE.eps");
  c1->SaveAs("Fake_Pt_SAMPLE.gif");
  c1->WaitPrimitive();

  delete c1;
  delete leg1;
  //&&&&&&&&&&&&&&&&&&&&&&&&&&&
  // fakerate vs hits
  //&&&&&&&&&&&&&&&&&&&&&&&&&&&
  
  TCanvas *c1 = new TCanvas("c1", "c1",129,17,926,703);
  // c1->SetBorderSize(2);
  // c1->SetFrameFillColor(0);
  // c1->SetFillColor(0);
  c1->SetGrid(1,1);
  for(int i=0;i<nscenarios;i++){
    
    fakeratevshits[i]->SetTitle("Fake rate vs hits for SAMPLE events");
    fakeratevshits[i]->SetMarkerStyle(20+i);
    fakeratevshits[i]->SetMarkerColor(2+i);
    fakeratevshits[i]->SetMarkerSize(0.9);
    fakeratevshits[i]->SetLineColor(1);
    fakeratevshits[i]->SetLineWidth(1);
  }  

  TLegend *leg1 = new TLegend(0.65,0.75,0.9,0.9);
  leg1->SetTextAlign(32);
  leg1->SetTextColor(1);
  leg1->SetTextSize(0.04);

  for(int i=0;i<nscenarios;i++){
    fakeratevshits[i]->SetTitle("Fake rate vs hits for SAMPLE events");
    fakeratevshits[i]->SetXTitle("hits");
    fakeratevshits[i]->SetYTitle("Fake Rate");
    fakeratevshits[i]->SetAxisRange(-0.005,0.7,"Y");
    if(i==0)fakeratevshits[i]->Draw();
    else fakeratevshits[i]->Draw("same");
    leg1->AddEntry(fakeratevshits[i],((TObjString *)(samplelist->At(i)))->GetString(),"P");
  }

  leg1->Draw();
  
  c1->Update();
  c1->SaveAs("Fake_hits_SAMPLE.eps");
  c1->SaveAs("Fake_hits_SAMPLE.gif");
  c1->WaitPrimitive();

  delete c1;
  delete leg1;
  //&&&&&&&&&&&&&&&&&&&&&&&&&&&
  // effic vs eta
  //&&&&&&&&&&&&&&&&&&&&&&&&&&&
  
  TCanvas *c1 = new TCanvas("c1", "c1",129,17,926,703);
  // c1->SetBorderSize(2);
  // c1->SetFrameFillColor(0);
  // c1->SetFillColor(0);
  c1->SetGrid(1,1);
  for(int i=0;i<nscenarios;i++){
    
    efficvseta[i]->SetTitle("Efficiency vs #eta for SAMPLE events");
    efficvseta[i]->SetMarkerStyle(20+i);
    efficvseta[i]->SetMarkerColor(2+i);
    efficvseta[i]->SetMarkerSize(0.9);
    efficvseta[i]->SetLineColor(1);
    efficvseta[i]->SetLineWidth(1);
  }  

  TLegend *leg1 = new TLegend(0.1,0.1,0.35,0.35);
  leg1->SetTextAlign(32);
  leg1->SetTextColor(1);
  leg1->SetTextSize(0.04);

  for(int i=0;i<nscenarios;i++){
    efficvseta[i]->SetXTitle("#eta");
    efficvseta[i]->SetYTitle("Efficiency");
    efficvseta[i]->SetAxisRange(-0.005,1.,"Y");
    if(i==0)efficvseta[i]->Draw();
    else efficvseta[i]->Draw("same");
    leg1->AddEntry(efficvseta[i],((TObjString *)(samplelist->At(i)))->GetString(),"P");
  }

  leg1->Draw();
  
  c1->Update();
  c1->SaveAs("Efficiency_eta_SAMPLE.eps");
  c1->SaveAs("Efficiency_eta_SAMPLE.gif");
  c1->WaitPrimitive();

  delete c1;
  delete leg1;
  //&&&&&&&&&&&&&&&&&&&&&&&&&&&
  // effic vs Pt
  //&&&&&&&&&&&&&&&&&&&&&&&&&&&
  
  TCanvas *c1 = new TCanvas("c1", "c1",129,17,926,703);
  // c1->SetBorderSize(2);
  // c1->SetFrameFillColor(0);
  // c1->SetFillColor(0);
  c1->SetGrid(1,1);
  for(int i=0;i<nscenarios;i++){
    
    efficvsPt[i]->SetTitle("Efficiency vs Pt for SAMPLE events");
    efficvsPt[i]->SetMarkerStyle(20+i);
    efficvsPt[i]->SetMarkerColor(2+i);
    efficvsPt[i]->SetMarkerSize(0.9);
    efficvsPt[i]->SetLineColor(1);
    efficvsPt[i]->SetLineWidth(1);
  }  

  TLegend *leg1 = new TLegend(0.2,0.1,0.45,0.35);
  leg1->SetTextAlign(32);
  leg1->SetTextColor(1);
  leg1->SetTextSize(0.04);

  for(int i=0;i<nscenarios;i++){
    efficvsPt[i]->SetXTitle("Pt");
    efficvsPt[i]->SetYTitle("Efficiency");
    efficvsPt[i]->SetAxisRange(-0.005,1.,"Y");
    if(i==0)efficvsPt[i]->Draw();
    else efficvsPt[i]->Draw("same");
    leg1->AddEntry(efficvsPt[i],((TObjString *)(samplelist->At(i)))->GetString(),"P");
  }

  leg1->Draw();
  
  c1->Update();
  c1->SaveAs("Efficiency_Pt_SAMPLE.eps");
  c1->SaveAs("Efficiency_Pt_SAMPLE.gif");
  c1->WaitPrimitive();

  delete c1;
  delete leg1;
  //&&&&&&&&&&&&&&&&&&&&&&&&&&&
  // effic vs hits
  //&&&&&&&&&&&&&&&&&&&&&&&&&&&
  
  TCanvas *c1 = new TCanvas("c1", "c1",129,17,926,703);
  // c1->SetBorderSize(2);
  // c1->SetFrameFillColor(0);
  // c1->SetFillColor(0);
  c1->SetGrid(1,1);
  for(int i=0;i<nscenarios;i++){
    
    efficvshits[i]->SetTitle("Efficiency vs hits for SAMPLE events");
    efficvshits[i]->SetMarkerStyle(20+i);
    efficvshits[i]->SetMarkerColor(2+i);
    efficvshits[i]->SetMarkerSize(0.9);
    efficvshits[i]->SetLineColor(1);
    efficvshits[i]->SetLineWidth(1);
  }  


// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// /// EFFICIENCIES VS HITS STACKED
// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  TLegend *leg1 = new TLegend(0.2,0.1,0.45,0.35);
  leg1->SetTextAlign(32);
  leg1->SetTextColor(1);
  leg1->SetTextSize(0.04);

  for(int i=0;i<nscenarios;i++){
    efficvshits[i]->SetXTitle("hits");
    efficvshits[i]->SetYTitle("Efficiency");
    efficvshits[i]->SetAxisRange(0.,1.,"Y");
    if(i==0)efficvshits[i]->Draw();
    else efficvshits[i]->Draw("same");
    leg1->AddEntry(efficvshits[i],((TObjString *)(samplelist->At(i)))->GetString(),"P");
  }

  leg1->Draw();
  
  c1->Update();
  c1->SaveAs("Efficiency_hits_SAMPLE.eps");
  c1->SaveAs("Efficiency_hits_SAMPLE.gif");
  c1->WaitPrimitive();

  delete c1;
  delete leg1;

  //&&&&&&&&&&&&&&&&&&&&&&&
  // dxy res vs eta
  //%%%%%%%%%%%%%%%%%%%%%%%
  TCanvas *c1 = new TCanvas("c1", "c1",129,17,926,703);
  // c1->SetBorderSize(2);
  // c1->SetFrameFillColor(0);
  // c1->SetFillColor(0);
  c1->SetGrid(1,1);
  for(int i=0;i<nscenarios;i++){
    
    sigmadxyvseta[i]->SetTitle("Dxy resolution vs #eta for SAMPLE events");
    sigmadxyvseta[i]->SetMarkerStyle(20+i);
    sigmadxyvseta[i]->SetMarkerColor(2+i);
    sigmadxyvseta[i]->SetMarkerSize(0.9);
    sigmadxyvseta[i]->SetLineColor(1);
    sigmadxyvseta[i]->SetLineWidth(1);
  }  


  TLegend *leg1 = new TLegend(0.2,0.1,0.45,0.35);
  leg1->SetTextAlign(32);
  leg1->SetTextColor(1);
  leg1->SetTextSize(0.04);

  for(int i=0;i<nscenarios;i++){
    sigmadxyvseta[i]->SetXTitle("#eta");
    sigmadxyvseta[i]->SetYTitle("dxy resolution");
    sigmadxyvseta[i]->SetAxisRange(0.0005,0.1,"Y");
    if(i==0)sigmadxyvseta[i]->Draw();
    else sigmadxyvseta[i]->Draw("same");
    leg1->AddEntry(sigmadxyvseta[i],((TObjString *)(samplelist->At(i)))->GetString(),"P");
  }
  c1->SetLogy();
  leg1->Draw();
  
  c1->Update();
  c1->SaveAs("Dxy_res_eta_SAMPLE.eps");
  c1->SaveAs("Dxy_res_eta_SAMPLE.gif");
  c1->WaitPrimitive();

  delete c1;
  delete leg1;

  //&&&&&&&&&&&&&&&&&&
  // dxy res vs pt
  //&&&&&&&&&&&&&&&&&&
  TCanvas *c1 = new TCanvas("c1", "c1",129,17,926,703);
  // c1->SetBorderSize(2);
  // c1->SetFrameFillColor(0);
  // c1->SetFillColor(0);
  c1->SetGrid(1,1);
  for(int i=0;i<nscenarios;i++){
    
    sigmadxyvspt[i]->SetTitle("Dxy resolution vs Pt for SAMPLE events");
    sigmadxyvspt[i]->SetMarkerStyle(20+i);
    sigmadxyvspt[i]->SetMarkerColor(2+i);
    sigmadxyvspt[i]->SetMarkerSize(0.9);
    sigmadxyvspt[i]->SetLineColor(1);
    sigmadxyvspt[i]->SetLineWidth(1);
  }  


  TLegend *leg1 = new TLegend(0.1,0.1,0.35,0.35);
  leg1->SetTextAlign(32);
  leg1->SetTextColor(1);
  leg1->SetTextSize(0.04);

  for(int i=0;i<nscenarios;i++){
    sigmadxyvspt[i]->SetXTitle("Pt");
    sigmadxyvspt[i]->SetYTitle("dxy resolution");
    sigmadxyvspt[i]->SetAxisRange(0.,30,"X");
    sigmadxyvspt[i]->SetAxisRange(0.0005,0.1,"Y");
    if(i==0)sigmadxyvspt[i]->Draw();
    else sigmadxyvspt[i]->Draw("same");
    leg1->AddEntry(sigmadxyvspt[i],((TObjString *)(samplelist->At(i)))->GetString(),"P");
  }

  leg1->Draw();
  c1->SetLogy();
  c1->Update();
  c1->SaveAs("Dxy_res_Pt_SAMPLE.eps");
  c1->SaveAs("Dxy_res_Pt_SAMPLE.gif");
  c1->WaitPrimitive();

  delete c1;
  delete leg1;

// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// pull dxy
// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  TCanvas *c1 = new TCanvas("c1", "c1",129,17,926,703);
  // c1->SetBorderSize(2);
  // c1->SetFrameFillColor(0);
  // c1->SetFillColor(0);
  c1->SetGrid(1,1);
  for(int i=0;i<nscenarios;i++){
    
    pullDxy[i]->SetTitle("Pull dxy for SAMPLE events");
    pullDxy[i]->SetMarkerStyle(20+i);
    pullDxy[i]->SetMarkerColor(2+i);
    pullDxy[i]->SetMarkerSize(0.9);
    pullDxy[i]->SetLineColor(2+i);
    pullDxy[i]->SetLineWidth(1);
  }  

  TLegend *leg1 = new TLegend(0.1,0.6,0.35,0.85);
  leg1->SetTextAlign(32);
  leg1->SetTextColor(1);
  leg1->SetTextSize(0.04);

  for(int i=0;i<nscenarios;i++){
    pullDxy[i]->SetXTitle("Pulldxy");
    //    pullDxy[i]->SetYTitle("Pulldxy");
    //    pullDxy[i]->SetAxisRange(-0.005,0.1.,"Y");
    if(i==0)pullDxy[i]->Draw();
    else pullDxy[i]->Draw("same");
    leg1->AddEntry(pullDxy[i],((TObjString *)(samplelist->At(i)))->GetString(),"l");
  }

  leg1->Draw();
  c1->SetLogy();
  c1->Update();
  c1->SaveAs("Pulldxy_SAMPLE.eps");
  c1->SaveAs("Pulldxy_SAMPLE.gif");
  c1->WaitPrimitive();

  delete c1;
  delete leg1;

  //&&&&&&&&&&&&&&&&&&&&&&&
  // dz res vs eta
  //%%%%%%%%%%%%%%%%%%%%%%%
  TCanvas *c1 = new TCanvas("c1", "c1",129,17,926,703);
  // c1->SetBorderSize(2);
  // c1->SetFrameFillColor(0);
  // c1->SetFillColor(0);
  c1->SetGrid(1,1);
  for(int i=0;i<nscenarios;i++){
    
    sigmadzvseta[i]->SetTitle("Dz resolution vs #eta for SAMPLE events");
    sigmadzvseta[i]->SetMarkerStyle(20+i);
    sigmadzvseta[i]->SetMarkerColor(2+i);
    sigmadzvseta[i]->SetMarkerSize(0.9);
    sigmadzvseta[i]->SetLineColor(1);
    sigmadzvseta[i]->SetLineWidth(1);
  }  


  TLegend *leg1 = new TLegend(0.2,0.1,0.45,0.35);
  leg1->SetTextAlign(32);
  leg1->SetTextColor(1);
  leg1->SetTextSize(0.04);

  for(int i=0;i<nscenarios;i++){
    sigmadzvseta[i]->SetXTitle("#eta");
    sigmadzvseta[i]->SetYTitle("dz resolution");
    sigmadzvseta[i]->SetAxisRange(0.0005,0.5,"Y");
    if(i==0)sigmadzvseta[i]->Draw();
    else sigmadzvseta[i]->Draw("same");
    leg1->AddEntry(sigmadzvseta[i],((TObjString *)(samplelist->At(i)))->GetString(),"P");
  }

  leg1->Draw();
  c1->SetLogy();
  
  c1->Update();
  c1->SaveAs("Dz_res_eta_SAMPLE.eps");
  c1->SaveAs("Dz_res_eta_SAMPLE.gif");
  c1->WaitPrimitive();

  delete c1;
  delete leg1;

  //&&&&&&&&&&&&&&&&&&
  // dz res vs pt
  //&&&&&&&&&&&&&&&&&&
  TCanvas *c1 = new TCanvas("c1", "c1",129,17,926,703);
  // c1->SetBorderSize(2);
  // c1->SetFrameFillColor(0);
  // c1->SetFillColor(0);
  c1->SetGrid(1,1);
  for(int i=0;i<nscenarios;i++){
    
    sigmadzvspt[i]->SetTitle("Dz resolution vs Pt for SAMPLE events");
    sigmadzvspt[i]->SetMarkerStyle(20+i);
    sigmadzvspt[i]->SetMarkerColor(2+i);
    sigmadzvspt[i]->SetMarkerSize(0.9);
    sigmadzvspt[i]->SetLineColor(1);
    sigmadzvspt[i]->SetLineWidth(1);
  }  


  TLegend *leg1 = new TLegend(0.2,0.1,0.45,0.35);
  leg1->SetTextAlign(32);
  leg1->SetTextColor(1);
  leg1->SetTextSize(0.04);

  for(int i=0;i<nscenarios;i++){
    sigmadzvspt[i]->SetXTitle("Pt");
    sigmadzvspt[i]->SetYTitle("dz resolution");
    sigmadzvspt[i]->SetAxisRange(0,30.,"X");
    sigmadzvspt[i]->SetAxisRange(0.0005,0.5,"Y");
    if(i==0)sigmadzvspt[i]->Draw();
    else sigmadzvspt[i]->Draw("same");
    leg1->AddEntry(sigmadzvspt[i],((TObjString *)(samplelist->At(i)))->GetString(),"P");
  }

  leg1->Draw();
  c1->SetLogy();
  
  c1->Update();
  c1->SaveAs("Dz_res_Pt_SAMPLE.eps");
  c1->SaveAs("Dz_res_Pt_SAMPLE.gif");
  c1->WaitPrimitive();

  delete c1;
  delete leg1;

// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// pull dz
// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  TCanvas *c1 = new TCanvas("c1", "c1",129,17,926,703);
  // c1->SetBorderSize(2);
  // c1->SetFrameFillColor(0);
  // c1->SetFillColor(0);
  c1->SetGrid(1,1);
  for(int i=0;i<nscenarios;i++){
    
    pullDz[i]->SetTitle("Pull dz for SAMPLE events");
    pullDz[i]->SetMarkerStyle(20+i);
    pullDz[i]->SetMarkerColor(2+i);
    pullDz[i]->SetMarkerSize(0.9);
    pullDz[i]->SetLineColor(2+i);
    pullDz[i]->SetLineWidth(1);
  }  

  TLegend *leg1 = new TLegend(0.1,0.6,0.35,0.85);
  leg1->SetTextAlign(32);
  leg1->SetTextColor(1);
  leg1->SetTextSize(0.04);

  for(int i=0;i<nscenarios;i++){
    pullDz[i]->SetXTitle("Pulldz");
    //    pullDz[i]->SetAxisRange(-0.005,1.,"Y");
    if(i==0)pullDz[i]->Draw();
    else pullDz[i]->Draw("same");
    leg1->AddEntry(pullDz[i],((TObjString *)(samplelist->At(i)))->GetString(),"l");
  }

  leg1->Draw();
  c1->SetLogy();
  
  c1->Update();
  c1->SaveAs("Pulldz_SAMPLE.eps");
  c1->SaveAs("Pulldz_SAMPLE.gif");
  c1->WaitPrimitive();

  delete c1;
  delete leg1;

  //&&&&&&&&&&&&&&&&&&&&&&&
  // pt res vs eta
  //%%%%%%%%%%%%%%%%%%%%%%%
  TCanvas *c1 = new TCanvas("c1", "c1",129,17,926,703);
  // c1->SetBorderSize(2);
  // c1->SetFrameFillColor(0);
  // c1->SetFillColor(0);
  c1->SetGrid(1,1);
  for(int i=0;i<nscenarios;i++){
    
    sigmaptvseta[i]->SetTitle("Pt resolution vs #eta for SAMPLE events");
    sigmaptvseta[i]->SetMarkerStyle(20+i);
    sigmaptvseta[i]->SetMarkerColor(2+i);
    sigmaptvseta[i]->SetMarkerSize(0.9);
    sigmaptvseta[i]->SetLineColor(1);
    sigmaptvseta[i]->SetLineWidth(1);
  }  


  TLegend *leg1 = new TLegend(0.2,0.1,0.45,0.35);
  leg1->SetTextAlign(32);
  leg1->SetTextColor(1);
  leg1->SetTextSize(0.04);

  for(int i=0;i<nscenarios;i++){
    sigmaptvseta[i]->SetXTitle("#eta");
    sigmaptvseta[i]->SetYTitle("pt resolution");
    sigmaptvseta[i]->SetAxisRange(0.0005,0.1,"Y");
    if(i==0)sigmaptvseta[i]->Draw();
    else sigmaptvseta[i]->Draw("same");
    leg1->AddEntry(sigmaptvseta[i],((TObjString *)(samplelist->At(i)))->GetString(),"P");
  }

  leg1->Draw();
  c1->SetLogy();
  
  c1->Update();
  c1->SaveAs("Pt_res_eta_SAMPLE.eps");
  c1->SaveAs("Pt_res_eta_SAMPLE.gif");
  c1->WaitPrimitive();

  delete c1;
  delete leg1;

  //&&&&&&&&&&&&&&&&&&
  // pt res vs pt
  //&&&&&&&&&&&&&&&&&&
  TCanvas *c1 = new TCanvas("c1", "c1",129,17,926,703);
  // c1->SetBorderSize(2);
  // c1->SetFrameFillColor(0);
  // c1->SetFillColor(0);
  c1->SetGrid(1,1);
  for(int i=0;i<nscenarios;i++){
    
    sigmaptvspt[i]->SetTitle("Pt resolution vs Pt for SAMPLE events");
    sigmaptvspt[i]->SetMarkerStyle(20+i);
    sigmaptvspt[i]->SetMarkerColor(2+i);
    sigmaptvspt[i]->SetMarkerSize(0.9);
    sigmaptvspt[i]->SetLineColor(1);
    sigmaptvspt[i]->SetLineWidth(1);
  }  


  TLegend *leg1 = new TLegend(0.2,0.1,0.45,0.35);
  leg1->SetTextAlign(32);
  leg1->SetTextColor(1);
  leg1->SetTextSize(0.04);

  for(int i=0;i<nscenarios;i++){
    sigmaptvspt[i]->SetXTitle("Pt");
    sigmaptvspt[i]->SetYTitle("pt resolution");
    sigmaptvspt[i]->SetAxisRange(0.,30.,"X");
    sigmaptvspt[i]->SetAxisRange(0.0005,0.1,"Y");
    if(i==0)sigmaptvspt[i]->Draw();
    else sigmaptvspt[i]->Draw("same");
    leg1->AddEntry(sigmaptvspt[i],((TObjString *)(samplelist->At(i)))->GetString(),"P");
  }

  leg1->Draw();
  c1->SetLogy();
  
  c1->Update();
  c1->SaveAs("Pt_res_Pt_SAMPLE.eps");
  c1->SaveAs("Pt_res_Pt_SAMPLE.gif");
  c1->WaitPrimitive();

  delete c1;
  delete leg1;

// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// pull pt
// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  TCanvas *c1 = new TCanvas("c1", "c1",129,17,926,703);
  // c1->SetBorderSize(2);
  // c1->SetFrameFillColor(0);
  // c1->SetFillColor(0);
  c1->SetGrid(1,1);
  for(int i=0;i<nscenarios;i++){
    
    pullPt[i]->SetTitle("Pull pt for SAMPLE events");
    pullPt[i]->SetMarkerStyle(20+i);
    pullPt[i]->SetMarkerColor(2+i);
    pullPt[i]->SetMarkerSize(0.9);
    pullPt[i]->SetLineColor(2+i);
    pullPt[i]->SetLineWidth(1);
  }  

  TLegend *leg1 = new TLegend(0.1,0.6,0.35,0.85);
  leg1->SetTextAlign(32);
  leg1->SetTextColor(1);
  leg1->SetTextSize(0.04);

  for(int i=0;i<nscenarios;i++){
    pullPt[i]->SetXTitle("Pullpt");
    //    pullPt[i]->SetAxisRange(-0.005,1.,"Y");
    if(i==0)pullPt[i]->Draw();
    else pullPt[i]->Draw("same");
    leg1->AddEntry(pullPt[i],((TObjString *)(samplelist->At(i)))->GetString(),"l");
  }

  leg1->Draw();
  c1->SetLogy();
  
  c1->Update();
  c1->SaveAs("Pullpt_SAMPLE.eps");
  c1->SaveAs("Pullpt_SAMPLE.gif");
  c1->WaitPrimitive();

  delete c1;
  delete leg1;
  delete samplelist;
  gROOT->Reset();
  gROOT->Clear();
  
}

