{

  TCanvas *c=new TCanvas("c","c",600,600);
  TH1 *totalpu=0;
   TString url("out.json_targetpu.root");
   TFile *fIn=TFile::Open(url);
   if(fIn==0) continue;
   TH1 *pu=fIn->Get("pileup");
   if(pu==0) continue;
   pu->GetXaxis()->SetTitle("Pileup (observed)");
//   pu->SetTitle(titles[f]);
   pu->SetDirectory(0);
   pu->SetFillColor(41+2);
   pu->SetFillStyle(3003);
   pu->SetLineColor(41+2);
   pu->Draw(totalpu==0?"hist":"histsame");
   if(totalpu==0) { totalpu = (TH1 *) pu->Clone("totalpileup");  totalpu->SetDirectory(0); totalpu->SetTitle("Total"); totalpu->SetLineWidth(2); totalpu->SetLineColor(1); totalpu->SetFillStyle(0); }
   else           { totalpu->Add(pu); }
   fIn->Close();

  totalpu->Draw("histsame");
  TLegend *leg=c->BuildLegend();
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->SetFillColor(0);
  leg->SetTextFont(42);
//  char buf[200];
//  sprintf(buf,"#splitline{CMS preliminary, #sqrt{s}=8 TeV}{#sigma_{MB}=%s #mu b}",mb.Data());
//  leg->SetHeader(buf);
//  leg->SetNColumns(2);

  //print out what to use in the cfg
  cout << endl << "cms.vdouble(" << flush; 
  for(int ibin=1; ibin<=totalpu->GetXaxis()->GetNbins(); ibin++) cout << totalpu->GetBinContent(ibin)/totalpu->Integral() << ",";
  cout << ")" << endl << endl;

}
