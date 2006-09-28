void plot()
{
  // Plain style
  gROOT->SetStyle("Plain");

  TCanvas c1;
  c1.Divide(2,2);

  // Open file
  TFile file("result.root","read");
  file.ls();
  TNtuple * ntuple = (TNtuple *) file.Get("vzero");

  ntuple->SetMarkerStyle(kOpenCircle);

  TPostScript ps("plot.ps",112);
  ps.Range(26,20);

  c1.cd(1); ntuple->Draw("pt:alpha","pt < 0.3");
  c1.Update();

  c1.cd(2); ntuple->Draw("dcaz");
  c1.Update();

  c1.cd(3); ntuple->Draw("r");
  c1.Update();

  c1.cd(4); ntuple->Draw("b");
  c1.Update();

  ps.Close();
}
