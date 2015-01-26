{
  gSystem->Load("libFWCoreFWLite.so");
  gSystem->Load("libValidationRecoParticleFlow.so");

  gStyle->SetOptStat(1111);

  string dir = "/DQMData/Run\ 1/PFTask/Run\ summary/particleFlowManager";
  const char* file = "DQM_V0001_R000000001__A__B__C.root";

  float ptMin = 0;
  float ptMax = 9999;
 
  Styles styles;

  TFile f(file);
  f.cd( dir.c_str() );

  pt_.Draw();
  styles.FormatPad( gPad, true, false, true);
  styles.FormatHisto( pt_, styles.spred );

  TCanvas c2;
  eta_.Draw();
  styles.FormatPad( gPad, true, false, false);
  styles.FormatHisto( eta_, styles.spred );

  TCanvas c3;
  phi_.Draw();
  styles.FormatPad( gPad, true, false, false);
  styles.FormatHisto( phi_, styles.spred );

  TCanvas c4;
  charge_.Draw();
  styles.FormatPad( gPad, true, false, false);
  styles.FormatHisto( charge_, styles.spred );

  TCanvas c5;
  particleId_.Draw();
  styles.FormatPad( gPad, true, false, false);
  styles.FormatHisto( particleId_, styles.spred );
  particleId_.GetXaxis()->SetBinLabel(1, "h+-");
  particleId_.GetXaxis()->SetBinLabel(2, "e");
  particleId_.GetXaxis()->SetBinLabel(3, "mu");
  particleId_.GetXaxis()->SetBinLabel(4, "#gamma");
  particleId_.GetXaxis()->SetBinLabel(5, "h0");
  particleId_.GetXaxis()->SetBinLabel(6, "HF_h");
  particleId_.GetXaxis()->SetBinLabel(7, "HF_em");
  particleId_.GetXaxis()->SetLabelSize(0.06);
  particleId_.GetXaxis()->SetLabelOffset(0.02);

  TCanvas c6;
  elementsInBlocksSize_.Draw();
  styles.FormatPad( gPad, true, false, true);
  styles.FormatHisto( elementsInBlocksSize_, styles.spred );
  
  TCanvas c7;
  delta_et_Over_et_VS_et_.Draw("colz");
  styles.FormatPad( gPad, false, false, false);
//   styles.FormatHisto( delta_pt_, styles.spred );
  
}
