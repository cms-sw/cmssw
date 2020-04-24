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
  TPad* tPad = dynamic_cast<TPad*>(gPad);

  TH1* pt_ = 0;
  gDirectory->GetObject("pt_", pt_);
  pt_->Draw();
  styles.FormatPad( tPad, true, false, true);
  styles.FormatHisto( pt_, styles.spred );

  TCanvas c2;
  TH1* eta_ = 0;
  gDirectory->GetObject("eta_", eta_);
  eta_->Draw();
  styles.FormatPad( tPad, true, false, false);
  styles.FormatHisto( eta_, styles.spred );

  TCanvas c3;
  TH1* phi_ = 0;
  gDirectory->GetObject("phi_", phi_);
  phi_->Draw();
  styles.FormatPad( tPad, true, false, false);
  styles.FormatHisto( phi_, styles.spred );

  TCanvas c4;
  TH1* charge_ = 0;
  gDirectory->GetObject("charge_", charge_);
  charge_->Draw();
  styles.FormatPad( tPad, true, false, false);
  styles.FormatHisto( charge_, styles.spred );

  TCanvas c5;
  TH1* particleId_ = 0;
  gDirectory->GetObject("particleId_", particleId_);
  particleId_->Draw();
  styles.FormatPad( tPad, true, false, false);
  styles.FormatHisto( particleId_, styles.spred );
  particleId_->GetXaxis()->SetBinLabel(1, "h+-");
  particleId_->GetXaxis()->SetBinLabel(2, "e");
  particleId_->GetXaxis()->SetBinLabel(3, "mu");
  particleId_->GetXaxis()->SetBinLabel(4, "#gamma");
  particleId_->GetXaxis()->SetBinLabel(5, "h0");
  particleId_->GetXaxis()->SetBinLabel(6, "HF_h");
  particleId_->GetXaxis()->SetBinLabel(7, "HF_em");
  particleId_->GetXaxis()->SetLabelSize(0.06);
  particleId_->GetXaxis()->SetLabelOffset(0.02);

  TCanvas c6;
  TH1* elementsInBlocksSize_ = 0;
  gDirectory->GetObject("elementsInBlocksSize_", elementsInBlocksSize_);
  elementsInBlocksSize_->Draw();
  styles.FormatPad( tPad, true, false, true);
  styles.FormatHisto( elementsInBlocksSize_, styles.spred );
  
  TCanvas c7;
  TH1* delta_et_Over_et_VS_et_ = 0;
  gDirectory->GetObject("delta_et_Over_et_VS_et_", delta_et_Over_et_VS_et_);
  delta_et_Over_et_VS_et_->Draw("colz");
  styles.FormatPad( tPad, false, false, false);
//   styles.FormatHisto( delta_pt_, styles.spred );
  
}
