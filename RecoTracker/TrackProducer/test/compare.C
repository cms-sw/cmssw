{
gROOT->SetBatch();
gROOT->SetStyle("Plain");

TFile *_file0 = TFile::Open("histo10GeV.root");
TDirectory * RS = _file0->Get("RSWMaterial");
TDirectory * CTF = _file0->Get("CTFWMaterial");

TCanvas c;

TH1F * ptRS = RS->Get("pt");
TH1F * ptCTF = CTF->Get("pt");
ptRS.SetLineColor( kRed );
ptRS.GetYaxis().SetRangeUser(0,(ptRS.GetMaximum()>ptCTF.GetMaximum() ? ptRS.GetMaximum()*=1.1 : ptCTF.GetMaximum()*=1.1));
ptRS->Draw();
ptCTF->Draw("same");
TLegend * ll = new TLegend(0.8,0.2,0.95,0.35);
ll->AddEntry(ptRS, "RS","L");
ll->AddEntry(ptCTF,"CTF","L");
ll->Draw();
c.SaveAs( "pt.gif" );
c.Clear();

// TH1F * pt2RS = RS->Get("pt2");
// TH1F * pt2CTF = CTF->Get("pt2");
// ptRS.SetLineColor( kRed );
// ptRS.GetYaxis().SetRangeUser(0,(ptRS.GetMaximum()>ptCTF.GetMaximum() ? ptRS.GetMaximum()*=1.1 : ptCTF.GetMaximum()*=1.1));
// ptRS->Draw();
// ptCTF->Draw("same");
// ll->Draw();
// c.SaveAs( "pt2.gif" );
// c.Clear();

TH1F * etaRS = RS->Get("eta");
TH1F * etaCTF = CTF->Get("eta");
etaRS.SetLineColor( kRed );
etaRS.GetYaxis().SetRangeUser(0,(etaRS.GetMaximum()>etaCTF.GetMaximum() ? etaRS.GetMaximum()*=1.1 : etaCTF.GetMaximum()*=1.1));
etaRS->Draw();
etaCTF->Draw("same");
ll->Draw();
c.SaveAs( "eta.gif" );
c.Clear();

TH1F * hitsRS = RS->Get("hits");
TH1F * hitsCTF = CTF->Get("hits");
hitsRS.SetLineColor( kRed );
hitsRS.GetYaxis().SetRangeUser(0,(hitsRS.GetMaximum()>hitsCTF.GetMaximum() ? hitsRS.GetMaximum()*=1.1 : hitsCTF.GetMaximum()*=1.1));
hitsRS->Draw();
hitsCTF->Draw("same");
ll->Draw();
c.SaveAs( "hits.gif" );
c.Clear();

TH1F * efficRS = RS->Get("effic");
TH1F * efficCTF = CTF->Get("effic");
efficRS.SetLineColor( kRed );
efficRS.GetYaxis().SetRangeUser(0,(efficRS.GetMaximum()>efficCTF.GetMaximum() ? efficRS.GetMaximum()*=1.1 : efficCTF.GetMaximum()*=1.1));
efficRS->Draw();
efficCTF->Draw("same");
ll->Draw();
c.SaveAs( "effic.gif" );
c.Clear();

TH1F * nchi2RS = RS->Get("nchi2");
TH1F * nchi2CTF = CTF->Get("nchi2");
nchi2RS.SetLineColor( kRed );
nchi2RS.GetYaxis().SetRangeUser(0,(nchi2RS.GetMaximum()>nchi2CTF.GetMaximum() ? nchi2RS.GetMaximum()*=10 : nchi2CTF.GetMaximum()*=10));
nchi2RS->Draw();
nchi2CTF->Draw("same");
ll->Draw();
c.SaveAs( "nchi2.gif" );
c.Clear();

TH1F * tracksRS = RS->Get("tracks");
TH1F * tracksCTF = CTF->Get("tracks");
tracksRS.SetLineColor( kRed );
tracksRS.GetYaxis().SetRangeUser(0,(tracksRS.GetMaximum()>tracksCTF.GetMaximum() ? tracksRS.GetMaximum()*=1.1 : tracksCTF.GetMaximum()*=1.1));
tracksRS->Draw();
tracksCTF->Draw("same");
ll->Draw();
tracksRS->SetMinimum(0.000001);
tracksCTF->SetMinimum(0.000001);
c.SetLogy();
c.SaveAs( "tracks.gif" );
c.SetLogy(0);
c.Clear();

TH1F * chargeRS = RS->Get("charge");
TH1F * chargeCTF = CTF->Get("charge");
chargeRS.SetLineColor( kRed );
chargeRS.GetYaxis().SetRangeUser(0,(chargeRS.GetMaximum()>chargeCTF.GetMaximum() ? chargeRS.GetMaximum()*=1.1 : chargeCTF.GetMaximum()*=1.1));
chargeRS->Draw();
chargeCTF->Draw("same");
ll->Draw();
chargeRS->SetMinimum(0.000001);
chargeCTF->SetMinimum(0.000001);
c.SetLogy();
c.SaveAs( "charge.gif" );
c.SetLogy(0);
c.Clear();

TH1F * pullPhi0RS = RS->Get("pullPhi0");
TH1F * pullPhi0CTF = CTF->Get("pullPhi0");
pullPhi0RS.SetLineColor( kRed );
pullPhi0RS.GetYaxis().SetRangeUser(0,(pullPhi0RS.GetMaximum()>pullPhi0CTF.GetMaximum() ? pullPhi0RS.GetMaximum()*=1.1 : pullPhi0CTF.GetMaximum()*=1.1));
pullPhi0RS->Draw();
pullPhi0CTF->Draw("same");
ll->Draw();
c.SaveAs( "pullPhi0.gif" );
c.Clear();

TH1F * pullThetaRS = RS->Get("pullTheta");
TH1F * pullThetaCTF = CTF->Get("pullTheta");
pullThetaRS.SetLineColor( kRed );
pullThetaRS.GetYaxis().SetRangeUser(0,(pullThetaRS.GetMaximum()>pullThetaCTF.GetMaximum() ? pullThetaRS.GetMaximum()*=1.1 : pullThetaCTF.GetMaximum()*=1.1));
pullThetaRS->Draw();
pullThetaCTF->Draw("same");
ll->Draw();
c.SaveAs( "pullTheta.gif" );
c.Clear();



}
