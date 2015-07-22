{
gSystem->Load("libFWCoreFWLite.so");
FWLiteEnabler::enable();
TFile file("RefittedTracks.root");
/////////SET THESE VALUES///////////////
float intervals[] = {0,0.8,1.4,2.0,2.5};
int bins = 4;
////////////////////////////////////////
vector<float> ptrms;
vector<float> effic;
TTree * tree = (TTree *) gROOT->FindObject("Events");
TFile outFile("refitgraphs.root","recreate");

string title, title2;
for (int i=1;i<(bins+1);i++){
  cout << i << endl;
  ostringstream str0, str1, str2, str3;
  str1 <<"ptres"<<intervals[i];
  title=str1.str();
  str0 <<"Pt residue (Tracks&refTracks) 4.99<Pt<5.01 "<< intervals[i-1] << "<eta<"<<intervals[i];
  title2=str0.str();
  TH1F ptres2(title.c_str(),title2.c_str(),100,-0.5,0.5);
  str2 << "recoTracks_TrackProducer__Refit.obj.pt()-recoTracks_TrackRefitter__Refit.obj.pt() >> " << title;
  str3 << "fabs(EmbdSimTracks_SimG4Object__test.obj.momentum().pseudoRapidity())>"<< intervals[i-1] <<"&&fabs(EmbdSimTracks_SimG4Object__test.obj.momentum().pseudoRapidity())<"<<intervals[i];
  tree->Draw(str2.str().c_str(),str3.str().c_str(),"goff");
  ptrms.push_back(ptres2.GetRMS());
  ptres2.Write();

  str0.str("");
  str1.str("");
  str2.str("");
  str3.str("");
  str1 <<"missingtracks_vs_eta"<<intervals[i];
  title=str1.str();
  str0 <<"(Missing RefittedTracks)/(Tracks) 4.99<Pt<5.01 "<< intervals[i-1] << "<eta<"<<intervals[i];
  title2=str0.str();
  str3 << "fabs(EmbdSimTracks_SimG4Object__test.obj.momentum().pseudoRapidity())>"<< intervals[i-1] <<"&&fabs(EmbdSimTracks_SimG4Object__test.obj.momentum().pseudoRapidity())<"<<intervals[i];
  //str3 << intervals[i-1]<<"<fabs(EmbdSimTracks_SimG4Object__test.obj.momentum().pseudoRapidity())<"<<intervals[i];
  ostringstream str4, str5;
  str4 << "generatedtracks_vs_eta" << intervals[i];
  str5 << "EmbdSimTracks_SimG4Object__test.@obj.size() >> " << str4.str();
  TH1F prod(str4.str().c_str(),str4.str().c_str(),10,0,10);
  tree->Draw(str5.str().c_str(),str3.str().c_str(),"goff");
  float tot=prod.GetEntries(); 
  str2 << "(recoTracks_TrackProducer__Refit.@obj.size()-recoTracks_TrackRefitter__Refit.@obj.size()) >> " << title;
  TH1F eff(title.c_str(),title2.c_str(),3,-1.5,1.5);
  tree->Draw(str2.str().c_str(),str3.str().c_str(),"goff");
  if (tot!=0) eff.Scale(1/tot);
//   cout <<  eff.GetBinContent(eff.FindBin(0)) << endl;
  effic.push_back( eff.GetBinContent( eff.FindBin(0) ) );
  eff.Write();
}

TH1F  ptres("ptres_distrib",  "Pt residue (Tracks&refTracks) 4.99<Pt<5.01",100,-0.0001,0.0001);
tree->Draw("recoTracks_TrackProducer__Refit.obj.pt()-recoTracks_TrackRefitter__Refit.obj.pt() >> ptres_distrib","","goff");
TH1F etares("etaresidue_distrib","Eta residue (Tracks&refTracks) 4.99<Pt<5.01",100,-0.00001,0.00001);
tree->Draw("recoTracks_TrackProducer__Refit.obj.eta()-recoTracks_TrackRefitter__Refit.obj.eta() >> etaresidue_distrib","","goff");
TH1F chi2histo("chi2_distrib","NChi2 distribution (refTracks) 4.99<Pt<5.01",100,0,10);
tree->Draw("recoTracks_TrackRefitter__Refit.obj.normalizedChi2() >> chi2_distrib","","goff");
// TH1F  ptresV("ptres_distrib_noinvalid",  "Pt residue (Tracks&refTracks) 4.99<Pt<5.01 Tracks without invalid hits",100,-0.0004,0.0004);
TH1F  ptresV("ptres_distrib_noinvalid",  "Pt residue (Tracks&refTracks) 4.99<Pt<5.01 Tracks without invalid hits",100,-0.03,0.03);
tree->Draw("recoTracks_TrackProducer__Refit.obj.pt()-recoTracks_TrackRefitter__Refit.obj.pt() >> ptres_distrib_noinvalid","(recoTracks_TrackProducer__Refit.obj.invalid_-recoTracks_TrackRefitter__Refit.obj.invalid_)==0","goff");
TH1F  ptresI("ptres_distrib_invalid",  "Pt residue (Tracks&refTracks) 4.99<Pt<5.01 Tracks with invalid hits",100,-0.03,0.03);
tree->Draw("recoTracks_TrackProducer__Refit.obj.pt()-recoTracks_TrackRefitter__Refit.obj.pt() >> ptres_distrib_invalid","(recoTracks_TrackProducer__Refit.obj.invalid_-recoTracks_TrackRefitter__Refit.obj.invalid_)!=0","goff");
TH1F  found("found_hits",  "Found Hits difference (Tracks-refTracks) 4.99<Pt<5.01",10,-5,5);
tree->Draw("recoTracks_TrackProducer__Refit.obj.found_-recoTracks_TrackRefitter__Refit.obj.found_ >> found_hits","","goff");
TH1F  invalid("invalid_hits",  "Invalid Hits difference (Tracks-refTracks) 4.99<Pt<5.01",10,-5,5);
tree->Draw("recoTracks_TrackProducer__Refit.obj.invalid_-recoTracks_TrackRefitter__Refit.obj.invalid_ >> invalid_hits","","goff");

TH1F ptrmsh("PtRMS_vs_eta","PtRMS vs eta (Tracks&refTracks) 4.99<Pt<5.01",bins,intervals);
TH1F effh("eff_vs_eta","efficiency vs eta (refTracks) 4.99<Pt<5.01",bins,intervals);
//TH1F ptrmsh("ptrmsh","ptrmsh",3,0,2.5);
float binfix=0.0001;
for (int i=1;i<(bins+1);i++){
  ptrmsh.Fill(intervals[i]-binfix,ptrms[i-1]);
  effh.Fill(intervals[i]-binfix,effic[i-1]);
}
// ptresI.SetLineColor(2);
// ptresI.GetYaxis().SetRangeUser(0,1000);
// ptresI.Draw();
// ptresV.Draw("SAME");
// TLegend * ll = new TLegend(0.6,0.6,0.89,0.79);
// ll->AddEntry("ptres_distrib_invalid","track with invalid hit","L");
// ll->AddEntry("ptres_distrib_valid"  ,"track without invalid hit","L");
// ll->Draw();

// ptres.Write();
// ptresV.Write();
// ptresI.Write();
// etares.Write();
// ptrmsh.Write();
// effh.Write();
// chi2histo.Write();
// found.Write();
// invalid.Write();
// outFile.Close();
}
