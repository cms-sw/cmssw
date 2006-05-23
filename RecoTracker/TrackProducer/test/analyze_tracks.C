{
gSystem->Load("libPhysicsToolsFWLite.so");
AutoLibraryLoader::enable();
TFile file("tracks100.root");
/////////SET THESE VALUES///////////////
float intervals[] = {0,0.8,1.4,2.0,2.5};
int bins = 4;
////////////////////////////////////////
vector<float> ptrms;
vector<float> effic;
TTree * tree = (TTree *) gROOT->FindObject("Events");
TFile outFile("graphs.root","recreate");

string title, title2;
for (int i=1;i<(bins+1);i++){
  ostringstream str0, str1, str2, str3;
  str1 <<"ptres"<<intervals[i];
  title=str1.str();
  str0 <<"Pt residue 4.99<Pt<5.01 "<< intervals[i-1] << "<eta<"<<intervals[i];
  title2=str0.str();
  TH1F ptres2(title.c_str(),title2.c_str(),100,-0.5,0.5);
  str2 << "EmbdSimTracks_SimG4Object__test.obj.momentum().perp()-recoTracks_TrackProducer__ctftracks.obj.pt() >> " << title;
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
  str0 <<"(Missing Tracks)/(Total Tracks) 4.99<Pt<5.01 "<< intervals[i-1] << "<eta<"<<intervals[i];
  title2=str0.str();
  str3 << "fabs(EmbdSimTracks_SimG4Object__test.obj.momentum().pseudoRapidity())>"<< intervals[i-1] <<"&&fabs(EmbdSimTracks_SimG4Object__test.obj.momentum().pseudoRapidity())<"<<intervals[i];
  //str3 << intervals[i-1]<<"<fabs(EmbdSimTracks_SimG4Object__test.obj.momentum().pseudoRapidity())<"<<intervals[i];
  ostringstream str4, str5;
  str4 << "producedtracks_vs_eta" << intervals[i];
  str5 << "EmbdSimTracks_SimG4Object__test.@obj.size() >> " << str4.str();
  TH1F prod(str4.str().c_str(),str4.str().c_str(),10,0,10);
  tree->Draw(str5.str().c_str(),str3.str().c_str(),"goff");
  float tot=prod.GetEntries(); 
  str2 << "(-EmbdSimTracks_SimG4Object__test.@obj.size()+recoTracks_TrackProducer__ctftracks.@obj.size()) >> " << title;
  TH1F eff(title.c_str(),title2.c_str(),3,-1.5,1.5);
  tree->Draw(str2.str().c_str(),str3.str().c_str(),"goff");
  if (tot!=0) eff.Scale(1/tot);
//   cout <<  eff.GetBinContent(eff.FindBin(0)) << endl;
  effic.push_back( eff.GetBinContent( eff.FindBin(0) ) );
  eff.Write();
}

TH1F  ptres("ptres_distrib",  "Pt residue 4.99<Pt<5.01 0<eta<2.5",100,-0.5,0.5);
TH1F etares("etaresidue_distrib","Eta residue 4.99<Pt<5.01 0<eta<2.5",100,-0.005,0.005);
TH1F etagres("etagen_distrib","Eta generated 4.99<Pt<5.01 0<eta<2.5",100,-2.5,2.5);
TH1F etafres("etafound_distrib","Eta found 4.99<Pt<5.01 0<eta<2.5",100,-2.5,2.5);
TH1F chi2histo("chi2_distrib","NChi2 distribution 4.99<Pt<5.01 0<eta<2.5",100,0,10);
tree->Draw("EmbdSimTracks_SimG4Object__test.obj.momentum().perp()-recoTracks_TrackProducer__ctftracks.obj.pt() >> ptres_distrib","","goff");
tree->Draw("EmbdSimTracks_SimG4Object__test.obj.momentum().pseudoRapidity()-recoTracks_TrackProducer__ctftracks.obj.eta() >> etaresidue_distrib","","goff");
tree->Draw("EmbdSimTracks_SimG4Object__test.obj.momentum().pseudoRapidity() >> etagen_distrib","","goff");
tree->Draw("recoTracks_TrackProducer__ctftracks.obj.eta() >> etafound_distrib","","goff");
tree->Draw("recoTracks_TrackProducer__ctftracks.obj.normalizedChi2() >> chi2_distrib","","goff");

TH1F ptrmsh("PtRMS_vs_eta","PtRMS vs eta 4.99<Pt<5.01",bins,intervals);
TH1F effh("eff_vs_eta","efficiency vs eta 4.99<Pt<5.01",bins,intervals);
//TH1F ptrmsh("ptrmsh","ptrmsh",3,0,2.5);
float binfix=0.0001;
for (int i=1;i<(bins+1);i++){
  ptrmsh.Fill(intervals[i]-binfix,ptrms[i-1]);
  effh.Fill(intervals[i]-binfix,effic[i-1]);
}

ptres.Write();
etares.Write();
etagres.Write();
etafres.Write();
ptrmsh.Write();
effh.Write();
chi2histo.Write();
outFile.Close();
}
