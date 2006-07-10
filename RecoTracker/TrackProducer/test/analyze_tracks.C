{
gSystem->Load("libFWCoreFWLite.so");
AutoLibraryLoader::enable();
/////////SET THESE VALUES///////////////
TFile file("reco-application-tracking-finaltrackfits-rsfinalfitwithmaterialORIG.root");
float intervals[] = {0,0.5,1.0,1.5,2.0,2.5};
int bins = 5;
char * ptinterval  = "4.99<P_{T}<5.01";
char * etainterval = "|#eta|<2.5";
char * rs  = "recoTracks_RSWMaterial__RsWithMaterial";
char * ctf = "recoTracks_CTFWMaterial__CtfWMaterial";
char * sim = "EmbdSimTracks_SimG4Object__Simulation";
////////////////////////////////////////
vector<float> ptrmsRS;
vector<float> efficRS;
vector<float> ptrmsCTF;
vector<float> efficCTF;
TTree * tree = (TTree *) gROOT->FindObject("Events");
TFile outFile("graphsORIG.root","recreate");

for (int i=1;i<(bins+1);i++){

  ostringstream title, name, expr, cut;
  //RS
  name <<"ptresRS"<<intervals[i];
  title <<"Pt residue "<< ptinterval << " "<< intervals[i-1] << "<#eta<"<<intervals[i];
  TH1F ptres2RS(name.str().c_str(),title.str().c_str(),100,-1,1);
  expr << ""<<sim<<".obj.momentum().perp()-"<<rs<<".obj.pt() >> " << name.str();
  cut << "fabs("<<sim<<".obj.momentum().pseudoRapidity())>"<< intervals[i-1] <<"&&fabs("<<sim<<".obj.momentum().pseudoRapidity())<"<<intervals[i];
  tree->Draw(expr.str().c_str(),cut.str().c_str(),"goff");
  ptrmsRS.push_back(ptres2RS.GetRMS());
  ptres2RS.Write();

  //CTF
  name.str("");
  expr.str("");
  name <<"ptresCTF"<<intervals[i];
  TH1F ptres2CTF(name.str().c_str(),title.str().c_str(),100,-1,1);
  expr << ""<<sim<<".obj.momentum().perp()-"<<ctf<<".obj.pt() >> " << name.str();
  tree->Draw(expr.str().c_str(),cut.str().c_str(),"goff");
  ptrmsCTF.push_back(ptres2CTF.GetRMS());
  ptres2CTF.Write();


  name.str("");
  title.str("");
  expr.str("");
  cut.str("");
  //   ostringstream str4, str5;
  cut << "fabs("<<sim<<".obj.momentum().pseudoRapidity())>"<< intervals[i-1] <<"&&fabs("<<sim<<".obj.momentum().pseudoRapidity())<"<<intervals[i];
  name << "producedtracks_vs_eta" << intervals[i];
  expr << ""<<sim<<".@obj.size() >> " << name.str();
  title << name.str();
  TH1F prod(name.str().c_str(),title.str().c_str(),10,0,10);
  tree->Draw(expr.str().c_str(),cut.str().c_str(),"goff");
  float tot=prod.GetEntries(); 

  //RS
  title.str("");
  name.str("");
  expr.str("");
  name <<"missingtracks_vs_eta_RS"<<intervals[i];
  title <<"(Missing Tracks)/(Total Tracks) "<< ptinterval << " "<< intervals[i-1] << "<#eta<"<<intervals[i];
  expr << "(-"<<sim<<".@obj.size()+"<<rs<<".@obj.size()) >> " << name.str();
  TH1F effRS(name.str().c_str(),title.str().c_str(),3,-1.5,1.5);
  tree->Draw(expr.str().c_str(),cut.str().c_str(),"goff");
  if (tot!=0) effRS.Scale(1/tot);
//   cout <<  eff.GetBinContent(eff.FindBin(0)) << endl;
  efficRS.push_back( effRS.GetBinContent( effRS.FindBin(0) ) );
  effRS.Write();

  //CTF
  name.str("");
  expr.str("");
  name <<"missingtracks_vs_eta_CTF"<<intervals[i];
  expr << "(-"<<sim<<".@obj.size()+"<<ctf<<".@obj.size()) >> " << name.str();
  TH1F effCTF(name.str().c_str(),title.str().c_str(),3,-1.5,1.5);
  tree->Draw(expr.str().c_str(),cut.str().c_str(),"goff");
  if (tot!=0) effCTF.Scale(1/tot);
  efficCTF.push_back( effCTF.GetBinContent( effCTF.FindBin(0) ) );
  effCTF.Write();
}

name.str("ptres_distrib_RS");
title.str("");
expr.str("");
title << "Pt residue " << ptinterval << " " << etainterval;
expr <<sim<<".obj.momentum().perp()-"<<rs<<".obj.pt() >> ptres_distrib_RS";
TH1F ptresRS(name.str().c_str(),title.str().c_str(),100,-1,1);
tree->Draw(expr.str().c_str(),"","goff");

name.str("ptres_distrib_CTF");
title.str("");
expr.str("");
title << "Pt residue "  << ptinterval << " " << etainterval;
expr <<sim<<".obj.momentum().perp()-"<<ctf<<".obj.pt() >> ptres_distrib_CTF";
TH1F ptresCTF(name.str().c_str(),title.str().c_str(),100,-1,1);
tree->Draw(expr.str().c_str(),"","goff");

name.str("chi2_distrib_RS");
title.str("");
expr.str("");
title <<"NChi2 distribution " << ptinterval << " " << etainterval;
expr <<rs<<".obj.chi2()/"<<rs<<".obj.ndof() >> chi2_distrib_RS";
TH1F chi2histoRS(name.str().c_str(),title.str().c_str(),100,0,10);
tree->Draw(expr.str().c_str(),"","goff");

name.str("chi2_distrib_CTF");
title.str("");
expr.str("");
title <<"NChi2 distribution " << ptinterval << " " << etainterval;
expr <<ctf<<".obj.chi2()/"<<ctf<<".obj.ndof() >> chi2_distrib_CTF";
TH1F chi2histoCTF(name.str().c_str(),title.str().c_str(),100,0,10);
tree->Draw(expr.str().c_str(),"","goff");

name.str("etaresidue_distrib_RS");
title.str("");
expr.str("");
title <<"Eta residue "<< ptinterval << " " << etainterval;
expr <<sim<<".obj.momentum().pseudoRapidity()-"<<rs<<".obj.eta() >> etaresidue_distrib_RS";
TH1F etaresRS(name.str().c_str(),title.str().c_str(),100,-3,3);
tree->Draw(expr.str().c_str(),"","goff");

name.str("etaresidue_distrib_CTF");
title.str("");
expr.str("");
title <<"Eta residue "<< ptinterval << " " << etainterval;
expr <<sim<<".obj.momentum().pseudoRapidity()-"<<ctf<<".obj.eta() >> etaresidue_distrib_CTF";
TH1F etaresCTF(name.str().c_str(),title.str().c_str(),100,-3,3);
tree->Draw(expr.str().c_str(),"","goff");

name.str("etagen_distrib");
title.str("");
expr.str("");
title <<"Eta generated "<< ptinterval;
expr <<sim<<".obj.momentum().pseudoRapidity() >> etagen_distrib";
TH1F etagen(name.str().c_str(),title.str().c_str(),100,-3,3);
tree->Draw(expr.str().c_str(),"","goff");

name.str("etafound_distrib_RS");
title.str("");
expr.str("");
title <<"Eta found "<< ptinterval << " " << etainterval;
expr <<rs<<".obj.eta() >> etafound_distrib_RS";
TH1F etafoundRS(name.str().c_str(),title.str().c_str(),100,-3,3);
tree->Draw(expr.str().c_str(),"","goff");

name.str("etafound_distrib_CTF");
title.str("");
expr.str("");
title <<"Eta found "<< ptinterval << " " << etainterval;
expr <<ctf<<".obj.eta() >> etafound_distrib_CTF";
TH1F etafoundCTF(name.str().c_str(),title.str().c_str(),100,-3,3);
tree->Draw(expr.str().c_str(),"","goff");

name.str("hits_distrib_RS");
title.str("");
expr.str("");
title <<"Hits found "<< ptinterval << " " << etainterval;
expr <<rs<<".obj.found() >> hits_distrib_RS";
TH1F hitsRS(name.str().c_str(),title.str().c_str(),100,0,25);
tree->Draw(expr.str().c_str(),"","goff");

name.str("hits2_distrib_RS");
title.str("");
expr.str("");
cut.str("");
title <<"Hits found (>1tracks) "<< ptinterval << " " << etainterval;
expr <<rs<<".obj.found() >> hits2_distrib_RS";
cut <<rs<<".@obj.size()>1";
TH1F hits2RS(name.str().c_str(),title.str().c_str(),100,0,25);
tree->Draw(expr.str().c_str(),cut.str().c_str(),"goff");

name.str("hits3_distrib_RS");
title.str("");
expr.str("");
cut.str("");
title <<"Hits found (1track) "<< ptinterval << " " << etainterval;
expr <<rs<<".obj.found() >> hits3_distrib_RS";
cut <<rs<<".@obj.size()==1";
TH1F hits3RS(name.str().c_str(),title.str().c_str(),100,0,25);
tree->Draw(expr.str().c_str(),cut.str().c_str(),"goff");

name.str("hits_distrib_CTF");
title.str("");
expr.str("");
title <<"Hits found "<< ptinterval << " " << etainterval;
expr <<ctf<<".obj.found() >> hits_distrib_CTF";
TH1F hitsCTF(name.str().c_str(),title.str().c_str(),100,0,25);
tree->Draw(expr.str().c_str(),"","goff");

name.str("hits2_distrib_CTF");
title.str("");
expr.str("");
cut.str("");
title <<"Hits found (>1tracks) "<< ptinterval << " " << etainterval;
expr <<ctf<<".obj.found() >> hits2_distrib_CTF";
cut <<ctf<<".@obj.size()>1";
TH1F hits2CTF(name.str().c_str(),title.str().c_str(),100,0,25);
tree->Draw(expr.str().c_str(),cut.str().c_str(),"goff");

float binfix=0.0001;
name.str("PtRMS_vs_eta_RS");
title.str("");
title <<"PtRMS vs #eta "<< ptinterval;
TH1F ptrmshRS(name.str().c_str(),title.str().c_str(),bins,intervals);
name.str("eff_vs_eta_RS");
title.str("");
title <<"efficiency vs #eta "<< ptinterval;
TH1F effhRS(name.str().c_str(),title.str().c_str(),bins,intervals);
for (int i=1;i<(bins+1);i++){
  ptrmshRS.Fill(intervals[i]-binfix,ptrmsRS[i-1]);
  effhRS.Fill(intervals[i]-binfix,efficRS[i-1]);
}
name.str("PtRMS_vs_eta_CTF");
title.str("");
title <<"PtRMS vs #eta "<< ptinterval;
TH1F ptrmshCTF(name.str().c_str(),title.str().c_str(),bins,intervals);
name.str("eff_vs_eta_CTF");
title.str("");
title <<"efficiency vs #eta "<< ptinterval;
TH1F effhCTF(name.str().c_str(),title.str().c_str(),bins,intervals);
for (int i=1;i<(bins+1);i++){
  ptrmshCTF.Fill(intervals[i]-binfix,ptrmsCTF[i-1]);
  effhCTF.Fill(intervals[i]-binfix,efficCTF[i-1]);
}

ptresRS.Write();
ptresCTF.Write();
chi2histoRS.Write();
chi2histoCTF.Write();
etaresRS.Write();
etaresCTF.Write();
etagen.Write();
etafoundRS.Write();
etafoundCTF.Write();
hitsRS.Write();
hits2RS.Write();
hits3RS.Write();
hitsCTF.Write();
hits2CTF.Write();
ptrmshRS.Write();
effhRS.Write();
ptrmshCTF.Write();
effhCTF.Write();

hits3RS.SetLineColor(2);
hits3RS.Draw();
hits2RS.Draw("SAME");
// file.Close();
// outFile.Close();
}
