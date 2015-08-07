{
gSystem->Load("libFWCoreFWLite.so");
AutoLibraryLoader::enable();
/////////SET THESE VALUES///////////////
TFile file("TracksParTest2.root");
TTree * tree = (TTree *) gROOT->FindObject("Events");
TFile outFile("graphsParTest2.root","recreate");
float intervals[] = {0,0.5,1.0,1.5,2.0,2.5};
int bins = 5;
const char * ptinterval  = "4.99<P_{T}<5.01";
const char * etainterval = "|#eta|<2.5";
const char * rs  = "recoTracks_rsWithMaterialTracks__RsWithMaterial";
const char * ctf = "recoTracks_ctfWithMaterialTracks__FinalFits";
const char * sim = "SimTracks_SimG4Object__Test";
////////////////////////////////////////
vector<float> tot;
vector<float> ptrmsRS;
vector<float> efficRS;
vector<float> ptrmsCTF;
vector<float> efficCTF;
ostringstream title, name, expr, cut;
float binfix=0.0001;
int i;
//****************SIMULATION***************************************
cout << "Processing SIMULATION" << endl;
//compute the number of generated tracks per eta interval
for (i=1;i<(bins+1);i++){
  name.str("");
  title.str("");
  expr.str("");
  cut.str("");
  cut << "fabs("<<sim<<".obj.momentum().pseudoRapidity())>"<< intervals[i-1] <<"&&fabs("<<sim<<".obj.momentum().pseudoRapidity())<"<<intervals[i];
  name << "producedtracks_vs_eta" << intervals[i];
  expr << ""<<sim<<".@obj.size() >> " << name.str();
  title << name.str();
  TH1F prod(name.str().c_str(),title.str().c_str(),10,0,10);
  tree->Draw(expr.str().c_str(),cut.str().c_str(),"goff");
  tot.push_back(prod.GetEntries()); 
//   prod.Write();
}
//plot number of generated tracks versus eta
name.str("etagen_distrib");
title.str("");
expr.str("");
title <<"Eta generated "<< ptinterval;
expr <<sim<<".obj.momentum().pseudoRapidity() >> etagen_distrib";
TH1F etagen(name.str().c_str(),title.str().c_str(),100,-3,3);
tree->Draw(expr.str().c_str(),"","goff");
etagen.Write();
//*****************************************************************

//****************ROAD SEARCH**************************************
// cout << "Processing ROAD SEARCH" << endl;
// for (i=1;i<(bins+1);i++){
//   //plot Pt residue distribution per eta interval
//   name <<"ptresRS"<<intervals[i];
//   title <<"Pt residue "<< ptinterval << " "<< intervals[i-1] << "<#eta<"<<intervals[i];
//   TH1F ptres2RS(name.str().c_str(),title.str().c_str(),100,-1,1);
//   expr << ""<<sim<<".obj.momentum().perp()-"<<rs<<".obj.pt() >> " << name.str();
//   cut << "fabs("<<sim<<".obj.momentum().pseudoRapidity())>"<< intervals[i-1] <<"&&fabs("<<sim<<".obj.momentum().pseudoRapidity())<"<<intervals[i];
//   tree->Draw(expr.str().c_str(),cut.str().c_str(),"goff");
//   ptrmsRS.push_back(ptres2RS.GetRMS());
//   ptres2RS.Write();

//   //plot the number of missing tracks per eta interval
//   //and computes the efficiency in that eta interval
//   title.str("");
//   name.str("");
//   expr.str("");
//   name <<"missingtracks_vs_eta_RS"<<intervals[i];
//   title <<"(Missing Tracks)/(Total Tracks) "<< ptinterval << " "<< intervals[i-1] << "<#eta<"<<intervals[i];
//   expr << "(-"<<sim<<".@obj.size()+"<<rs<<".@obj.size()) >> " << name.str();
//   TH1F effRS(name.str().c_str(),title.str().c_str(),3,-1.5,1.5);
//   tree->Draw(expr.str().c_str(),cut.str().c_str(),"goff");
//   if (tot[i-1]!=0) effRS.Scale(1/tot[i-1]);
// //   cout <<  eff.GetBinContent(eff.FindBin(0)) << endl;
//   efficRS.push_back( effRS.GetBinContent( effRS.FindBin(0) ) );
//   effRS.Write();
// }

// //plot global Pt residue distribution
// name.str("ptres_distrib_RS");
// title.str("");
// expr.str("");
// title << "Pt residue " << ptinterval << " " << etainterval;
// expr <<sim<<".obj.momentum().perp()-"<<rs<<".obj.pt() >> ptres_distrib_RS";
// TH1F ptresRS(name.str().c_str(),title.str().c_str(),100,-1,1);
// tree->Draw(expr.str().c_str(),"","goff");
// ptresRS.Write();

// //plot global normalized chi2 distribution
// name.str("chi2_distrib_RS");
// title.str("");
// expr.str("");
// title <<"NChi2 distribution " << ptinterval << " " << etainterval;
// expr <<rs<<".obj.chi2()/"<<rs<<".obj.ndof() >> chi2_distrib_RS";
// TH1F chi2histoRS(name.str().c_str(),title.str().c_str(),100,0,10);
// tree->Draw(expr.str().c_str(),"","goff");
// chi2histoRS.Write();

// //plot global eta residue distribution
// name.str("etaresidue_distrib_RS");
// title.str("");
// expr.str("");
// title <<"Eta residue "<< ptinterval << " " << etainterval;
// expr <<sim<<".obj.momentum().pseudoRapidity()-"<<rs<<".obj.eta() >> etaresidue_distrib_RS";
// TH1F etaresRS(name.str().c_str(),title.str().c_str(),100,-3,3);
// tree->Draw(expr.str().c_str(),"","goff");
// etaresRS.Write();

// //plot global eta distribution of reconstructed tracks
// name.str("etafound_distrib_RS");
// title.str("");
// expr.str("");
// title <<"Eta found "<< ptinterval << " " << etainterval;
// expr <<rs<<".obj.eta() >> etafound_distrib_RS";
// TH1F etafoundRS(name.str().c_str(),title.str().c_str(),100,-3,3);
// tree->Draw(expr.str().c_str(),"","goff");
// etafoundRS.Write();

// //plot number of hits distribution of reconstructed tracks
// name.str("hits_distrib_RS");
// title.str("");
// expr.str("");
// title <<"Hits found "<< ptinterval << " " << etainterval;
// expr <<rs<<".obj.numberOfValidHits() >> hits_distrib_RS";
// TH1F hitsRS(name.str().c_str(),title.str().c_str(),100,0,25);
// tree->Draw(expr.str().c_str(),"","goff");
// hitsRS.Write();

// //plot number of hits distribution of reconstructed tracks
// //when the number of reconstructed tracks is >1
// name.str("hits2_distrib_RS");
// title.str("");
// expr.str("");
// cut.str("");
// title <<"Hits found (>1tracks) "<< ptinterval << " " << etainterval;
// expr <<rs<<".obj.numberOfValidHits() >> hits2_distrib_RS";
// cut <<rs<<".@obj.size()>1";
// TH1F hits2RS(name.str().c_str(),title.str().c_str(),100,0,25);
// tree->Draw(expr.str().c_str(),cut.str().c_str(),"goff");
// hits2RS.Write();

// //plot number of hits distribution of reconstructed tracks
// //when the number of reconstructed tracks is =1
// name.str("hits3_distrib_RS");
// title.str("");
// expr.str("");
// cut.str("");
// title <<"Hits found (1track) "<< ptinterval << " " << etainterval;
// expr <<rs<<".obj.numberOfValidHits() >> hits3_distrib_RS";
// cut <<rs<<".@obj.size()==1";
// TH1F hits3RS(name.str().c_str(),title.str().c_str(),100,0,25);
// tree->Draw(expr.str().c_str(),cut.str().c_str(),"goff");
// hits3RS.Write();

// //plot the Pt RMS per eta interval
// //plot the efficiency per eta interval
// name.str("PtRMS_vs_eta_RS");
// title.str("");
// title <<"PtRMS vs #eta "<< ptinterval;
// TH1F ptrmshRS(name.str().c_str(),title.str().c_str(),bins,intervals);
// name.str("eff_vs_eta_RS");
// title.str("");
// title <<"efficiency vs #eta "<< ptinterval;
// TH1F effhRS(name.str().c_str(),title.str().c_str(),bins,intervals);
// for (int i=1;i<(bins+1);i++){
//   ptrmshRS.Fill(intervals[i]-binfix,ptrmsRS[i-1]);
//   effhRS.Fill(intervals[i]-binfix,efficRS[i-1]);
// }
// effhRS.Write();
// ptrmshRS.Write();
//*****************************************************************


//****************COMBINATORIAL TRACK FINDER***********************
cout << "Processing COMBINATORIAL TRACK FINDER" << endl;
for (i=1;i<(bins+1);i++){
  //plot Pt residue distribution per eta interval
  name.str("");
  expr.str("");
  title.str("");
  cut.str("");
  name <<"ptresCTF"<<intervals[i];
  title <<"Pt residue "<< ptinterval << " "<< intervals[i-1] << "<#eta<"<<intervals[i];
  TH1F ptres2CTF(name.str().c_str(),title.str().c_str(),100,-1,1);
  expr << ""<<sim<<".obj.momentum().perp()-"<<ctf<<".obj.pt() >> " << name.str();
  cut << "fabs("<<sim<<".obj.momentum().pseudoRapidity())>"<< intervals[i-1] <<"&&fabs("<<sim<<".obj.momentum().pseudoRapidity())<"<<intervals[i];
tree->Draw(expr.str().c_str(),cut.str().c_str(),"goff");
  ptrmsCTF.push_back(ptres2CTF.GetRMS());
  ptres2CTF.Write();

  //plot the number of missing tracks per eta interval
  //and computes the efficiency in that eta interval
  name.str("");
  expr.str("");
  name <<"missingtracks_vs_eta_CTF"<<intervals[i];
  expr << "(-"<<sim<<".@obj.size()+"<<ctf<<".@obj.size()) >> " << name.str();
  TH1F effCTF(name.str().c_str(),title.str().c_str(),3,-1.5,1.5);
  tree->Draw(expr.str().c_str(),cut.str().c_str(),"goff");
  if (tot[i-1]!=0) effCTF.Scale(1/tot[i-1]);
  efficCTF.push_back( effCTF.GetBinContent( effCTF.FindBin(0) ) );
  effCTF.Write();
}

//plot global Pt residue distribution
name.str("ptres_distrib_CTF");
title.str("");
expr.str("");
title << "Pt residue "  << ptinterval << " " << etainterval;
expr <<sim<<".obj.momentum().perp()-"<<ctf<<".obj.pt() >> ptres_distrib_CTF";
TH1F ptresCTF(name.str().c_str(),title.str().c_str(),100,-1,1);
tree->Draw(expr.str().c_str(),"","goff");
ptresCTF.Write();


//plot global normalized chi2 distribution
name.str("chi2_distrib_CTF");
title.str("");
expr.str("");
title <<"NChi2 distribution " << ptinterval << " " << etainterval;
expr <<ctf<<".obj.chi2()/"<<ctf<<".obj.ndof() >> chi2_distrib_CTF";
TH1F chi2histoCTF(name.str().c_str(),title.str().c_str(),100,0,10);
tree->Draw(expr.str().c_str(),"","goff");
chi2histoCTF.Write();

//plot global eta residue distribution
name.str("etaresidue_distrib_CTF");
title.str("");
expr.str("");
title <<"Eta residue "<< ptinterval << " " << etainterval;
expr <<sim<<".obj.momentum().pseudoRapidity()-"<<ctf<<".obj.eta() >> etaresidue_distrib_CTF";
TH1F etaresCTF(name.str().c_str(),title.str().c_str(),100,-3,3);
tree->Draw(expr.str().c_str(),"","goff");
etaresCTF.Write();


//plot global eta distribution of reconstructed tracks
name.str("etafound_distrib_CTF");
title.str("");
expr.str("");
title <<"Eta found "<< ptinterval << " " << etainterval;
expr <<ctf<<".obj.eta() >> etafound_distrib_CTF";
TH1F etafoundCTF(name.str().c_str(),title.str().c_str(),100,-3,3);
tree->Draw(expr.str().c_str(),"","goff");
etafoundCTF.Write();

//plot number of hits distribution of reconstructed tracks
name.str("hits_distrib_CTF");
title.str("");
expr.str("");
title <<"Hits found "<< ptinterval << " " << etainterval;
expr <<ctf<<".obj.numberOfValidHits() >> hits_distrib_CTF";
TH1F hitsCTF(name.str().c_str(),title.str().c_str(),100,0,25);
tree->Draw(expr.str().c_str(),"","goff");
hitsCTF.Write();

//plot number of hits distribution of reconstructed tracks
//when the number of reconstructed tracks is >1
name.str("hits2_distrib_CTF");
title.str("");
expr.str("");
cut.str("");
title <<"Hits found (>1tracks) "<< ptinterval << " " << etainterval;
expr <<ctf<<".obj.numberOfValidHits() >> hits2_distrib_CTF";
cut <<ctf<<".@obj.size()>1";
TH1F hits2CTF(name.str().c_str(),title.str().c_str(),100,0,25);
tree->Draw(expr.str().c_str(),cut.str().c_str(),"goff");
hits2CTF.Write();

//plot the Pt RMS per eta interval
//plot the efficiency per eta interval
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
ptrmshCTF.Write();
effhCTF.Write();
//*****************************************************************


//hits3RS.SetLineColor(2);
//hits3RS.Draw();
//hits2RS.Draw("SAME");
file.Close();
outFile.Close();
}
