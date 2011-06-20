#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TH1F.h"
#include <iostream>
#include <fstream>

 gROOT ->Reset(); 

 string statp = "KS prob";
 Double_t ks[12];
  
 Char_t histo[200];
 std::strstream buf;
 std::string value;
 
 TH1F * ch[12];
 TH1F * rh[12];
 TText* te = new TText();
 
 TFile * rfile;   
 TFile * cfile;
 
 std::string system;
 std::string variable;
 std::strstream hn;
 std::string hname;
 std::string histogram;
 ofstream outfile("LowKS_pos_list.dat");

void SiTrackerHitsComparePosition()
 {
 
 gROOT->ProcessLine(".x HistoCompare.C");
 HistoCompare * PV = new HistoCompare();

 gStyle->SetNdivisions(504,"XYZ");
 gStyle->SetStatH(0.18);
 gStyle->SetStatW(0.35);
 
 char*  cfilename = "TrackerHitHisto.root"; //current
 char*  rfilename = "../TrackerHitHisto.root";  //reference

 delete gROOT->GetListOfFiles()->FindObject(rfilename);
 delete gROOT->GetListOfFiles()->FindObject(cfilename); 

 rfile = new TFile(rfilename);
 TDirectory * rdir=gDirectory; 
 cfile = new TFile(cfilename);
 TDirectory * cdir=gDirectory; 

 if(rfile->cd("DQMData/Run 1/TrackerHitsV"))rfile->cd("DQMData/Run 1/TrackerHitsV/Run summary/TrackerHit");
 else rfile->cd("DQMData/TrackerHitsV/TrackerHit");
 rdir=gDirectory;

 if(cfile->cd("DQMData/Run 1/TrackerHitsV"))cfile->cd("DQMData/Run 1/TrackerHitsV/Run summary/TrackerHit");
 else cfile->cd("DQMData/TrackerHitsV/TrackerHit");
 cdir=gDirectory; 

 
  TLegend leg(0.3, 0.83, 0.55, 0.90);
 //Get list of Keys from the Reference file.
  TList* ref_list = rfile->GetListOfKeys() ;
  if (!ref_list) {
      std::cout<<"=========>> AutoComaprison:: There is no Keys available in the Reference file."<<std::endl;
      exit(1) ;
   }

  //Get list of Keys from the New file.
  TList* new_list = cfile->GetListOfKeys() ;
  if (!new_list) {
      std::cout<<"=========>> AutoComaprison:: There is no Keys available in New file."<<std::endl;
      exit(1) ;
   }


  //Iterate on the List of Keys of the  Reference file.
  TIter     refkey_iter( ref_list) ;
  TKey*     ref_key ;
  TObject*  ref_obj ;

  char rver[50];
  char cver[50];
  while ( ref_key = (TKey*) refkey_iter() ) {
      ref_obj = ref_key->ReadObj() ;
      if (strcmp(ref_obj->IsA()->GetName(),"TObjString")==0) {

         TObjString * rversion = dynamic_cast< TObjString*> (ref_obj);
         sprintf(rver, "%s", rversion->GetName());
         std::cout<<" Ref. version =" << rver<<std::endl;
	 break;
      }
  }

  //Iterate on the List of Keys of the  Reference file.
  TIter     newkey_iter( new_list) ;
  TKey*     new_key ;
  TObject*  new_obj ;
  while ( new_key = (TKey*) newkey_iter() ) {
      new_obj = new_key->ReadObj() ;
      if (strcmp(new_obj->IsA()->GetName(),"TObjString")==0) {

         TObjString * cversion = dynamic_cast< TObjString*> (new_obj);
         sprintf(cver, "%s", cversion->GetName());
         std::cout<<" Cur version =" << cver<<std::endl;
	 break;

      }
  }



 
 TH1F * hsum_st = new TH1F("ks_st", "KS summary position STRIPS", 22 , -0.05 , 1.05);
 TH1F * hsum_px = new TH1F("ks_px", "KS summary position PIXELS", 22 , -0.05 , 1.05);
 TH1F * hsum_TIB = new TH1F("ks_TIB", "KS summary position TIB", 22 , -0.05 , 1.05);
 TH1F * hsum_TOB = new TH1F("ks_TOB", "KS summary position TOB", 22 , -0.05 , 1.05);
 TH1F * hsum_TID = new TH1F("ks_TID", "KS summary position TID", 22 , -0.05 , 1.05);
 TH1F * hsum_TEC = new TH1F("ks_TEC", "KS summary position TEC", 22 , -0.05 , 1.05);
 TH1F * hsum_BPIX = new TH1F("ks_BPIX", "KS summary position BPIX", 22 , -0.05 , 1.05);
 TH1F * hsum_FPIX = new TH1F("ks_FPIX", "KS summary position FPIX", 22 , -0.05 , 1.05);

//=======================================================
   variable = "Entryx-Exitx";
//=======================================================

// TOB
   system = "TOB";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_TOB->Fill(ks[i]);

       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());
   
// TIB
   system = "TIB";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_TIB->Fill(ks[i]);

       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());
   
// TID
   system = "TID";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_TID->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());

// TEC
   system = "TEC";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_TEC->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());

// BPIX
   system = "BPIX";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_BPIX->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();
     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());
   
   system = "FPIX";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_FPIX->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());

//=======================================================
   variable = "Entryy-Exity";
//=======================================================

// TOB
   system = "TOB";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_TOB->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());
   
// TIB
   system = "TIB";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_TIB->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());
   
// TID
   system = "TID";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_TID->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());

// TEC
   system = "TEC";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_TEC->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());

// BPIX
   system = "BPIX";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_BPIX->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());
   
   system = "FPIX";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]); 
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_FPIX->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());

//=======================================================
   variable = "Entryz-Exitz";
//=======================================================

// TOB
   system = "TOB";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_TOB->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());
   
// TIB
   system = "TIB";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_TIB->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());
   
// TID
   system = "TID";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_TID->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());

// TEC
   system = "TEC";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_TEC->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());

// BPIX
   system = "BPIX";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_BPIX->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());
   
   system = "FPIX";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_FPIX->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());

//=======================================================
   variable = "Localx";
//=======================================================

// TOB
   system = "TOB";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_TOB->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());
   
// TIB
   system = "TIB";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_TIB->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());
   
// TID
   system = "TID";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_TID->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());

// TEC
   system = "TEC";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_TEC->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());

// BPIX
   system = "BPIX";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_BPIX->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());
   
   system = "FPIX";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_FPIX->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());

//=======================================================
   variable = "Localy";
//=======================================================

// TOB
   system = "TOB";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_TOB->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());
   
// TIB
   system = "TIB";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_TIB->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());
   
// TID
   system = "TID";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_TID->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());

// TEC
   system = "TEC";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_TEC->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());

// BPIX
   system = "BPIX";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_BPIX->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());
   
   system = "FPIX";
   string outname_eps = "pos_"+variable+"_"+system+"_KS.eps";
   string outname_gif = "pos_"+variable+"_"+system+"_KS.gif";
   histogram = ""+system+"Hit/"+variable+"_"+system+"_";
   
   TCanvas * c = new TCanvas(system.c_str(),system.c_str(),600,800);
   c->Divide(3,4);   
   for (Int_t i=0; i<12; i++) {       
     hn << histogram << i+1 <<std::endl; 
     hn >> hname;
     rh[i] = (TH1F*)rdir->Get(hname.c_str())->Clone();
     ch[i] = (TH1F*)cdir->Get(hname.c_str())->Clone();      
     c->cd(i+1);
     if (PV->KSok(rh[i] , ch[i])) {
       ks[i] = PV->KSCompute(rh[i] , ch[i] , te );
       PV->KSdraw(rh[i] , ch[i]);
       rh[i]->Draw("h");      
       ch[i]->Draw("h same");             
       buf<<"KS="<<ks[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_FPIX->Fill(ks[i]);
       leg.Clear();
       leg.AddEntry(rh[i],rver , "l");
       leg.AddEntry(ch[i],cver , "l");
       leg.Draw();

     }
     if (ks[i] < 0.1) outfile << ch[i]->GetName() <<" KS probability = "<< ks[i] <<" "<<endl;
   } 
    c->Print(outname_eps.c_str());
    c->Print(outname_gif.c_str());

//===================================================
// Summary plots
//===================================================

 hsum_st -> Add (hsum_TIB);
 hsum_st -> Add (hsum_TOB);
 hsum_st -> Add (hsum_TID);
 hsum_st -> Add (hsum_TEC); 
 hsum_px -> Add (hsum_BPIX);
 hsum_px -> Add (hsum_FPIX);
 
 TCanvas * s = new TCanvas("s","s",600,800);
 s->Divide(2,4);
 
 s->cd (1);
 hsum_TIB -> Draw();
 s->cd (2);
 hsum_TOB -> Draw();
 s->cd (3);
 hsum_TID -> Draw();
 s->cd (4);
 hsum_TEC -> Draw();
 s->cd (5);
 hsum_BPIX -> Draw();
 s->cd (6);
 hsum_FPIX -> Draw();
 s->cd (7);
 hsum_st -> Draw();
 s->cd (8);
 hsum_px -> Draw();
 
 s->Print("pos_summary_KS.eps");  
 s->Print("pos_summary_KS.gif");  

}


