void SiTrackerHitsCompareEnergy()
{

 gROOT ->Reset();
 gStyle->SetNdivisions(504,"XYZ");
 gStyle->SetStatH(0.18);
 gStyle->SetStatW(0.35);
 
 char*  cfilename = "TrackerHitHisto.root"; //current
 char*  rfilename = "../TrackerHitHisto.root";  //reference

 delete gROOT->GetListOfFiles()->FindObject(rfilename);
 delete gROOT->GetListOfFiles()->FindObject(cfilename); 

 TText* te = new TText();
 TFile * rfile = new TFile(rfilename);
 TDirectory * rdir=gDirectory; 
 TFile * cfile = new TFile(cfilename);
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


 ofstream outfile("LowKS_energy_list.dat");

 string statp = "KS prob";
 Double_t ks1e[12],ks2e[12],ks3e[12],ks4e[12],ks5e[12],ks6e[12];
 
 gROOT->ProcessLine(".x HistoCompare.C");
 HistoCompare * PV = new HistoCompare();
 
 Char_t histo[200];
 std::strstream buf;
 std::string value;
 
 TH1F * hsum_st = new TH1F("ks_st", "KS summary eloss STRIPS", 22 , -0.05 , 1.05);
 TH1F * hsum_px = new TH1F("ks_px", "KS summary eloss PIXELS", 22 , -0.05 , 1.05);
 TH1F * hsum_TIB = new TH1F("ks_TIB", "KS summary eloss TIB", 22 , -0.05 , 1.05);
 TH1F * hsum_TOB = new TH1F("ks_TOB", "KS summary eloss TOB", 22 , -0.05 , 1.05);
 TH1F * hsum_TID = new TH1F("ks_TID", "KS summary eloss TID", 22 , -0.05 , 1.05);
 TH1F * hsum_TEC = new TH1F("ks_TEC", "KS summary eloss TEC", 22 , -0.05 , 1.05);
 TH1F * hsum_BPIX = new TH1F("ks_BPIX", "KS summary eloss BPIX", 22 , -0.05 , 1.05);
 TH1F * hsum_FPIX = new TH1F("ks_FPIX", "KS summary eloss FPIX", 22 , -0.05 , 1.05);


// TIB
   TCanvas * TIB = new TCanvas("TIB","TIB",600,800);
   TIB->Divide(3,4);

   TH1F * ch1e[12];
   TH1F * rh1e[12];
   
   for (Int_t i=0; i<12; i++) {        
     sprintf(histo,"TIBHit/Eloss_TIB_%i",i+1);
     rh1e[i] = (TH1F*)rdir->Get(histo)->Clone();
     ch1e[i] = (TH1F*)cdir->Get(histo)->Clone();
      
     TIB->cd(i+1);
     if (PV->KSok(rh1e[i] , ch1e[i])) {
       ks1e[i] = PV->KSCompute(rh1e[i] , ch1e[i] , te );
       PV->KSdraw(rh1e[i] , ch1e[i]);
       rh1e[i]->Draw("h");      
       ch1e[i]->Draw("h same");             
       buf<<"KS="<<ks1e[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_TIB->Fill(ks1e[i]);
       leg.Clear();
       leg.AddEntry(rh1e[i],rver , "l");
       leg.AddEntry(ch1e[i],cver , "l");
       leg.Draw();


     }
//     std::cout << " i =" << i << " KS = " << ks1e[i] << std::endl; 
     if (ks1e[i] < 0.1) outfile << ch1e[i]->GetName() <<" KS probability = "<< ks1e[i] <<" "<<endl;
   }

// TOB
   TCanvas * TOB = new TCanvas("TOB","TOB",600,800);
   TOB->Divide(3,4);
   
   for (Int_t i=0; i<12; i++) {        
     sprintf(histo,"TOBHit/Eloss_TOB_%i",i+1);
     rh1e[i] = (TH1F*)rdir->Get(histo)->Clone();
     ch1e[i] = (TH1F*)cdir->Get(histo)->Clone();
      
     TOB->cd(i+1);
     if (PV->KSok(rh1e[i] , ch1e[i])) {
       ks1e[i] = PV->KSCompute(rh1e[i] , ch1e[i] , te );
       PV->KSdraw(rh1e[i] , ch1e[i]);
       rh1e[i]->Draw("h");      
       ch1e[i]->Draw("h same");             
       buf<<"KS="<<ks1e[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_TOB->Fill(ks1e[i]);
       leg.Clear();
       leg.AddEntry(rh1e[i],rver , "l");
       leg.AddEntry(ch1e[i],cver , "l");
       leg.Draw();

     }
     if (ks1e[i] < 0.1) outfile << ch1e[i]->GetName() <<" KS probability = "<< ks1e[i] <<" "<<endl;
   }

// TID
   TCanvas * TID = new TCanvas("TID","TID",600,800);
   TID->Divide(3,4);
   
   for (Int_t i=0; i<12; i++) {        
     sprintf(histo,"TIDHit/Eloss_TID_%i",i+1);
     rh1e[i] = (TH1F*)rdir->Get(histo)->Clone();
     ch1e[i] = (TH1F*)cdir->Get(histo)->Clone();
      
     TID->cd(i+1);
     if (PV->KSok(rh1e[i] , ch1e[i])) {
       ks1e[i] = PV->KSCompute(rh1e[i] , ch1e[i] , te );
       PV->KSdraw(rh1e[i] , ch1e[i]);
       rh1e[i]->Draw("h");      
       ch1e[i]->Draw("h same");             
       buf<<"KS="<<ks1e[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_TID->Fill(ks1e[i]);
       leg.Clear();
       leg.AddEntry(rh1e[i],rver , "l");
       leg.AddEntry(ch1e[i],cver , "l");
       leg.Draw();

     }
     if (ks1e[i] < 0.1) outfile << ch1e[i]->GetName() <<" KS probability = "<< ks1e[i] <<" "<<endl;
   }

// TEC
   TCanvas * TEC = new TCanvas("TEC","TEC",600,800);
   TEC->Divide(3,4);
   
   for (Int_t i=0; i<12; i++) {        
     sprintf(histo,"TECHit/Eloss_TEC_%i",i+1);
     rh1e[i] = (TH1F*)rdir->Get(histo)->Clone();
     ch1e[i] = (TH1F*)cdir->Get(histo)->Clone();
      
     TEC->cd(i+1);
     if (PV->KSok(rh1e[i] , ch1e[i])) {
       ks1e[i] = PV->KSCompute(rh1e[i] , ch1e[i] , te );
       PV->KSdraw(rh1e[i] , ch1e[i]);
       rh1e[i]->Draw("h");      
       ch1e[i]->Draw("h same");             
       buf<<"KS="<<ks1e[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_TEC->Fill(ks1e[i]);
       leg.Clear();
       leg.AddEntry(rh1e[i],rver , "l");
       leg.AddEntry(ch1e[i],cver , "l");
       leg.Draw();

     }
     if (ks1e[i] < 0.1) outfile << ch1e[i]->GetName() <<" KS probability = "<< ks1e[i] <<" "<<endl;
   }

// BPIX
   TCanvas * BPIX = new TCanvas("BPIX","BPIX",600,800);
   BPIX->Divide(3,4);
   
   for (Int_t i=0; i<12; i++) {        
     sprintf(histo,"BPIXHit/Eloss_BPIX_%i",i+1);
     rh1e[i] = (TH1F*)rdir->Get(histo)->Clone();
     ch1e[i] = (TH1F*)cdir->Get(histo)->Clone();
      
     BPIX->cd(i+1);
     if (PV->KSok(rh1e[i] , ch1e[i])) {
       ks1e[i] = PV->KSCompute(rh1e[i] , ch1e[i] , te );
       PV->KSdraw(rh1e[i] , ch1e[i]);
       rh1e[i]->Draw("h");      
       ch1e[i]->Draw("h same");             
       buf<<"KS="<<ks1e[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_BPIX->Fill(ks1e[i]);
       leg.Clear();
       leg.AddEntry(rh1e[i],rver , "l");
       leg.AddEntry(ch1e[i],cver , "l");
       leg.Draw();

     }
     if (ks1e[i] < 0.1) outfile << ch1e[i]->GetName() <<" KS probability = "<< ks1e[i] <<" "<<endl;
   }

// FPIX
   TCanvas * FPIX = new TCanvas("FPIX","FPIX",600,800);
   FPIX->Divide(3,4);
   
   for (Int_t i=0; i<12; i++) {        
     sprintf(histo,"FPIXHit/Eloss_FPIX_%i",i+1);
     rh1e[i] = (TH1F*)rdir->Get(histo)->Clone();
     ch1e[i] = (TH1F*)cdir->Get(histo)->Clone();
      
     FPIX->cd(i+1);
     if (PV->KSok(rh1e[i] , ch1e[i])) {
       ks1e[i] = PV->KSCompute(rh1e[i] , ch1e[i] , te );
       PV->KSdraw(rh1e[i] , ch1e[i]);
       rh1e[i]->Draw("h");      
       ch1e[i]->Draw("h same");             
       buf<<"KS="<<ks1e[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       hsum_FPIX->Fill(ks1e[i]);
       leg.Clear();
       leg.AddEntry(rh1e[i],rver , "l");
       leg.AddEntry(ch1e[i],cver , "l");
       leg.Draw();

     }
     if (ks1e[i] < 0.1) outfile << ch1e[i]->GetName() <<" KS probability = "<< ks1e[i] <<" "<<endl;
   }

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
 
 TIB->Print("eloss_TIB_KS.eps");
 TOB->Print("eloss_TOB_KS.eps");
 TID->Print("eloss_TID_KS.eps");
 TEC->Print("eloss_TEC_KS.eps");
 BPIX->Print("eloss_BPIX_KS.eps");
 FPIX->Print("eloss_FPIX_KS.eps");
 s->Print("eloss_summary_KS.eps");  

 TIB->Print("eloss_TIB_KS.gif");
 TOB->Print("eloss_TOB_KS.gif");
 TID->Print("eloss_TID_KS.gif");
 TEC->Print("eloss_TEC_KS.gif");
 BPIX->Print("eloss_BPIX_KS.gif");
 FPIX->Print("eloss_FPIX_KS.gif");
 s->Print("eloss_summary_KS.gif");  

}
