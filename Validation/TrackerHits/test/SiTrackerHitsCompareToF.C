void SiTrackerHitsCompareToF()
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
 Char_t profileo[200];
 Char_t profilen[200];
 std::strstream buf;
 std::string value;
 
   TCanvas * ToF = new TCanvas("ToF","ToF",600,800);
   ToF->Divide(2,2);

   TProfile * ch1e[4];
   TProfile * rh1e[4];
   cout<<cdir->GetPath()<<endl;
   cout<<rdir->GetPath()<<endl;
     sprintf(histo,"tof_eta");
     sprintf(profileo,"tof_eta_old");
     sprintf(profilen,"tof_eta_new");
     rh1e[0] = ((TH2F*)rdir->Get(histo))->ProfileX(profileo);
     ch1e[0] = ((TH2F*)cdir->Get(histo))->ProfileX(profilen);
     sprintf(histo,"tof_phi");
     sprintf(profileo,"tof_phi_old");
     sprintf(profilen,"tof_phi_new");
     rh1e[1] = ((TH2F*)rdir->Get(histo))->ProfileX(profileo);
     ch1e[1] = ((TH2F*)cdir->Get(histo))->ProfileX(profilen);
     sprintf(histo,"tof_r");
     sprintf(profileo,"tof_r_old");
     sprintf(profilen,"tof_r_new");
     rh1e[2] = ((TH2F*)rdir->Get(histo))->ProfileX(profileo);
     ch1e[2] = ((TH2F*)cdir->Get(histo))->ProfileX(profilen);
     sprintf(histo,"tof_z");
     sprintf(profileo,"tof_z_old");
     sprintf(profilen,"tof_z_new");
     rh1e[3] = ((TH2F*)rdir->Get(histo))->ProfileX(profileo);
     ch1e[3] = ((TH2F*)cdir->Get(histo))->ProfileX(profilen);
     
     for (Int_t i=0; i<4; i++) {      
       ToF->cd(i+1);
       if (PV->KSok(rh1e[i] , ch1e[i])) {
       ks1e[i] = PV->KSCompute(rh1e[i] , ch1e[i] , te );
       PV->KSdraw(rh1e[i] , ch1e[i]);
       rh1e[i]->Draw("h");      
       ch1e[i]->Draw("h same");             
       buf<<"KS="<<ks1e[i]<<std::endl;
       buf>>value;
       te->DrawTextNDC(0.5,0.7, value.c_str());
       leg.Clear();
       leg.AddEntry(rh1e[i],rver , "l");
       leg.AddEntry(ch1e[i],cver , "l");
       leg.Draw();


     }
//     std::cout << " i =" << i << " KS = " << ks1e[i] << std::endl; 
     if (ks1e[i] < 0.1) outfile << ch1e[i]->GetName() <<" KS probability = "<< ks1e[i] <<" "<<endl;
   }

 
 ToF->Print("Tof.eps");
 ToF->Print("Tof.gif");

}
