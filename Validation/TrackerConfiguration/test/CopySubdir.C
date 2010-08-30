void CopyDir(TDirectory *source) {
   //copy all objects and subdirs of directory source as a subdir of the current directory   
   source->ls();
   TDirectory *savdir = gDirectory;
   TDirectory *adir = savdir->mkdir(source->GetName());
   adir->cd();
   //loop on all entries of this directory
   TKey *key;
   TIter nextkey(source->GetListOfKeys());
   while ((key = (TKey*)nextkey())) {
      const char *classname = key->GetClassName();
      TClass *cl = gROOT->GetClass(classname);
      if (!cl) continue;
      if (cl->InheritsFrom("TDirectory")) {
         source->cd(key->GetName());
         TDirectory *subdir = gDirectory;
         adir->cd();
         CopyDir(subdir);
         adir->cd();
      } else if (cl->InheritsFrom("TTree")) {
         TTree *T = (TTree*)source->Get(key->GetName());
         adir->cd();
         TTree *newT = T->CloneTree(-1,"fast");
         newT->Write();
      } else {
         source->cd();
         TObject *obj = key->ReadObj();
         adir->cd();
         obj->Write();
         delete obj;
     }
  }
  adir->SaveSelf(kTRUE);
  savdir->cd();
}

void CopySubdir(const char * oldfile, const char * newfile){

  TFile *oldf = TFile::Open(oldfile);
  oldf->cd("DQMData/Run 1/Tracking");
  TDirectory *dirtracking=gDirectory;
  oldf->cd("DQMData/Run 1/TrackerHitsV");
  TDirectory *dirsimhit=gDirectory;
  oldf->cd("DQMData/Run 1/TrackerRecHitsV");
  TDirectory *dirrechits=gDirectory;
  oldf->cd("DQMData/Run 1/TrackerDigisV");
  TDirectory *dirdigis=gDirectory;
  oldf->cd("DQMData/Run 1/Tracking");
  TDirectory *dirTP=gDirectory;
  TFile *newf =new TFile(newfile,"RECREATE");
  TDirectory *dirnew=newf->mkdir("DQMData");
  dirnew=dirnew->mkdir("Run 1");
  dirnew->cd();
  CopyDir(dirtracking);
  CopyDir(dirsimhit);
  CopyDir(dirrechits);
  CopyDir(dirdigis);
  CopyDir(dirTP);
  TList* new_list = oldf->GetListOfKeys() ;
  newf->cd();
  TIter     newkey_iter( new_list) ;
  TKey*     new_key ;
  TObject*  new_obj ;
  while ( new_key = (TKey*) newkey_iter() ) {
    new_obj = new_key->ReadObj() ;
    if (strcmp(new_obj->IsA()->GetName(),"TObjString")==0) {
      TObjString * cversion = (TObjString*) new_obj;
      if(cversion->GetString().Contains("CMSSW")){
	cversion->Write();
	break;
      }
    }
  }
}
