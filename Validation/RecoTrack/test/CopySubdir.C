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

void CopySubdir(const char * oldfile, const char * newfile, const char * dirname, const char * type="Track"){

  TFile *oldf = TFile::Open(oldfile);
  bool success=oldf->cd(Form("DQMData/Run 1/RecoTrackV/Run summary/%s",type));
  //  cerr<<success<<endl;
  //cerr<<Form("DQMData/RecoTrackV/%s",type)<<endl;
  if(!success)success=oldf->cd(Form("DQMData/RecoTrackV/%s",type));
  //  cerr<<success<<endl;
  //gDirectory->ls();
  TDirectory *dirold=gDirectory;
  dirold->cd(dirname);
  dirold=gDirectory;
  TFile *newf =new TFile(newfile,"RECREATE");
  TDirectory *dirnew=newf->mkdir("DQMData");
  dirnew=dirnew->mkdir("Run 1");
  dirnew=dirnew->mkdir("Tracking");
  dirnew=dirnew->mkdir("Run summary");
  dirnew=dirnew->mkdir(type);
  gDirectory=dirnew;
  //  dirold->ls();
  CopyDir(dirold);
  
}
