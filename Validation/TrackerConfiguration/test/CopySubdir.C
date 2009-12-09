void CopyDir(TDirectory *source) {
   //copy all objects and subdirs of directory source as a subdir of the current directory   
  //   source->ls();
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

  TDirectory *dirtracking; bool ok_tracking=false;
  TDirectory *dirsimhit; bool ok_simhit=false;
  TDirectory *dirrechits; bool ok_rechits=false;
  TDirectory *dirdigis; bool ok_digis=false;
  TDirectory *dirTP; bool ok_TP=false;
  TDirectory *dirtrackingrechits; bool ok_trackingrechits=false;

  TFile *oldf = TFile::Open(oldfile);
  if (oldf->cd("DQMData/Run 1/Tracking")) {
    oldf->cd("DQMData/Run 1/Tracking");
    dirtracking=gDirectory;
    ok_tracking=true;
  }
  if (oldf->cd("DQMData/Run 1/TrackerHitsV")) {
    oldf->cd("DQMData/Run 1/TrackerHitsV");
    dirsimhit=gDirectory;
    ok_simhit=true;
  }
  if (oldf->cd("DQMData/Run 1/TrackerRecHitsV")) {
    oldf->cd("DQMData/Run 1/TrackerRecHitsV");
    dirrechits=gDirectory;
    ok_rechits=true;
  }
  if (oldf->cd("DQMData/Run 1/TrackerDigisV")) {
    oldf->cd("DQMData/Run 1/TrackerDigisV");
    dirdigis=gDirectory;
    ok_digis=true;
  }
  if (oldf->cd("DQMData/Run 1/TrackingMCTruthV")) {
    oldf->cd("DQMData/Run 1/TrackingMCTruthV");
    dirTP=gDirectory;
    ok_TP=true;
  }
  if (oldf->cd("DQMData/Run 1/RecoTrackV")) {
    oldf->cd("DQMData/Run 1/RecoTrackV");
    dirtrackingrechits=gDirectory;
    ok_trackingrechits=true;
  }

  TFile *newf =new TFile(newfile,"RECREATE");
  TDirectory *dirnew=newf->mkdir("DQMData");
  dirnew=dirnew->mkdir("Run 1");
  dirnew->cd();
  if (ok_tracking) CopyDir(dirtracking);
  if (ok_simhit) CopyDir(dirsimhit);
  if (ok_rechits) CopyDir(dirrechits);
  if (ok_digis) CopyDir(dirdigis);
  if (ok_TP) CopyDir(dirTP);
  if (ok_trackingrechits) CopyDir(dirtrackingrechits);
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
