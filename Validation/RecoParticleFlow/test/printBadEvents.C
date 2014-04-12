#include <TDirectory.h>
#include <string.h>

using namespace std;

void printBadEvents(string filename){
  string run = filename.substr(filename.find("_R000")+10, 1);
  int nrun = atoi(run.c_str());
  string fname(filename.substr(filename.find_last_of("/")+1) );
  cout << "FileName is " << fname.c_str() << endl;

  ofstream outfile;
  stringstream namefile;
  namefile << "BadEvents_" << nrun << ".txt";
  outfile.open(namefile.str().c_str());
  outfile << "Bad events in file " << fname.c_str() << endl; 

  TDirectory* topDir; 
  TFile* file = TFile::Open(filename.c_str());
  if (!file->IsOpen()) {
    cerr << "Failed to open " << filename << endl; 
    return;
  }
  string dir = "DQMData/Run " + run + "/ParticleFlow/Run summary/ElectronValidation/JetPtRes/BadEvents";
  topDir = dynamic_cast<TDirectory*>( file->Get(dir.c_str()));
  topDir->cd();
  if (topDir){
    TIter next(topDir->GetListOfKeys());
    TKey *key;
    while  ( (key = dynamic_cast<TKey*>(next())) ) {
      string sflag = key->GetName();
      string info(sflag.substr(1, sflag.find_first_of(">")-1 ) );
      string run(info.substr(0, info.find_first_of("_")) );
      string evt = info.substr( info.find_first_of("_")+1, info.find_last_of("_")-2);
      string ls(info.substr(info.find_last_of("_")+1,info.find_first_of("_")+1));
      string ptres = ( sflag.substr( sflag.find( "f=" ) + 2, 6 ) ).c_str();
      cout << "Event info: Run " << run << " LS " << ls << " Evt " << evt << " Jet Pt Res = " << ptres << endl; 
      outfile << "Event info: Run " << run << " LS " << ls << " Evt " << evt << " Jet Pt Res = " << ptres << endl; 
    }
  }
}
