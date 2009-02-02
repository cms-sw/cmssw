
class Comparator {

public:
  Comparator( const char* file0, const char* file1) {
    file0_ = new TFile( file0 );
    if( file0_->IsZombie() ) exit(1);
    file1_ = new TFile( file1 );
    if( file1_->IsZombie() ) exit(1);
  }

  void Draw( const char* key ) {

    TH1* h0 = Histo( key, 0);
    TH1* h1 = Histo( key, 1);
    if( h0 && h1 ) {
      h0->Draw();
      h1->Scale( h0->GetEntries()/h1->GetEntries() );
      h1->Draw("same");
      h0_ = h0;
      h1_ = h1;
    }
  }
  
  void cd(const char* path ) {
    path_ = path;
  }
  
  TH1* h0_;
  TH1* h1_;
  
  
private:
  TH1* Histo( const char* key, unsigned fileIndex) {
    if(fileIndex<0 || fileIndex>1) { 
      cerr<<"bad file index: "<<fileIndex<<endl;
      return 0;
    }
    TDirectory* f;
    if(fileIndex == 0) f = file0_;
    if(fileIndex == 1) f = file1_;
    
    string skey = path_; skey += "/"; skey += key;
    TH1* h = (TH1*) f->Get(skey.c_str());
    if(!h)  
      cerr<<"no key "<<key<<" in directory "<<f->GetName()<<endl;
    return h;
  }

  TDirectory* file0_;
  TDirectory* file1_;
  

  string path_;
};

