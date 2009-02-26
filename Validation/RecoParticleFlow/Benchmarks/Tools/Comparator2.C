
class Comparator {

public:

  enum Mode {
    NORMAL,
    SCALE,
    EFF
  };


  Comparator( const char* file0,
	      const char* dir0,
	      const char* file1,
	      const char* dir1 ) : s0_(0), s1_(0) {

    
    file0_ = new TFile( file0 );
    if( file0_->IsZombie() ) exit(1);
    dir0_ = file0_->GetDirectory( dir0 );
    if(! dir0_ ) exit(1);
    
    file1_ = new TFile( file1 );
    if( file1_->IsZombie() ) exit(1);
    dir1_ = file1_->GetDirectory( dir1 );
    if(! dir1_ ) exit(1);
  }

  void Draw( const char* key, Mode mode) {

    TH1* h0 = Histo( key, 0);
    TH1* h1 = Histo( key, 1)->Clone("h1");

    Draw( h0, h1, mode);    
  }

  
  void Draw( const char* key0, const char* key1, Mode mode) {
    TH1* h0 = Histo( key0, 0);
    TH1* h1 = Histo( key1, 1);
    
    Draw( h0, h1, mode);
  }


  void cd(const char* path ) {
    path_ = path;
  }
  
  TH1* h0() {return h0_;}
  TH1* h1() {return h1_;}
  
  void SetStyles( Style* s0, Style* s1) { 
    s0_ = s0; s1_ = s1;
  }
  
private:
  TH1* Histo( const char* key, unsigned dirIndex) {
    if(dirIndex<0 || dirIndex>1) { 
      cerr<<"bad dir index: "<<dirIndex<<endl;
      return 0;
    }
    TDirectory* dir;
    if(dirIndex == 0) dir = dir0_;
    if(dirIndex == 1) dir = dir1_;
    
    dir->cd();

//     string skey = path_; skey += "/"; skey += key;
    TH1* h = (TH1*) dir->Get(key);
    if(!h)  
      cerr<<"no key "<<key<<" in directory "<<dir->GetName()<<endl;
    return h;
  }


  void Draw( TH1* h0, TH1* h1, Mode mode ) {
    if( !(h0 && h1) ) { 
      cerr<<"invalid histo"<<endl;
      return;
    }
    switch(mode) {
    case SCALE:
      h1->Scale( h0->GetEntries()/h1->GetEntries() );
    case NORMAL:
      if(s0_)
	FormatHisto( h0 , s0_);
      if(s1_)
 	FormatHisto( h1 , s1_);
      h0->Draw();
      h1->Draw("same");
      h0_ = h0;
      h1_ = h1;
      break;
    case EFF:
      h1->Divide( h0 );
      if(s1_)
 	FormatHisto( h1 , s0_);
      h1->Draw();
      h0_ = h0;
      h1_ = h1;	
    default:
      break;
    }
  }

  TFile*      file0_;
  TDirectory* dir0_;
  TFile*      file1_;
  TDirectory* dir1_;
  
  TH1* h0_;
  TH1* h1_;
  
  Style* s0_;
  Style* s1_;
  

  string path_;
};

