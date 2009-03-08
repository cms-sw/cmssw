
class Comparator {

public:

  enum Mode {
    NORMAL,
    SCALE,
    EFF
  };


  Comparator() : rebin_(-1), xMin_(0), xMax_(0), resetAxis_(false), 
		 s0_(0), s1_(0), legend_(0,0,1,1) {}

  Comparator( const char* file0,
	      const char* dir0,
	      const char* file1,
	      const char* dir1 ) : 
    rebin_(-1), xMin_(0), xMax_(0), resetAxis_(false), 
    s0_(0), s1_(0), legend_(0,0,1,1) {
    
    SetDirs( file0, dir0, file1, dir1);
  }
  
  void SetDirs( const char* file0,
		const char* dir0,
		const char* file1,
		const char* dir1  ) {

    file0_ = new TFile( file0 );
    if( file0_->IsZombie() ) exit(1);
    dir0_ = file0_->GetDirectory( dir0 );
    if(! dir0_ ) exit(1);
    
    file1_ = new TFile( file1 );
    if( file1_->IsZombie() ) exit(1);
    dir1_ = file1_->GetDirectory( dir1 );
    if(! dir1_ ) exit(1);
  }

  // set the rebinning factor and the range
  void SetAxis( int rebin,
		float xmin, 
		float xmax) {
    rebin_ = rebin;
    xMin_ = xmin;
    xMax_ = xmax;
    resetAxis_ = true;
  }
  
  // set the rebinning factor, unset the range
  void SetAxis( int rebin ) {
    rebin_ = rebin;
    resetAxis_ = false;
  }
  
  // draws a Y projection of a slice along X
  void DrawSlice( const char* key, 
		  int binxmin, int binxmax, 
		  Mode mode ) {
    
    static int num = 0;
    
    ostrstream out0;
    out0<<"h0_2d_"<<num;
    ostrstream out1;
    out1<<"h1_2d_"<<num;
    num++;

    string name0 = out0.str();
    string name1 = out1.str();
      

    TH1* h0 = Histo( key, 0);
    TH1* h1 = Histo( key, 1);

    TH2* h0_2d = dynamic_cast< TH2* >(h0);
    TH2* h1_2d = dynamic_cast< TH2* >(h1);
    
    if(h0_2d->GetNbinsY() == 1 || 
       h1_2d->GetNbinsY() == 1 ) {
      cerr<<key<<" is not 2D"<<endl;
      return;
    }
    
    TH1::AddDirectory( false );

    TH1D* h0_slice = h0_2d->ProjectionY(name0.c_str(),
					binxmin, binxmax, "");
    TH1D* h1_slice = h1_2d->ProjectionY(name1.c_str(),
					binxmin, binxmax, "");
    TH1::AddDirectory( true );
    Draw( h0_slice, h1_slice, mode);        
  }


  void Draw( const char* key, Mode mode) {

    TH1::AddDirectory( false );
    TH1* h0 = Histo( key, 0);
    TH1* h1 = Histo( key, 1)->Clone("h1");

    TH1::AddDirectory( true );
    Draw( h0, h1, mode);    
  }

  
  void Draw( const char* key0, const char* key1, Mode mode) {
    TH1* h0 = Histo( key0, 0);
    TH1* h1 = Histo( key1, 1);
    
    Draw( h0, h1, mode);
  }

  // cd to a give path
  void cd(const char* path ) {
    path_ = path;
  }
  
  // return the two temporary 1d histograms, that have just
  // been plotted
  TH1* h0() {return h0_;}
  TH1* h1() {return h1_;}

  const TLegend& Legend() {return legend_;}
  
  // set the styles for further plots
  void SetStyles( Style* s0, Style* s1,
		  const char* leg0,
		  const char* leg1) { 
    s0_ = s0; 
    s1_ = s1;
    
    legend_.Clear();
    legend_.AddEntry( s0_, leg0, "mlf");
    legend_.AddEntry( s1_, leg1, "mlf");
  }
  
private:

  // retrieve an histogram in one of the two directories
  TH1* Histo( const char* key, unsigned dirIndex) {
    if(dirIndex<0 || dirIndex>1) { 
      cerr<<"bad dir index: "<<dirIndex<<endl;
      return 0;
    }
    TDirectory* dir;
    if(dirIndex == 0) dir = dir0_;
    if(dirIndex == 1) dir = dir1_;
    
    dir->cd();

    TH1* h = (TH1*) dir->Get(key);
    if(!h)  
      cerr<<"no key "<<key<<" in directory "<<dir->GetName()<<endl;
    return h;
  }

  // draw 2 1D histograms.
  // the histograms can be normalized to the same number of entries, 
  // or plotted as a ratio.
  void Draw( TH1* h0, TH1* h1, Mode mode ) {
    if( !(h0 && h1) ) { 
      cerr<<"invalid histo"<<endl;
      return;
    }
    
    TH1::AddDirectory( false );
    h0_ = (TH1*) h0->Clone( "h0_");
    h1_ = (TH1*) h1->Clone( "h1_");
    TH1::AddDirectory( true );
    
    // unsetting the title, since the title of projections
    // is still the title of the 2d histo
    // and this is better anyway
    h0_->SetTitle("");
    h1_->SetTitle("");    

    h0_->SetStats(1);
    h1_->SetStats(1);

    if(rebin_>1) {
      h0_->Rebin( rebin_);
      h1_->Rebin( rebin_);
    }
    if(resetAxis_) {
      h0_->GetXaxis()->SetRangeUser( xMin_, xMax_);
      h1_->GetXaxis()->SetRangeUser( xMin_, xMax_);
    }

    switch(mode) {
    case SCALE:
      h1_->Scale( h0_->GetEntries()/h1_->GetEntries() );
    case NORMAL:
      if(s0_)
	FormatHisto( h0_ , s0_);
      if(s1_)
 	FormatHisto( h1_ , s1_);
      
      if( h1_->GetMaximum()>h0_->GetMaximum()) {
	h0_->SetMaximum( h1_->GetMaximum()*1.15 );
      }
      h0_->Draw();
      h1_->Draw("same");
	
      break;
    case EFF:
      h1_->Divide( h0_ );
      if(s1_)
 	FormatHisto( h1_ , s0_);
      h1_->Draw();
    default:
      break;
    }
  }

  int rebin_;
  float xMin_;
  float xMax_;
  bool resetAxis_;

  TFile*      file0_;
  TDirectory* dir0_;
  TFile*      file1_;
  TDirectory* dir1_;
  
  TH1* h0_;
  TH1* h1_;
  
  Style* s0_;
  Style* s1_;
  
  TLegend legend_;

  string path_;
};

