Double_t fitFunction_g(Double_t *x, Double_t *par)
{
  const double PI=2.0*acos(0.);
  //std::cout << "PI = " << 2.0*acos(0.) << std::endl;
  const Double_t value=par[2]/(par[0]*sqrt(2*PI))*exp(-(x[0]-par[1])*(x[0]-par[1])/(2*par[0]*par[0]));
  return value;
};

Double_t fitFunction_f(Double_t *x, Double_t *par)
{
  //const Double_t value=par[2]/(par[0]*sqrt(2*PI))*exp(-(x[0]-par[1])*(x[0]-par[1])/(2*par[0]*par[0]));
  //const Double_t value=sqrt(par[0]*par[0]/(x[0]*x[0])+par[1]*par[1]/(x[0])+par[2]*par[2]);
  const Double_t value=sqrt(par[0]*par[0]+par[1]*par[1]*(x[0]-par[3])+par[2]*par[2]*(x[0]-par[3])*(x[0]-par[3]))/x[0];
  return value;
};

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


  void DrawResp(const char* key, int binxmin, int binxmax, Mode mode, double Ymin, double Ymax )
  {

    //std::cout << "binxmin = " << binxmin << std::endl;
    //std::cout << "binxmax = " << binxmax << std::endl;

    TDirectory* dir = dir1_;
    dir->cd();
    //gStyle->SetPalette(1);
    TH2F *h2 = (TH2*) dir->Get(key);
    //h2->Draw("colz");

    const unsigned int nbin=10;

    double y[nbin];
    double ey[nbin];
    double x[nbin];
    double ex[nbin];
    for (unsigned int nbinc=0;nbinc<nbin;++nbinc)
    {
      const double binxminc=binxmin+nbinc*(binxmax-binxmin)/nbin;
      const double binxmaxc=binxminc+(binxmax-binxmin)/nbin;
      x[nbinc]=(binxmaxc-binxminc)/2.+binxminc;
      ex[nbinc]=(binxmaxc-binxminc)/2.;
  
      //std::cout << "binxminc = " << binxminc << std::endl;
      //std::cout << "binxmaxc = " << binxmaxc << std::endl;
  
      TH1D* h0_slice = h2->ProjectionY("h0_slice",binxminc, binxmaxc, "");
      //h0_slice->Sumw2();
      y[nbinc]= h0_slice->GetMean(1);
  
      // calcul des incertitudes:
      //const double Sum_of_Weights = h0_slice->GetSumOfWeights();
      //const double Sum_of_Squared_Weights = h0_slice->GetSumw2()->GetSum();
      //std::cout << "Sum_of_Weights = " << Sum_of_Weights << std::endl;
      //std::cout << "Sum_of_Squared_Weights = " << Sum_of_Squared_Weights << std::endl;
      //const double Neq = pow(Sum_of_Weights,2) / Sum_of_Squared_Weights;
      //const double mean_error = h0_slice->GetRMS() / sqrt(Neq);
      //std::cout << "mean_error = " << mean_error << std::endl;
      //std::cout << "GetMeanError(1) = " << h0_slice->GetMeanError(1) << std::endl;
      //ey[nbinc]=mean_error;
      ey[nbinc]=h0_slice->GetMeanError(1);
      delete h0_slice;
    }
  
    TGraphErrors *gr = new TGraphErrors(nbin,x,y,ex,ey);
    gr->SetMaximum(Ymax);
    gr->SetMinimum(Ymin);
    gr->SetMarkerStyle(21);
    gr->SetMarkerColor(4);
    gr->SetTitle("Response");
    gr->GetXaxis()->SetTitle("trueMET");
    gr->Draw("AP");

    dir = dir0_;
    dir->cd();
    //gStyle->SetPalette(1);
    TH2F *h2b = (TH2*) dir->Get(key);
    //h2->Draw("colz");

    double yb[nbin];
    double eyb[nbin];
    for (unsigned int nbinc=0;nbinc<nbin;++nbinc)
    {
      const double binxminc=binxmin+nbinc*(binxmax-binxmin)/nbin;
      const double binxmaxc=binxminc+(binxmax-binxmin)/nbin;
  
      TH1D* h0_sliceb = h2b->ProjectionY("h0_sliceb",binxminc, binxmaxc, "");
      //h0_sliceb->Sumw2();
      yb[nbinc]= h0_sliceb->GetMean(1);
  
      // calcul des incertitudes:
      //const double Sum_of_Weightsb = h0_sliceb->GetSumOfWeights();
      //const double Sum_of_Squared_Weightsb = h0_sliceb->GetSumw2()->GetSum();
      //const double Neqb = pow(Sum_of_Weightsb,2) / Sum_of_Squared_Weightsb;
      //const double mean_errorb = h0_slice->GetRMS() / sqrt(Neqb);
      eyb[nbinc]=h0_sliceb->GetMeanError(1);
      delete h0_sliceb;
    }
    TGraphErrors *grb = new TGraphErrors(nbin,x,yb,ex,eyb);
    grb->SetMarkerStyle(21);
    grb->SetMarkerColor(2);
    grb->Draw("P");
  }

  void DrawSigmaEt_Et(const char* key, int binxmin, int binxmax, Mode mode)
  {
    //std::cout << "binxmin = " << binxmin << std::endl;
    //std::cout << "binxmax = " << binxmax << std::endl;
    TDirectory* dir = dir1_;
    dir->cd();
    //gStyle->SetPalette(1);
    TH2F *h2 = (TH2*) dir->Get(key);
    //h2->Draw("colz");

    const unsigned int nbin=9;

    double y[nbin];
    double ey[nbin];
    double x[nbin];
    double ex[nbin];
  
    //binning computation
    // (we want approx. the same number of entries per bin)
    //TH1D* h0_slice1 = h2->ProjectionY("h0_slice1",0., binxmax, "");
    TH1D* h0_slice1 = h2->ProjectionY("h0_slice1",binxmin, binxmax, "");
    const unsigned int totalNumberOfEvents=h0_slice1->GetEntries();
    //std::cout << "totalNumberOfEvents = " << totalNumberOfEvents << std::endl;
    unsigned int neventsc=0;
    for (unsigned int binc=0;binc<nbin;++binc)
    {
      unsigned int binXmaxc;
      if (binc==0) binXmaxc=binxmin;
      else binXmaxc=x[binc-1]+ex[binc-1];
  
      //std::cout << "binXmaxc = " << binXmaxc << std::endl;
      //std::cout << "(binc+1)*totalNumberOfEvents/nbin = " <<
      // (binc+1)*totalNumberOfEvents/nbin << std::endl;
      //std::cout << "neventsc = " << neventsc << std::endl;
  
      while (static_cast<double>(neventsc)<(binc+1)*totalNumberOfEvents/nbin)
      {
        TH1D* h0_slice1c = h2->ProjectionY("h0_slice1",binxmin, binXmaxc, "");
        neventsc=h0_slice1c->GetEntries();
        //std::cout << "FL : neventsc = " << neventsc << std::endl;
        //std::cout << "FL : binXmaxc = " << binXmaxc << std::endl;
        ++binXmaxc;
        delete h0_slice1c;
      }
      //std::cout << "binXmaxc = " << binXmaxc << std::endl;
      if (binc==0)
      {
        x[binc]=(binXmaxc-1-binxmin)/2.+binxmin;
        ex[binc]=(binXmaxc-1-binxmin)/2.;
      }
      else
      {
        x[binc]=(binXmaxc-1-x[binc-1]-ex[binc-1])/2.+x[binc-1]+ex[binc-1];
        //std::cout << "x[binc-1] = " << x[binc-1] << std::endl;
        //std::cout << "ex[binc-1] = " << ex[binc-1] << std::endl;
        //std::cout << "binXmaxc = " << binXmaxc << std::endl;
        ex[binc]=(binXmaxc-1-x[binc-1]-ex[binc-1])/2.;
      }
      //std::cout << "x[" << binc << "] = " << x[binc] << std::endl;
      //std::cout << "ex[" << binc << "] = " << ex[binc] << std::endl;
      //std::cout << "neventsc = " << neventsc << std::endl;
    }
  
    for (unsigned int nbinc=0;nbinc<nbin;++nbinc)
    {
      //const double binxminc=binxmin+nbinc*(binxmax-binxmin)/nbin;
      //const double binxmaxc=binxminc+(binxmax-binxmin)/nbin;
      const double binxminc=x[nbinc]-ex[nbinc];
      double binxmaxc=x[nbinc]+ex[nbinc];
      if (nbinc==nbin-1) binxmaxc=binxmax;
      //x[nbinc]=(binxmaxc-binxminc)/2.+binxminc;
      //ex[nbinc]=(binxmaxc-binxminc)/2.;
  
      //std::cout << "binxminc = " << binxminc << std::endl;
      //std::cout << "binxmaxc = " << binxmaxc << std::endl;
  
      TH1D* h0_slice = h2->ProjectionY("h0_slice",binxminc, binxmaxc, "");
      //h0_slice->Sumw2();
      h0_slice->Draw();
      //std::cout << "GetRMS(1) = " << h0_slice->GetRMS(1) << std::endl;
  
      TF1 *fitFcng = new TF1("fitFcng",fitFunction_g,-100.,100.,3);
      fitFcng->SetParameters(50.,0.1,100.);
      fitFcng->SetLineColor(2);
      h0_slice->Fit("fitFcng","0R");
      fitFcng->Draw("same");
  
      std::ostringstream oss;
      oss << nbinc;
      const std::string plotfitname="Plots/fitbin_DrawSigmaEt_Et_"+oss.str()+".eps";
      gPad->SaveAs( plotfitname.c_str() );
  
      const double sigmaG=fitFcng->GetParameter(0);
      //std::cout << "Sigma = " << sigmaG << std::endl;
      y[nbinc]= sigmaG/x[nbinc];
  
      //std::cout << "FL: x[" << nbinc << "] = " << x[nbinc] << std::endl;
  
      // calcul des incertitudes:
      ey[nbinc]=y[nbinc]*(fitFcng->GetParError(0)/fitFcng->GetParameter(0)+ex[nbinc]/x[nbinc]);
      //ey[nbinc]=0.0;
      //std::cout << "ey[nbinc] = " << ey[nbinc] << std::endl;
      delete h0_slice;
    }
  
    TGraphErrors *gr = new TGraphErrors(nbin,x,y,ex,ey);
    gr->SetMaximum(1.1);
    gr->SetMinimum(0.0);
    gr->SetMarkerStyle(21);
    gr->SetMarkerColor(4);
    gr->SetTitle("Sigma(MET)/trueMET");
    gr->GetXaxis()->SetTitle("trueMET");
    gr->Draw("AP");
  
    TF1 *fitFcne = new TF1("fitFcne",fitFunction_f,20.,200.,4);
    fitFcne->SetNpx(500);
    fitFcne->SetLineWidth(3);
    fitFcne->SetLineStyle(1);
    fitFcne->SetLineColor(4);
    gr->Fit("fitFcne","0R");
    fitFcne->Draw("same"); 
  
  
    double yrms[nbin];
    double eyrms[nbin];
    for (unsigned int nbinc=0;nbinc<nbin;++nbinc)
    {
      //const double binxminc=binxmin+nbinc*(binxmax-binxmin)/nbin;
      //const double binxmaxc=binxminc+(binxmax-binxmin)/nbin;
      const double binxminc=x[nbinc]-ex[nbinc];
      double binxmaxc=x[nbinc]+ex[nbinc];
      if (nbinc==nbin-1) binxmaxc=binxmax;
      //x[nbinc]=(binxmaxc-binxminc)/2.+binxminc;
      //ex[nbinc]=(binxmaxc-binxminc)/2.;
  
      //std::cout << "binxminc = " << binxminc << std::endl;
      //std::cout << "binxmaxc = " << binxmaxc << std::endl;
  
      TH1D* h0_slice = h2->ProjectionY("h0_slice",binxminc, binxmaxc, "");
      //h0_slice->Sumw2();
      yrms[nbinc]= h0_slice->GetRMS(1)/x[nbinc];
      //std::cout << "GetRMS(1) = " << h0_slice->GetRMS(1) << std::endl;
      //std::cout << "x[nbinc] = " << x[nbinc] << std::endl;
      //std::cout << "yrms[nbinc] = " << yrms[nbinc] << std::endl;
  
      // calcul des incertitudes:
      eyrms[nbinc]=yrms[nbinc]*(h0_slice->GetRMSError(1)/h0_slice->GetRMS(1)+ex[nbinc]/x[nbinc]);
      //ey[nbinc]=0.0;
      delete h0_slice;
    }
  
    TGraphErrors *grrms = new TGraphErrors(nbin,x,yrms,ex,eyrms);
    grrms->SetMaximum(1.1);
    grrms->SetMinimum(0.0);
    grrms->SetMarkerStyle(21);
    grrms->SetMarkerColor(5);
    grrms->SetTitle("RMS(MET)/trueMET");
    grrms->GetXaxis()->SetTitle("trueMET");
    //grrms->Draw("P");
  
    TF1 *fitFcnrms = new TF1("fitFcnrms",fitFunction_f,20.,200.,4);
    fitFcnrms->SetNpx(500);
    fitFcnrms->SetLineWidth(3);
    fitFcnrms->SetLineStyle(2);
    fitFcnrms->SetLineColor(4);
    grrms->Fit("fitFcnrms","0R");
    fitFcnrms->Draw("same"); 

    dir = dir0_;
    dir->cd();
    //gStyle->SetPalette(1);
    TH2F *h2b = (TH2*) dir->Get(key);
    //h2->Draw("colz");
    double ybg[nbin];
    double eybg[nbin];
  
    for (unsigned int nbinc=0;nbinc<nbin;++nbinc)
    {
      //const double binxminc=binxmin+nbinc*(binxmax-binxmin)/nbin;
      //const double binxmaxc=binxminc+(binxmax-binxmin)/nbin;
      const double binxminc=x[nbinc]-ex[nbinc];
      double binxmaxc=x[nbinc]+ex[nbinc];
      if (nbinc==nbin-1) binxmaxc=binxmax;
      //x[nbinc]=(binxmaxc-binxminc)/2.+binxminc;
      //ex[nbinc]=(binxmaxc-binxminc)/2.;
  
      //std::cout << "binxminc = " << binxminc << std::endl;
      //std::cout << "binxmaxc = " << binxmaxc << std::endl;
  
      TH1D* h0_slicebg = h2b->ProjectionY("h0_slicebg",binxminc, binxmaxc, "");
      //h0_slice->Sumw2();
      //h0_slicebg->Draw();
      //std::cout << "GetRMS(1) = " << h0_slice->GetRMS(1) << std::endl;
  
      TF1 *fitFcngbg = new TF1("fitFcngbg",fitFunction_g,-100.,100.,3);
      fitFcngbg->SetParameters(50.,0.1,100.);
      fitFcngbg->SetLineColor(2);
      h0_slicebg->Fit("fitFcngbg","0R");
      //fitFcng->Draw("same");
  
      //std::ostringstream oss;
      //oss << nbinc;
      //const std::string plotfitname="Plots/fitbin_b_"+oss.str()+".eps";
      //gPad->SaveAs( plotfitname.c_str() );
  
      const double sigmaG=fitFcngbg->GetParameter(0);
      //std::cout << "Sigma = " << sigmaG << std::endl;
      ybg[nbinc]= sigmaG/x[nbinc];
  
      //std::cout << "FL: x[" << nbinc << "] = " << x[nbinc] << std::endl;
      //std::cout << "FL: ybg[" << nbinc << "] = " << ybg[nbinc] << std::endl;
  
  //    // calcul des incertitudes:
      eybg[nbinc]=ybg[nbinc]*(fitFcng->GetParError(0)/fitFcng->GetParameter(0)+ex[nbinc]/x[nbinc]);
  //    //ey[nbinc]=0.0;
      //std::cout << "ey[nbinc] = " << ey[nbinc] << std::endl;
      delete h0_slicebg;
    }
  
    TGraphErrors *grbg = new TGraphErrors(nbin,x,ybg,ex,eybg);
    grbg->SetMaximum(1.1);
    grbg->SetMinimum(0.0);
    grbg->SetMarkerStyle(21);
    grbg->SetMarkerColor(2);
    grbg->SetTitle("Sigma(MET)/trueMET");
    grbg->GetXaxis()->SetTitle("trueMET");
    grbg->Draw("P");
  
    TF1 *fitFcnebg = new TF1("fitFcnebg",fitFunction_f,20.,200.,4);
    fitFcnebg->SetNpx(500);
    fitFcnebg->SetLineWidth(3);
    fitFcnebg->SetLineStyle(1);
    fitFcnebg->SetLineColor(2);
    grbg->Fit("fitFcnebg","0R");
    fitFcnebg->Draw("same"); 
  
    double yb[nbin];
    double eyb[nbin];
    for (unsigned int nbinc=0;nbinc<nbin;++nbinc)
    {
      const double binxminc=x[nbinc]-ex[nbinc];
      double binxmaxc=x[nbinc]+ex[nbinc];
      if (nbinc==nbin-1) binxmaxc=binxmax;
  
      TH1D* h0_sliceb = h2b->ProjectionY("h0_sliceb",binxminc, binxmaxc, "");
      //h0_sliceb->Sumw2();
      yb[nbinc]= h0_sliceb->GetRMS(1)/x[nbinc];
  
      // calcul des incertitudes:
      //eyb[nbinc]=0.0;
      eyb[nbinc]=yb[nbinc]*(h0_sliceb->GetRMSError(1)/h0_sliceb->GetRMS(1)+ex[nbinc]/x[nbinc]);
      delete h0_sliceb;
    }
  
    TGraphErrors *grb = new TGraphErrors(nbin,x,yb,ex,eyb);
    grb->SetMarkerStyle(21);
    grb->SetMarkerColor(2);
    //grb->Draw("P");
  
    TF1 *fitFcne2 = new TF1("fitFcne2",fitFunction_f,20.,200.,4);
    fitFcne2->SetNpx(500);
    fitFcne2->SetLineWidth(3);
    fitFcne2->SetLineColor(2);
    fitFcne2->SetLineStyle(2);
    grb->Fit("fitFcne2","0R");
    fitFcne2->Draw("same"); 
  }

  void DrawSigmaEt(const char* key, int binxmin, int binxmax, Mode mode)
  {
    //std::cout << "binxmin = " << binxmin << std::endl;
    //std::cout << "binxmax = " << binxmax << std::endl;
    TDirectory* dir = dir1_;
    dir->cd();
    //gStyle->SetPalette(1);
    TH2F *h2 = (TH2*) dir->Get(key);
    //h2->Draw("colz");

    const unsigned int nbin=6;

    double y[nbin];
    double ey[nbin];
    double x[nbin];
    double ex[nbin];
  
    //binning computation
    // (we want approx. the same number of entries per bin)
    //TH1D* h0_slice1 = h2->ProjectionY("h0_slice1",0., binxmax, "");
    TH1D* h0_slice1 = h2->ProjectionY("h0_slice1",binxmin, binxmax, "");
    const unsigned int totalNumberOfEvents=h0_slice1->GetEntries();
    //std::cout << "totalNumberOfEvents = " << totalNumberOfEvents << std::endl;
    unsigned int neventsc=0;
    for (unsigned int binc=0;binc<nbin;++binc)
    {
      unsigned int binXmaxc;
      if (binc==0) binXmaxc=binxmin;
      else binXmaxc=x[binc-1]+ex[binc-1];
  
      //std::cout << "binXmaxc = " << binXmaxc << std::endl;
      //std::cout << "(binc+1)*totalNumberOfEvents/nbin = " <<
      // (binc+1)*totalNumberOfEvents/nbin << std::endl;
      //std::cout << "neventsc = " << neventsc << std::endl;
  
      while (static_cast<double>(neventsc)<(binc+1)*totalNumberOfEvents/nbin)
      {
        TH1D* h0_slice1c = h2->ProjectionY("h0_slice1",binxmin, binXmaxc, "");
        neventsc=h0_slice1c->GetEntries();
        //std::cout << "FL : neventsc = " << neventsc << std::endl;
        //std::cout << "FL : binXmaxc = " << binXmaxc << std::endl;
        ++binXmaxc;
        //binXmaxc+=10;
        delete h0_slice1c;
      }
      //std::cout << "binXmaxc = " << binXmaxc << std::endl;
      if (binc==0)
      {
        x[binc]=(binXmaxc-1-binxmin)/2.+binxmin;
        ex[binc]=(binXmaxc-1-binxmin)/2.;
      }
      else
      {
        x[binc]=(binXmaxc-1-x[binc-1]-ex[binc-1])/2.+x[binc-1]+ex[binc-1];
        //std::cout << "x[binc-1] = " << x[binc-1] << std::endl;
        //std::cout << "ex[binc-1] = " << ex[binc-1] << std::endl;
        //std::cout << "binXmaxc = " << binXmaxc << std::endl;
        ex[binc]=(binXmaxc-1-x[binc-1]-ex[binc-1])/2.;
      }
      //std::cout << "x[" << binc << "] = " << x[binc] << std::endl;
      //std::cout << "ex[" << binc << "] = " << ex[binc] << std::endl;
      //std::cout << "neventsc = " << neventsc << std::endl;
    }
  
    for (unsigned int nbinc=0;nbinc<nbin;++nbinc)
    {
      //const double binxminc=binxmin+nbinc*(binxmax-binxmin)/nbin;
      //const double binxmaxc=binxminc+(binxmax-binxmin)/nbin;
      const double binxminc=x[nbinc]-ex[nbinc];
      double binxmaxc=x[nbinc]+ex[nbinc];
      if (nbinc==nbin-1) binxmaxc=binxmax;
      //x[nbinc]=(binxmaxc-binxminc)/2.+binxminc;
      //ex[nbinc]=(binxmaxc-binxminc)/2.;
  
      //std::cout << "binxminc = " << binxminc << std::endl;
      //std::cout << "binxmaxc = " << binxmaxc << std::endl;
  
      TH1D* h0_slice = h2->ProjectionY("h0_slice",binxminc, binxmaxc, "");
      //h0_slice->Sumw2();
      h0_slice->Rebin(5);
      h0_slice->Draw();
      //std::cout << "GetRMS(1) = " << h0_slice->GetRMS(1) << std::endl;
  
      TF1 *fitFcng = new TF1("fitFcng",fitFunction_g,-2.,2.,3);
      //TF1 *fitFcng = new TF1("fitFcng",fitFunction_g,-100.,100.,3);
      //fitFcng->SetParameters(1.,0.1,100.);
      fitFcng->SetParameters(h0_slice->GetRMS(1),0.1,100.);
      fitFcng->SetLineColor(2);
      h0_slice->Fit("fitFcng","0R");
      fitFcng->Draw("same");
  
      std::ostringstream oss;
      oss << nbinc;
      const std::string plotfitname="Plots/fitbin_DrawSigmaEt_"+oss.str()+".eps";
      gPad->SaveAs( plotfitname.c_str() );
  
      const double sigmaG=fitFcng->GetParameter(0);
      //std::cout << "Sigma = " << sigmaG << std::endl;
      y[nbinc]= sigmaG;
  
      //std::cout << "FL: x[" << nbinc << "] = " << x[nbinc] << std::endl;
  
      // calcul des incertitudes:
      ey[nbinc]=fitFcng->GetParError(0);
      //ey[nbinc]=0.0;
      //std::cout << "ey[nbinc] = " << ey[nbinc] << std::endl;
      delete h0_slice;
    }
  
    TGraphErrors *gr = new TGraphErrors(nbin,x,y,ex,ey);
    gr->SetMaximum(1.3);
    gr->SetMinimum(0.0);
    gr->SetMarkerStyle(21);
    gr->SetMarkerColor(4);
    gr->SetTitle("Sigma(Phi)");
    gr->GetXaxis()->SetTitle("trueMET");
    gr->Draw("AP");
  
    TF1 *fitFcne3 = new TF1("fitFcne3",fitFunction_f,20.,200.,4);
    fitFcne3->SetNpx(500);
    fitFcne3->SetLineWidth(3);
    fitFcne3->SetLineStyle(1);
    fitFcne3->SetLineColor(4);
    gr->Fit("fitFcne3","0R");
    fitFcne3->Draw("same"); 
  
  
    double yrms[nbin];
    double eyrms[nbin];
    for (unsigned int nbinc=0;nbinc<nbin;++nbinc)
    {
      //const double binxminc=binxmin+nbinc*(binxmax-binxmin)/nbin;
      //const double binxmaxc=binxminc+(binxmax-binxmin)/nbin;
      const double binxminc=x[nbinc]-ex[nbinc];
      double binxmaxc=x[nbinc]+ex[nbinc];
      if (nbinc==nbin-1) binxmaxc=binxmax;
      //x[nbinc]=(binxmaxc-binxminc)/2.+binxminc;
      //ex[nbinc]=(binxmaxc-binxminc)/2.;
  
      //std::cout << "binxminc = " << binxminc << std::endl;
      //std::cout << "binxmaxc = " << binxmaxc << std::endl;
  
      TH1D* h0_slice = h2->ProjectionY("h0_slice",binxminc, binxmaxc, "");
      //h0_slice->Sumw2();
      yrms[nbinc]= h0_slice->GetRMS(1);
      //std::cout << "GetRMS(1) = " << h0_slice->GetRMS(1) << std::endl;
      //std::cout << "x[nbinc] = " << x[nbinc] << std::endl;
      //std::cout << "yrms[nbinc] = " << yrms[nbinc] << std::endl;
  
      // calcul des incertitudes:
      eyrms[nbinc]=h0_slice->GetRMSError(1);
      //ey[nbinc]=0.0;
      delete h0_slice;
    }
  
    TGraphErrors *grrms = new TGraphErrors(nbin,x,yrms,ex,eyrms);
    grrms->SetMaximum(1.3);
    grrms->SetMinimum(0.0);
    grrms->SetMarkerStyle(21);
    grrms->SetMarkerColor(5);
    grrms->SetTitle("RMS(MET)/trueMET");
    grrms->GetXaxis()->SetTitle("trueMET");
    //grrms->Draw("P");
  
    TF1 *fitFcnrms3 = new TF1("fitFcnrms3",fitFunction_f,20.,200.,4);
    fitFcnrms3->SetNpx(500);
    fitFcnrms3->SetLineWidth(3);
    fitFcnrms3->SetLineStyle(2);
    fitFcnrms3->SetLineColor(4);
    grrms->Fit("fitFcnrms3","0R");
    fitFcnrms3->Draw("same"); 

    dir = dir0_;
    dir->cd();
    //gStyle->SetPalette(1);
    TH2F *h2b = (TH2*) dir->Get(key);
    //h2->Draw("colz");
    double ybg[nbin];
    double eybg[nbin];
  
    for (unsigned int nbinc=0;nbinc<nbin;++nbinc)
    {
      //const double binxminc=binxmin+nbinc*(binxmax-binxmin)/nbin;
      //const double binxmaxc=binxminc+(binxmax-binxmin)/nbin;
      const double binxminc=x[nbinc]-ex[nbinc];
      double binxmaxc=x[nbinc]+ex[nbinc];
      if (nbinc==nbin-1) binxmaxc=binxmax;
      //x[nbinc]=(binxmaxc-binxminc)/2.+binxminc;
      //ex[nbinc]=(binxmaxc-binxminc)/2.;
  
      //std::cout << "binxminc = " << binxminc << std::endl;
      //std::cout << "binxmaxc = " << binxmaxc << std::endl;
  
      TH1D* h0_slicebg = h2b->ProjectionY("h0_slicebg",binxminc, binxmaxc, "");
      //h0_slice->Sumw2();
      h0_slicebg->Rebin(5);
      //h0_slicebg->Draw();
      //std::cout << "GetRMS(1) = " << h0_slice->GetRMS(1) << std::endl;
  
      TF1 *fitFcngbg = new TF1("fitFcngbg",fitFunction_g,-2.,2.,3);
      //TF1 *fitFcngbg = new TF1("fitFcngbg",fitFunction_g,-100.,100.,3);
      //fitFcngbg->SetParameters(1.,0.1,100.);
      fitFcngbg->SetParameters(h0_slicebg->GetRMS(1),0.1,100.);
      fitFcngbg->SetLineColor(2);
      h0_slicebg->Fit("fitFcngbg","0R");
      //fitFcng->Draw("same");
  
      //std::ostringstream oss;
      //oss << nbinc;
      //const std::string plotfitname="Plots/fitbin_b_"+oss.str()+".eps";
      //gPad->SaveAs( plotfitname.c_str() );
  
      const double sigmaG=fitFcngbg->GetParameter(0);
      //std::cout << "Sigma = " << sigmaG << std::endl;
      ybg[nbinc]= sigmaG;
  
      //std::cout << "FL: x[" << nbinc << "] = " << x[nbinc] << std::endl;
      //std::cout << "FL: ybg[" << nbinc << "] = " << ybg[nbinc] << std::endl;
  
  //    // calcul des incertitudes:
      eybg[nbinc]=fitFcng->GetParError(0);
  //    //ey[nbinc]=0.0;
      //std::cout << "ey[nbinc] = " << ey[nbinc] << std::endl;
      delete h0_slicebg;
    }
  
    TGraphErrors *grbg = new TGraphErrors(nbin,x,ybg,ex,eybg);
    grbg->SetMaximum(1.3);
    grbg->SetMinimum(0.0);
    grbg->SetMarkerStyle(21);
    grbg->SetMarkerColor(2);
    grbg->SetTitle("Sigma(MET)/trueMET");
    grbg->GetXaxis()->SetTitle("trueMET");
    grbg->Draw("P");
  
    TF1 *fitFcnebg3 = new TF1("fitFcnebg3",fitFunction_f,20.,200.,4);
    fitFcnebg3->SetNpx(500);
    fitFcnebg3->SetLineWidth(3);
    fitFcnebg3->SetLineStyle(1);
    fitFcnebg3->SetLineColor(2);
    grbg->Fit("fitFcnebg3","0R");
    fitFcnebg3->Draw("same"); 
  
    double yb[nbin];
    double eyb[nbin];
    for (unsigned int nbinc=0;nbinc<nbin;++nbinc)
    {
      const double binxminc=x[nbinc]-ex[nbinc];
      double binxmaxc=x[nbinc]+ex[nbinc];
      if (nbinc==nbin-1) binxmaxc=binxmax;
  
      TH1D* h0_sliceb = h2b->ProjectionY("h0_sliceb",binxminc, binxmaxc, "");
      //h0_sliceb->Sumw2();
      yb[nbinc]= h0_sliceb->GetRMS(1);
  
      // calcul des incertitudes:
      //eyb[nbinc]=0.0;
      eyb[nbinc]=h0_sliceb->GetRMSError(1);
      delete h0_sliceb;
    }
  
    TGraphErrors *grb = new TGraphErrors(nbin,x,yb,ex,eyb);
    grb->SetMarkerStyle(21);
    grb->SetMarkerColor(2);
    //grb->Draw("P");
  
    TF1 *fitFcne23 = new TF1("fitFcne23",fitFunction_f,20.,200.,4);
    fitFcne23->SetNpx(500);
    fitFcne23->SetLineWidth(3);
    fitFcne23->SetLineColor(2);
    fitFcne23->SetLineStyle(2);
    grb->Fit("fitFcne23","0R");
    fitFcne23->Draw("same"); 
  }

  void DrawSigmaEt2(const char* key, int binxmin, int binxmax, Mode mode)
  {
    //std::cout << "binxmin = " << binxmin << std::endl;
    //std::cout << "binxmax = " << binxmax << std::endl;
    TDirectory* dir = dir1_;
    dir->cd();
    //gStyle->SetPalette(1);
    TH2F *h2 = (TH2*) dir->Get(key);
    //h2->Draw("colz");

    const unsigned int nbin=10;

    double y[nbin];
    double ey[nbin];
    double x[nbin];
    double ex[nbin];
  
//    //binning computation
//    // (we want approx. the same number of entries per bin)
//    //TH1D* h0_slice1 = h2->ProjectionY("h0_slice1",0., binxmax, "");
//    TH1D* h0_slice1 = h2->ProjectionY("h0_slice1",binxmin, binxmax, "");
//    const unsigned int totalNumberOfEvents=h0_slice1->GetEntries();
//    //std::cout << "totalNumberOfEvents = " << totalNumberOfEvents << std::endl;
//    unsigned int neventsc=0;
//    for (unsigned int binc=0;binc<nbin;++binc)
//    {
//      unsigned int binXmaxc;
//      if (binc==0) binXmaxc=binxmin;
//      else binXmaxc=x[binc-1]+ex[binc-1];
//  
//      //std::cout << "binXmaxc = " << binXmaxc << std::endl;
//      //std::cout << "(binc+1)*totalNumberOfEvents/nbin = " <<
//      // (binc+1)*totalNumberOfEvents/nbin << std::endl;
//      //std::cout << "neventsc = " << neventsc << std::endl;
//  
//      while (static_cast<double>(neventsc)<(binc+1)*totalNumberOfEvents/nbin)
//      {
//        TH1D* h0_slice1c = h2->ProjectionY("h0_slice1",binxmin, binXmaxc, "");
//        neventsc=h0_slice1c->GetEntries();
//        //std::cout << "FL : neventsc = " << neventsc << std::endl;
//        //std::cout << "FL : binXmaxc = " << binXmaxc << std::endl;
//        //++binXmaxc;
//        binXmaxc+=10;
//        delete h0_slice1c;
//      }
//      //std::cout << "binXmaxc = " << binXmaxc << std::endl;
//      if (binc==0)
//      {
//        x[binc]=(binXmaxc-1-binxmin)/2.+binxmin;
//        ex[binc]=(binXmaxc-1-binxmin)/2.;
//      }
//      else
//      {
//        x[binc]=(binXmaxc-1-x[binc-1]-ex[binc-1])/2.+x[binc-1]+ex[binc-1];
//        //std::cout << "x[binc-1] = " << x[binc-1] << std::endl;
//        //std::cout << "ex[binc-1] = " << ex[binc-1] << std::endl;
//        //std::cout << "binXmaxc = " << binXmaxc << std::endl;
//        ex[binc]=(binXmaxc-1-x[binc-1]-ex[binc-1])/2.;
//      }
//      //std::cout << "x[" << binc << "] = " << x[binc] << std::endl;
//      //std::cout << "ex[" << binc << "] = " << ex[binc] << std::endl;
//      //std::cout << "neventsc = " << neventsc << std::endl;
//    }
  
    for (unsigned int nbinc=0;nbinc<nbin;++nbinc)
    {
      const double binxminc=binxmin+nbinc*(binxmax-binxmin)/nbin;
      const double binxmaxc=binxminc+(binxmax-binxmin)/nbin;
      //const double binxminc=x[nbinc]-ex[nbinc];
      //double binxmaxc=x[nbinc]+ex[nbinc];
      //if (nbinc==nbin-1) binxmaxc=binxmax;
      x[nbinc]=(binxmaxc-binxminc)/2.+binxminc;
      ex[nbinc]=(binxmaxc-binxminc)/2.;
  
      //std::cout << "binxminc = " << binxminc << std::endl;
      //std::cout << "binxmaxc = " << binxmaxc << std::endl;
  
      TH1D* h0_slice = h2->ProjectionY("h0_slice",binxminc, binxmaxc, "");
      //h0_slice->Sumw2();
      h0_slice->Rebin(5);
      //if (nbinc<5) h0_slice->Rebin(5);
      //else h0_slice->Rebin(2);
      h0_slice->Draw();
      //std::cout << "GetRMS(1) = " << h0_slice->GetRMS(1) << std::endl;
  
      //TF1 *fitFcng = new TF1("fitFcng",fitFunction_g,-2.,2.,3);
      TF1 *fitFcng = new TF1("fitFcng",fitFunction_g,-100.,100.,3);
      //fitFcng->SetParameters(1.,0.1,100.);
      fitFcng->SetParameters(h0_slice->GetRMS(1),0.1,100.);
      fitFcng->SetLineColor(2);
      h0_slice->Fit("fitFcng","0R");
      fitFcng->Draw("same");
  
      std::ostringstream oss;
      oss << nbinc;
      const std::string plotfitname="Plots/fitbin_DrawSigmaEt_"+oss.str()+".eps";
      gPad->SaveAs( plotfitname.c_str() );
  
      const double sigmaG=fitFcng->GetParameter(0);
      //std::cout << "Sigma = " << sigmaG << std::endl;
      y[nbinc]= sigmaG;
  
      //std::cout << "FL: x[" << nbinc << "] = " << x[nbinc] << std::endl;
  
      // calcul des incertitudes:
      ey[nbinc]=fitFcng->GetParError(0);
      //ey[nbinc]=0.0;
      //std::cout << "ey[nbinc] = " << ey[nbinc] << std::endl;
      delete h0_slice;
    }
  
    TGraphErrors *gr = new TGraphErrors(nbin,x,y,ex,ey);
    //gr->SetMaximum(1.3);
    gr->SetMinimum(0.0);
    gr->SetMarkerStyle(21);
    gr->SetMarkerColor(4);
    gr->SetTitle("Sigma(DeltaSET)");
    gr->GetXaxis()->SetTitle("trueSET");
    gr->Draw("AP");
  
//    TF1 *fitFcne3 = new TF1("fitFcne3",fitFunction_f,20.,200.,4);
//    fitFcne3->SetNpx(500);
//    fitFcne3->SetLineWidth(3);
//    fitFcne3->SetLineStyle(1);
//    fitFcne3->SetLineColor(4);
//    gr->Fit("fitFcne3","0R");
//    fitFcne3->Draw("same"); 
  
  
    double yrms[nbin];
    double eyrms[nbin];
    for (unsigned int nbinc=0;nbinc<nbin;++nbinc)
    {
      const double binxminc=binxmin+nbinc*(binxmax-binxmin)/nbin;
      const double binxmaxc=binxminc+(binxmax-binxmin)/nbin;
      //const double binxminc=x[nbinc]-ex[nbinc];
      //double binxmaxc=x[nbinc]+ex[nbinc];
      //if (nbinc==nbin-1) binxmaxc=binxmax;
      //x[nbinc]=(binxmaxc-binxminc)/2.+binxminc;
      //ex[nbinc]=(binxmaxc-binxminc)/2.;
  
      //std::cout << "binxminc = " << binxminc << std::endl;
      //std::cout << "binxmaxc = " << binxmaxc << std::endl;
  
      TH1D* h0_slice = h2->ProjectionY("h0_slice",binxminc, binxmaxc, "");
      //h0_slice->Sumw2();
      yrms[nbinc]= h0_slice->GetRMS(1);
      //std::cout << "GetRMS(1) = " << h0_slice->GetRMS(1) << std::endl;
      //std::cout << "x[nbinc] = " << x[nbinc] << std::endl;
      //std::cout << "yrms[nbinc] = " << yrms[nbinc] << std::endl;
  
      // calcul des incertitudes:
      eyrms[nbinc]=h0_slice->GetRMSError(1);
      //ey[nbinc]=0.0;
      delete h0_slice;
    }
  
    TGraphErrors *grrms = new TGraphErrors(nbin,x,yrms,ex,eyrms);
    //grrms->SetMaximum(1.3);
    grrms->SetMinimum(0.0);
    grrms->SetMarkerStyle(22);
    grrms->SetMarkerColor(4);
    grrms->SetTitle("RMS(MET)/trueMET");
    grrms->GetXaxis()->SetTitle("trueMET");
    //grrms->Draw("P");
  
//    TF1 *fitFcnrms3 = new TF1("fitFcnrms3",fitFunction_f,20.,200.,4);
//    fitFcnrms3->SetNpx(500);
//    fitFcnrms3->SetLineWidth(3);
//    fitFcnrms3->SetLineStyle(2);
//    fitFcnrms3->SetLineColor(4);
//    grrms->Fit("fitFcnrms3","0R");
//    fitFcnrms3->Draw("same"); 

    dir = dir0_;
    dir->cd();
    //gStyle->SetPalette(1);
    TH2F *h2b = (TH2*) dir->Get(key);
    //h2->Draw("colz");
    double ybg[nbin];
    double eybg[nbin];
  
    for (unsigned int nbinc=0;nbinc<nbin;++nbinc)
    {
      const double binxminc=binxmin+nbinc*(binxmax-binxmin)/nbin;
      const double binxmaxc=binxminc+(binxmax-binxmin)/nbin;
      //const double binxminc=x[nbinc]-ex[nbinc];
      //double binxmaxc=x[nbinc]+ex[nbinc];
      //if (nbinc==nbin-1) binxmaxc=binxmax;
      //x[nbinc]=(binxmaxc-binxminc)/2.+binxminc;
      //ex[nbinc]=(binxmaxc-binxminc)/2.;
  
      //std::cout << "binxminc = " << binxminc << std::endl;
      //std::cout << "binxmaxc = " << binxmaxc << std::endl;
  
      TH1D* h0_slicebg = h2b->ProjectionY("h0_slicebg",binxminc, binxmaxc, "");
      //h0_slice->Sumw2();
      h0_slicebg->Rebin(5);
      //if (nbinc<5) h0_slice->Rebin(5);
      //else h0_slice->Rebin(2);
      //h0_slicebg->Draw();
      //std::cout << "GetRMS(1) = " << h0_slice->GetRMS(1) << std::endl;
  
      //TF1 *fitFcngbg = new TF1("fitFcngbg",fitFunction_g,-2.,2.,3);
      TF1 *fitFcngbg = new TF1("fitFcngbg",fitFunction_g,-100.,100.,3);
      //fitFcngbg->SetParameters(1.,0.1,100.);
      fitFcngbg->SetParameters(h0_slicebg->GetRMS(1),0.1,100.);
      fitFcngbg->SetLineColor(2);
      h0_slicebg->Fit("fitFcngbg","0R");
      //fitFcng->Draw("same");
  
      //std::ostringstream oss;
      //oss << nbinc;
      //const std::string plotfitname="Plots/fitbin_b_"+oss.str()+".eps";
      //gPad->SaveAs( plotfitname.c_str() );
  
      const double sigmaG=fitFcngbg->GetParameter(0);
      //std::cout << "Sigma = " << sigmaG << std::endl;
      ybg[nbinc]= sigmaG;
  
      //std::cout << "FL: x[" << nbinc << "] = " << x[nbinc] << std::endl;
      //std::cout << "FL: ybg[" << nbinc << "] = " << ybg[nbinc] << std::endl;
  
  //    // calcul des incertitudes:
      eybg[nbinc]=fitFcngbg->GetParError(0);
  //    //ey[nbinc]=0.0;
      //std::cout << "ey[nbinc] = " << ey[nbinc] << std::endl;
      delete h0_slicebg;
    }
  
    TGraphErrors *grbg = new TGraphErrors(nbin,x,ybg,ex,eybg);
    //grbg->SetMaximum(1.3);
    grbg->SetMinimum(0.0);
    grbg->SetMarkerStyle(21);
    grbg->SetMarkerColor(2);
    grbg->SetTitle("Sigma(MET)/trueMET");
    grbg->GetXaxis()->SetTitle("trueMET");
    grbg->Draw("P");
  
//    TF1 *fitFcnebg3 = new TF1("fitFcnebg3",fitFunction_f,20.,200.,4);
//    fitFcnebg3->SetNpx(500);
//    fitFcnebg3->SetLineWidth(3);
//    fitFcnebg3->SetLineStyle(1);
//    fitFcnebg3->SetLineColor(2);
//    grbg->Fit("fitFcnebg3","0R");
//    fitFcnebg3->Draw("same"); 
  
    double yb[nbin];
    double eyb[nbin];
    for (unsigned int nbinc=0;nbinc<nbin;++nbinc)
    {
      const double binxminc=binxmin+nbinc*(binxmax-binxmin)/nbin;
      const double binxmaxc=binxminc+(binxmax-binxmin)/nbin;
      //const double binxminc=x[nbinc]-ex[nbinc];
      //double binxmaxc=x[nbinc]+ex[nbinc];
      //if (nbinc==nbin-1) binxmaxc=binxmax;
  
      TH1D* h0_sliceb = h2b->ProjectionY("h0_sliceb",binxminc, binxmaxc, "");
      //h0_sliceb->Sumw2();
      yb[nbinc]= h0_sliceb->GetRMS(1);
  
      // calcul des incertitudes:
      //eyb[nbinc]=0.0;
      eyb[nbinc]=h0_sliceb->GetRMSError(1);
      delete h0_sliceb;
    }
  
    TGraphErrors *grb = new TGraphErrors(nbin,x,yb,ex,eyb);
    grb->SetMarkerStyle(22);
    grb->SetMarkerColor(2);
    //grb->Draw("P");
  
//    TF1 *fitFcne23 = new TF1("fitFcne23",fitFunction_f,20.,200.,4);
//    fitFcne23->SetNpx(500);
//    fitFcne23->SetLineWidth(3);
//    fitFcne23->SetLineColor(2);
//    fitFcne23->SetLineStyle(2);
//    grb->Fit("fitFcne23","0R");
//    fitFcne23->Draw("same"); 
  }

  void Draw2D_file1( const char* key, Mode mode) {
    TDirectory* dir = dir0_;
    dir->cd();
    gStyle->SetPalette(1);
    TH2F *h2 = (TH2*) dir->Get(key);
    h2->Draw("colz");
  }

  void Draw2D_file2( const char* key, Mode mode) {
    TDirectory* dir = dir1_;
    dir->cd();
    gStyle->SetPalette(1);
    TH2F *h2 = (TH2*) dir->Get(key);
    h2->Draw("colz");
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
  void SetStyles( Style* s0, 
		  Style* s1,
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

    //h0_->SetStats(1);
    //h1_->SetStats(1);

    if(rebin_>1) {
      h0_->Rebin( rebin_);
      h1_->Rebin( rebin_);
    }
    if(resetAxis_) {
      h0_->GetXaxis()->SetRangeUser( xMin_, xMax_);
      h1_->GetXaxis()->SetRangeUser( xMin_, xMax_);
    }

    TPaveStats *ptstats = new TPaveStats(0.7385057,0.720339,
					 0.9396552,0.8792373,"brNDC");
    ptstats->SetName("stats");
    ptstats->SetBorderSize(1);
    ptstats->SetLineColor(2);
    ptstats->SetFillColor(10);
    ptstats->SetTextAlign(12);
    ptstats->SetTextColor(2);
    ptstats->SetOptStat(1111);
    ptstats->SetOptFit(0);
    ptstats->Draw();
    h0_->GetListOfFunctions()->Add(ptstats);
    ptstats->SetParent(h0_->GetListOfFunctions());

    //std::cout << "FL: h0_->GetMean() = " << h0_->GetMean() << std::endl;
    //std::cout << "FL: h0_->GetRMS() = " << h0_->GetRMS() << std::endl;
    //std::cout << "FL: h1_->GetMean() = " << h1_->GetMean() << std::endl;
    //std::cout << "FL: h1_->GetRMS() = " << h1_->GetRMS() << std::endl;
    //std::cout << "FL: test2" << std::endl;
    TPaveStats *ptstats2 = new TPaveStats(0.7399425,0.529661,
    					  0.941092,0.6885593,"brNDC");
    ptstats2->SetName("stats");
    ptstats2->SetBorderSize(1);
    ptstats2->SetLineColor(4);
    ptstats2->SetFillColor(10);
    ptstats2->SetTextAlign(12);
    ptstats2->SetTextColor(4);
    TText *text = ptstats2->AddText("h1_");
    text->SetTextSize(0.03654661);

    std::ostringstream oss3;
    oss3 << h1_->GetEntries();
    const std::string txt_entries="Entries = "+oss3.str();
    text = ptstats2->AddText(txt_entries.c_str());
    std::ostringstream oss;
    oss << h1_->GetMean();
    const std::string txt_mean="Mean  = "+oss.str();
    text = ptstats2->AddText(txt_mean.c_str());
    std::ostringstream oss2;
    oss2 << h1_->GetRMS();
    const std::string txt_rms="RMS  = "+oss2.str();
    text = ptstats2->AddText(txt_rms.c_str());
    ptstats2->SetOptStat(1111);
    ptstats2->SetOptFit(0);
    ptstats2->Draw();
    h1_->GetListOfFunctions()->Add(ptstats2);
    ptstats2->SetParent(h1_->GetListOfFunctions());

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

