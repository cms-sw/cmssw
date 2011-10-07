#include "Validation/RecoJets/interface/ManipHist.h"
#include "Validation/RecoJets/interface/RootPostScript.h"
#include <cmath>

using namespace std;

void 
ManipHist::configBlockSum(ConfigFile& cfg)
{
  try{
    //-----------------------------------------------
    // histogram manipulations
    //-----------------------------------------------  
    readVector( cfg.read<std::string>( "histWeights" ), weights_);
  }
  catch(...){
    std::cerr << "ERROR during reading of config file" << std::endl;
    std::cerr << "      misspelled variables in cfg ?" << std::endl;
    std::cerr << "      [--called in configBlockSum]"  << std::endl;
    std::exit(1);
  }
}

void 
ManipHist::configBlockDivide(ConfigFile& cfg)
{
  try{
    //-----------------------------------------------
    // histogram manipulations
    //-----------------------------------------------
    errorType_ = cfg.read<int>( "errorType" );
  }
  catch(...){
    std::cerr << "ERROR during reading of config file"   << std::endl;
    std::cerr << "      misspelled variables in cfg ?"   << std::endl;
    std::cerr << "      [--called in configBlockDivide]" << std::endl;
    std::exit(1);
  }
}

void 
ManipHist::sumHistograms()
{
  //-----------------------------------------------
  // loop histograms via the list of histogram
  // names stored in histList_, open a new page 
  // for each histogram & plot each sample in 
  // the same canvas
  //-----------------------------------------------
  for(int idx=0; idx<(int)histList_.size(); ++idx){
    //-----------------------------------------------
    // loop all samples via the list sampleList_, which 
    // containas the histograms of each sample as 
    // TObjects in TObjectArrays
    //-----------------------------------------------
    TH1F buffer;
    std::vector<TObjArray>::const_iterator hist = sampleList_.begin();
    for(int jdx=0; hist!=sampleList_.end(); ++hist, ++jdx){
      TH1F& hadd = *((TH1F*)(*hist)[idx]);
      //apply weights if required
      if( jdx<((int)weights_.size()) ){
	if( weights_[jdx]>0 ){
	  hadd.Scale(weights_[jdx]);
	}
      }
      //buffer first histogram unchanged
      if(jdx==0){
	buffer = hadd;
      }
      //add histograms buffer last one
      else{
	hadd.Add(&buffer);
	buffer = hadd;
      }
    }
    //reset maximum for all histograms
    hist = sampleList_.begin();
    for(; hist!=sampleList_.end(); ++hist){
      TH1F& hadd = *((TH1F*)(*hist)[idx]);
      hadd.SetMaximum(buffer.GetMaximum());
    }
  }
}

void 
ManipHist::divideAndDrawPs()
{
  //-----------------------------------------------
  // define canvas
  //-----------------------------------------------
  TCanvas *canv = new TCanvas("canv", "histograms", 600, 600);
  setCanvasStyle( *canv  );

  //-----------------------------------------------
  // number of files/directories in root file must 
  // be >1 otherwise return with error message
  //-----------------------------------------------
  if((sampleList_.size()!=2) && (dirNameList_.size()+fileList_.size()!=3)){
    std::cerr << "ERROR number of indicated root files/directories " << std::endl;
    std::cerr << "      is insonsistent. Need sample & reference"    << std::endl;
    std::cerr << "      file/directory being specified solitarily"   << std::endl;
    return;
  }
  //-----------------------------------------------
  // open output file
  //-----------------------------------------------
  TString output( writeTo_.c_str() );
  output += "/";
  output += "inspectRatio";
  output += ".";
  output += writeAs_;
  TPostScript psFile( output, 111); //112 for portrait
  //-----------------------------------------------
  // loop histograms via the list of histogram
  // names stored in histList_, open a new page 
  // for each histogram & plot each sample in 
  // the same canvas, divide 
  //-----------------------------------------------
  for(int idx=0; idx<(int)histList_.size(); ++idx){
    psFile.NewPage();
    TH1F& hsam = *((TH1F*)(sampleList_[0])[idx] ); //recieve sample
    TH1F& href = *((TH1F*)(sampleList_[1])[idx] ); //recieve reference
    divideHistograms(hsam, href, errorType_);
    
    setCanvLog( *canv, idx );
    setCanvGrid( *canv, idx );
    setHistStyles( hsam, idx, 0 ); //there is only one sample
    
    hsam.Draw();
    canv->RedrawAxis( );
    canv->Update( );
    if(idx == (int)histList_.size()-1){
      psFile.Close();
    }
  }
  canv->Close();
  delete canv;
}

void 
ManipHist::divideAndDrawEps()
{
  //-----------------------------------------------
  // define canvas
  //-----------------------------------------------
  TCanvas *canv = new TCanvas("canv", "histograms", 600, 600);
  setCanvasStyle( *canv  );

  //-----------------------------------------------
  // number of files/directories in root file must 
  // be >1 otherwise return with error message
  //-----------------------------------------------
  if((sampleList_.size()!=2) && (dirNameList_.size()+fileList_.size()!=3)){
    std::cerr << "ERROR number of indicated root files/directories " << std::endl;
    std::cerr << "      is insonsistent. Need sample & reference"    << std::endl;
    std::cerr << "      file/directory being specified solitarily"   << std::endl;
    return;
  }
  //-----------------------------------------------
  // loop histograms via the list of histogram
  // names stored in histList_, open a new page 
  // for each histogram & plot each sample in 
  // the same canvas
  //-----------------------------------------------  
  for(int idx=0; idx<(int)histList_.size(); ++idx){
    //-----------------------------------------------
    // open output files
    //-----------------------------------------------
    TString output( writeTo_.c_str() );
    output += "/";
    output += histList_[ idx ];
    output += ".";
    output += writeAs_;
    TPostScript psFile( output, 113);
    psFile.NewPage();
    TH1F& hsam = *((TH1F*)(sampleList_[0])[idx] ); //recieve sample
    TH1F& href = *((TH1F*)(sampleList_[1])[idx] ); //recieve reference
    divideHistograms(hsam, href, errorType_);
    
    setCanvLog( *canv, idx );
    setCanvGrid( *canv, idx );
    setHistStyles( hsam, idx, 0 ); //there is only one sample
    
    hsam.Draw();
    canv->RedrawAxis( );
    canv->Update( );
    psFile.Close();
  }
  canv->Close();
  delete canv;
}

TH1F& 
ManipHist::divideHistograms(TH1F& nom, TH1F& denom, int err)
{
  //------------------------------------------------
  // divide histograms bin by bin & add appropriate 
  // error according to specification in errorType_
  //------------------------------------------------
  for(int idx=0; idx<denom.GetNbinsX(); ++idx){
    double dx=nom.GetBinError(idx+1), x=nom.GetBinContent(idx+1);
    double dn=denom.GetBinError(idx+1), n=denom.GetBinContent(idx+1);

    if(n==0){
      nom.SetBinContent( idx+1, 0. );
      nom.SetBinError  ( idx+1, 0. );      
    }
    else{
      nom.SetBinContent( idx+1, x/n );
      if(err==1){
	nom.SetBinError( idx+1, ratioCorrelatedError  (x, dx, n, dn) );
      }
      else{
	nom.SetBinError( idx+1, ratioUncorrelatedError(x, dx, n, dn) );
      }
    }
  }
  return nom;
}

double 
ManipHist::ratioCorrelatedError(double& x, double& dx, double& n, double& dn)
{
  //------------------------------------------------
  // Get error deps of correlated ratio eps=x/n:
  //
  // For gaussian distributed quantities the formular: 
  // * deps=eps*Sqrt((dx/x)^2+(1-2*eps)*(dn/n)^2))
  // automatically turns into 
  // * deps=Sqrt(eps*(1-eps)/n)
  //------------------------------------------------
  if(x<=0) return  0;
  if(n==0) return -1;
  return (x/n<1) ? (x/n)*sqrt(std::fabs((dx*dx)/(n*n)+(1.-2.*(x/n))*(dn*dn)/(n*n))): 
                   (n/x)*sqrt(std::fabs((dn*dn)/(x*x)+(1.-2.*(n/x))*(dx*dx)/(x*x)));
}

double 
ManipHist::ratioUncorrelatedError(double& x, double& dx, double& n, double& dn)
{
  //------------------------------------------------
  // Get error deps of uncorrelated ratio eps=x/n:
  //------------------------------------------------
  if(x==0) return 0;
  if(n==0) return 0;
  return (x/n)*sqrt((dx*dx)/(x*x)+(dn*dn)/(n*n));
}
