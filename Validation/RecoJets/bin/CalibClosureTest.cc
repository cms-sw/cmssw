#include <memory>
#include <string>
#include <fstream>
#include <iostream>

#include "Validation/RecoJets/interface/RootSystem.h"
#include "Validation/RecoJets/interface/RootHistograms.h"
#include "Validation/RecoJets/interface/RootPostScript.h"
#include "Validation/RecoJets/interface/ManipHist.h"
#include "Validation/RecoJets/bin/NiceStyle.cc"

using namespace std;

class CalibClosureTest : public ManipHist {
 public:
  CalibClosureTest(){};
  ~CalibClosureTest(){};
  virtual void readConfig( std::string );
  void configBlockSpecific(ConfigFile&);

  //extra members
  void drawEff();
  void drawCorrel();
  TString output(const char*, std::string&, const char*);
  TString output(const char*, std::string&, std::string&, const char*);

 private:

  // extra memebers
  std::vector<std::string> corrHistNameList_;
  std::vector<std::string> corrAxesLabelList_;
  std::vector<std::string> cmpObjLabelList_;
  std::vector<std::string> effHistNameList_;
  std::vector<std::string> effAxesLabelList_;
};


void CalibClosureTest::readConfig( std::string name )
{
  ConfigFile cfg( name, "=", "$" ); 
  configBlockIO  ( cfg );
  configBlockHist( cfg );
  configBlockFit ( cfg );
  configBlockSpecific( cfg );
}

void 
CalibClosureTest::configBlockSpecific(ConfigFile& cfg)
{
  //-----------------------------------------------
  // read all configurables defined in CompHisto-
  // grams from config file. Throw human readable 
  // exception when misspellings occure
  //-----------------------------------------------
  try{
    //-----------------------------------------------
    // input/output files
    //-----------------------------------------------
    readVector ( cfg.read<std::string>( "corrHistNames" ), corrHistNameList_ );
    readLabels ( cfg.read<std::string>( "corrAxesLabels"), corrAxesLabelList_);
    readVector ( cfg.read<std::string>( "effHistNames"  ), effHistNameList_  );
    readLabels ( cfg.read<std::string>( "effAxesLabels" ), effAxesLabelList_ );
    readVector ( cfg.read<std::string>( "cmpObjectLabels"), cmpObjLabelList_ );
  }
  catch(...){
    std::cerr << "ERROR during reading of config file" << std::endl;
    std::cerr << "      misspelled variables in cfg ?" << std::endl;
    std::cerr << "      [--called in configBlockSpecific--]" << std::endl;
    std::exit(1);
  }
}

TString 
CalibClosureTest::output(const char* pre, std::string& name, std::string& label, const char* post)
{
  TString buffer( output(pre, name, post) );
  buffer.Remove(buffer.Last('.'), buffer.Length());
  buffer+="_";
  buffer+=label;
  buffer+=".";
  buffer+=post;
  return buffer;
}

TString 
CalibClosureTest::output(const char* pre, std::string& name, const char* post)
{
  // prepare name of the output file
  TString buffer( name );
  if( ((TString)buffer(buffer.Last('_')+1, buffer.Length())).IsDigit() )
    buffer.Remove(buffer.Last('_'), buffer.Length()); // chop off everything before the 
  buffer.Remove(0, buffer.Last ('_')+1);              // second to last and last '_'
  TString output( writeTo_.c_str() );
  output+=pre;
  output+=buffer;
  output+= ".";
  output+=post;  
  return output;
}

void CalibClosureTest::drawEff()
{
  std::vector<TObjArray> effSampleList;
  loadHistograms(effHistNameList_, effSampleList);

  TCanvas *canv = new TCanvas("canv", "histograms", 600, 600);
  setCanvasStyle( *canv  );
  canv->SetGridx( 1 );
  canv->SetGridy( 1 );

  //---------------------------------------------
  // loop histograms via the list of histogram 
  // names stored in effHistNameList_, open a 
  // new page for each histogram; histograms are
  // expected in order ref(0)/sample(1),... 
  //---------------------------------------------
  for(unsigned int jdx=0; jdx<effHistNameList_.size()/2; ++jdx){
    TPostScript psFile(output("/inspectMatch_", effHistNameList_[2*jdx], "eps"), 113);
    psFile.NewPage();

    TLegend* leg = new TLegend(legXLeft_,legYLower_,legXRight_,legYUpper_);
    setLegendStyle( *leg ); 
    //-----------------------------------------------
    // loop each sample, draw all sample for a given 
    // histogram name in one plot 
    //-----------------------------------------------
    std::vector<TObjArray>::const_iterator hist = effSampleList.begin();
    for(unsigned int idx=0; hist!=effSampleList.end(); ++hist, ++idx){  
      TH1F& href = *((TH1F*)(*hist)[2*jdx]); //recieve histograms
      TH1F& hcmp = *((TH1F*)(*hist)[2*jdx+1]);
      setHistStyles( hcmp, jdx, idx );

      if(idx==0){
	// prepare axes labels
	char xstring[100], ystring[100];
	if( idx<effAxesLabelList_.size() )
	  sprintf(xstring, effAxesLabelList_[jdx].c_str(),  cmpObjLabelList_[1].c_str());
	sprintf(ystring, "match^{%s}/match^{%s}", cmpObjLabelList_[0].c_str(), cmpObjLabelList_[1].c_str());
	hcmp.GetXaxis()->SetTitle( xstring );
	hcmp.GetYaxis()->SetTitle( ystring );
	hcmp.SetMinimum(0.85*hcmp.GetMinimum());
	hcmp.SetMaximum(1.25*hcmp.GetMaximum());
	divideHistograms(hcmp, href, 1);
	hcmp.Draw();
      }
      else{
	divideHistograms(hcmp, href, 1);
	hcmp.Draw("same");
      }
      leg->AddEntry( &hcmp, legend( idx ).c_str(), "PL" );
    }
    leg->Draw( "same" );
    canv->RedrawAxis( );
    canv->Update( );
    psFile.Close();
    delete leg;
  }
  canv->Close();
  delete canv;
}

void CalibClosureTest::drawCorrel()
{
  std::vector<TObjArray> corrSampleList;
  loadHistograms(corrHistNameList_, corrSampleList);

  TCanvas *canv = new TCanvas("canv", "histograms", 600, 600);
  setCanvasStyle( *canv  );
  canv->SetGridx( 1 );
  canv->SetGridy( 1 );

  //---------------------------------------------
  // loop histograms via the list of histogram 
  // names stored in corrHistNameList_, open a 
  // new file for each histogram and each sample
  //---------------------------------------------
  for(unsigned int jdx=0; jdx<corrHistNameList_.size(); ++jdx){  
    //---------------------------------------------
    // loop all available samples, open a new file
    // for each sample
    //---------------------------------------------
    std::vector<TObjArray>::const_iterator hist = corrSampleList.begin();
    for(unsigned int idx=0; hist!=corrSampleList.end(); ++hist, ++idx){  
      TLegend* leg = new TLegend(legXLeft_,legYLower_,legXRight_,legYUpper_);
      setLegendStyle( *leg ); 
      
      TPostScript psFile(output("/inspectCorrel_", corrHistNameList_[jdx], outputLabelList_[idx], "eps"), 113);
      psFile.NewPage();
      
      TH2F& hcmp = *((TH2F*)(*hist)[jdx]);
      hcmp.SetLineColor  ( histColor_[idx] );
      hcmp.SetFillColor  ( histColor_[idx] );
      hcmp.SetMarkerColor( histColor_[idx] );
      char xstring[100], ystring[100];
      if( jdx<corrAxesLabelList_.size() ){
	sprintf(xstring, corrAxesLabelList_[jdx].c_str(), cmpObjLabelList_[0].c_str() );
	sprintf(ystring, corrAxesLabelList_[jdx].c_str(), cmpObjLabelList_[1].c_str() );
      }
      setAxesStyle( hcmp, xstring, ystring );
      leg->AddEntry( &hcmp, legend( idx ).c_str(), "FL" );
      hcmp.Draw("box");
      leg->Draw( "same" );
      canv->RedrawAxis( );
      canv->Update( );
      psFile.Close();
      delete leg;
    }
  }
  canv->Close();
  delete canv;
}

int main(int argc, char* argv[])
{
  setNiceStyle();
  gStyle->SetOptStat( 0 );
  gStyle->SetOptFit ( 0 );

  gStyle->SetStatColor(0);
  gStyle->SetStatBorderSize(0);
  gStyle->SetStatX(0.93);
  gStyle->SetStatY(0.93);
  gStyle->SetStatW(0.18);
  gStyle->SetStatH(0.18);

  if( argc<2 ){
    std::cerr << "ERROR:" 
	 << " Missing argument" << std::endl;
    return 1;
  }

  CalibClosureTest plots;
  //plots.setVerbose(true);
  plots.readConfig( argv[1] );
  plots.loadHistograms();
  if( !strcmp(plots.writeAs().c_str(), "ps") ){
    plots.fitAndDrawPs();
    plots.fillTargetHistograms();
    plots.drawPs();
  } else if( !strcmp(plots.writeAs().c_str(), "eps") ){
    plots.fitAndDrawEps();
    plots.fillTargetHistograms();
    plots.drawEps();
  } else{
    std::cerr << "ERROR:"
	 << " Unknown file format requested: "
	 << plots.writeAs() << std::endl; 
    return -1;
  } 
  plots.drawEff();
  plots.drawCorrel();

  std::cout << "works " << "thanx and GoodBye " << std::endl; 
  return 0;
}
