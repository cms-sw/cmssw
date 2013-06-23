#include "Validation/RecoJets/interface/FitHist.h"
#include "Validation/RecoJets/interface/CompMethods.h"
#include "Validation/RecoJets/interface/FitHist_fwd.h"
#include "Validation/RecoJets/interface/RootPostScript.h"

using namespace std;

void 
FitHist::configBlockFit(ConfigFile& cfg)
{
  //-----------------------------------------------
  // read all configurables defined in FitHisto-
  // grams from config file. Throw human readable 
  // exception when misspellings occure
  //-----------------------------------------------
  try{
    //-----------------------------------------------
    // histogram steering
    //-----------------------------------------------
    readVector( cfg.read<std::string>( "titleIndex"), axesIndex_);
    readLabels( cfg.read<std::string>( "xAxesFit" ), xAxesFit_ );
    readLabels( cfg.read<std::string>( "yAxesFit" ), yAxesFit_ );
    
    //-----------------------------------------------
    // histogram manipulations
    //-----------------------------------------------
    readVector( cfg.read<std::string>( "targetLabel" ), targetHistList_);
    fitFuncName_ = cfg.read<std::string>( "fitFunctionName" );
    fitFuncTitle_= cfg.read<std::string>( "fitFunctionTitle");
    fitFuncType_ = cfg.read<int>        ( "fitFunctionType" );
    fitFuncLowerBound_= cfg.read<double>( "fitLowerBound" );
    fitFuncUpperBound_= cfg.read<double>( "fitUpperBound" );
    evalType_ = cfg.read<int>( "evalType" );
  }
  catch(...){
    std::cerr << "ERROR during reading of config file" << std::endl;
    std::cerr << "      misspelled variables in cfg ?" << std::endl;
    std::cerr << "      [--called in configBlockFit]"  << std::endl;
    std::exit(1);
  }
}

bool 
FitHist::checkTargetHistList()
{
  //-----------------------------------------------
  // check list of target histogram labels, return 
  // true if all elements are supported and false 
  // otherwise
  //-----------------------------------------------
  bool statusOk=true;
  for(unsigned int idx=0; idx<targetHistList_.size(); ++idx){
    if( statusOk ) statusOk=isInFitTargetList(targetHistList_[idx]);
  }
  return statusOk;
}

bool 
FitHist::isInFitTargetList(std::string& label)
{
  //-----------------------------------------------
  // check if given target prefix is suported
  //-----------------------------------------------
  TString target(label);
  if( !(target.CompareTo(FitTarget::Cal   ) ||
        target.CompareTo(FitTarget::Res   ) ||
        target.CompareTo(FitTarget::Sigma ) ||
        target.CompareTo(FitTarget::Mean  ) )){
    std::cerr << "ERROR while filling target histogram" << std::endl;
    std::cerr << "      can't find prefix: "  << target << std::endl;
    return false;
  }
  return true;
}

TH1F& 
FitHist::findFitHistogram(const TObjArray& hist, TString& zip, TString& name, int& bin)
{
  //---------------------------------------------
  // returns fit histogram of bin 'bin'to 
  // corresponding target histogram
  //---------------------------------------------

  //prepare reference name
  TString refName( name );
  refName+="_";
  refName+=bin;
  refName.Remove(0, refName.First('_')+1); //chop of prefix
  
  //---------------------------------------------
  // loop array of histograms & search for ref
  //---------------------------------------------
  for(int idx=0; idx<hist.GetEntriesFast(); ++idx){
    TH1F& cmp = *((TH1F*)(hist)[idx]);
    TString buffer( cmp.GetName() );
    if( buffer.BeginsWith( zip ) ){
      TString cmpName( cmp.GetName() );        // chop off root directory and 
      cmpName.Remove(0, cmpName.First('/')+1); // file from histogram name  
      if( cmpName.BeginsWith(FitTarget::Fit) && cmpName.Contains(refName) ){
	return cmp;
      }
    }
  }
  std::cout << "WARNING: could not find required histogram fit_" 
       << refName << "_x" << std::endl
       << "         return reference to Null" << std::endl;
  return *(TH1F*)0;
}

TH1F& 
FitHist::findTargetHistogram(const TObjArray& hist, TString& zip, TString& name, TString& target)
{
  //---------------------------------------------
  // returns target histogram to corresponding 
  // fit histogram
  //---------------------------------------------

  //prepare reference name
  TString refName( name );
  refName.Remove(0, refName.First('_')+1); //chop of prefix
  refName.Remove(refName.Last('_'), refName.Length()); //chop of postfix

  //---------------------------------------------
  // loop array of histograms & search for ref
  //---------------------------------------------
  for(int idx=0; idx<hist.GetEntriesFast(); ++idx){
    TH1F& cmp = *((TH1F*)(hist)[idx]);
    TString buffer( cmp.GetName() );
    if( buffer.BeginsWith( zip ) ){
      // prepare compare string
      TString cmpName( cmp.GetName() );        // chop off root directory and 
      cmpName.Remove(0, cmpName.First('/')+1); // file from histogram name  
      if( cmpName.BeginsWith(target) && cmpName.Contains(refName) ){
	return cmp;
      }
    }
  }
  std::cout << "WARNING: could not find required histogram " 
       << target << refName << std::endl
       << "         return reference to Null" << std::endl;
  return *(TH1F*)0;
}

double 
FitHist::normalize(TString& buffer, double val)
{ 
  return ( (!buffer.CompareTo(FitTarget::Res ) && val>0.) ) ? 1./val : 1.; 
}

void 
FitHist::fillTargetHistogramBin(TH1F& htarget, TH1F& hfit, int bin, TString& buffer, Quantile& func)
{
  if( !buffer.CompareTo(FitTarget::Cal) || !buffer.CompareTo(FitTarget::Mean ) ){
    htarget.SetBinContent( bin, func.value( hfit ) ); htarget.SetBinError( bin, func.valueError( hfit ) );
  }
  if( !buffer.CompareTo(FitTarget::Res) || !buffer.CompareTo(FitTarget::Sigma) ){
    double norm=normalize(buffer, func.value( hfit ));
    htarget.SetBinContent( bin, norm*func.spread( hfit ) ); htarget.SetBinError  ( bin, norm*func.spreadError( hfit ) );
  }
}

void 
FitHist::fillTargetHistogramBin(TH1F& htarget, TH1F& hfit, int bin, TString& buffer, MaximalValue& func)
{
  if( !buffer.CompareTo(FitTarget::Cal) || !buffer.CompareTo(FitTarget::Mean ) ){
    htarget.SetBinContent( bin, func.value( hfit ) ); htarget.SetBinError( bin, func.valueError( hfit ) );
  }
  if( !buffer.CompareTo(FitTarget::Res) || !buffer.CompareTo(FitTarget::Sigma) ){
    double norm=normalize(buffer, func.value( hfit ));
    htarget.SetBinContent( bin, norm*func.spread( hfit ) ); htarget.SetBinError  ( bin, norm*func.spreadError( hfit ) );
  }
}

void 
FitHist::fillTargetHistogramBin(TH1F& htarget, TH1F& hfit, int bin, TString& buffer, HistogramMean& func)
{
  if( !buffer.CompareTo(FitTarget::Cal) || !buffer.CompareTo(FitTarget::Mean ) ){
    htarget.SetBinContent( bin, func.value( hfit ) ); htarget.SetBinError( bin, func.valueError( hfit ) );
  }
  if( !buffer.CompareTo(FitTarget::Res) || !buffer.CompareTo(FitTarget::Sigma) ){
    double norm=normalize(buffer, func.value( hfit ));
    htarget.SetBinContent( bin, norm*func.spread( hfit ) ); htarget.SetBinError  ( bin, norm*func.spreadError( hfit ) );
  }
}

void 
FitHist::fillTargetHistogramBin(TH1F& htarget, TH1F& hfit, int bin, TString& buffer, StabilizedGauss& func)
{
  if( !buffer.CompareTo(FitTarget::Cal) || !buffer.CompareTo(FitTarget::Mean ) ){
    htarget.SetBinContent( bin, func.value( hfit ) ); htarget.SetBinError( bin, func.valueError( hfit ) );
  }
  if( !buffer.CompareTo(FitTarget::Res) || !buffer.CompareTo(FitTarget::Sigma) ){
    double norm=normalize(buffer, func.value( hfit ));
    htarget.SetBinContent( bin, norm*func.spread( hfit ) ); htarget.SetBinError  ( bin, norm*func.spreadError( hfit ) );
  }
}

void 
FitHist::fillTargetHistogramBin(TH1F& htarget, TH1F& hfit, int bin)
{
  if( !hfit.GetEntries()>0 ) return;

  //-----------------------------------------------
  // fill corresponding bin in target histogram
  //-----------------------------------------------
  TString buffer(htarget.GetName());
  buffer.Remove(0, buffer.First('/')+1);
  buffer.Remove(buffer.First('_')+1, buffer.Length());
  
  switch(evalType_){
  case kStabilizedGauss:
    {
      StabilizedGauss func(fitFuncName_.c_str());
      fillTargetHistogramBin(htarget, hfit, bin, buffer, func);
      break;
    }
  case kHistogramMean:
    {
      HistogramMean func;
      fillTargetHistogramBin(htarget, hfit, bin, buffer, func);
      break;
    }
  case kMaximalValue:
    {
      MaximalValue func(0.9, 0.05);
      fillTargetHistogramBin(htarget, hfit, bin, buffer, func);
      break;
    }
  case kQuantile:
    {
      Quantile func(0.5, 0.05);
      fillTargetHistogramBin(htarget, hfit, bin, buffer, func);
      break;
    } 
  default:
    
    break;
  }
}

void 
FitHist::setFitHistogramAxes(TH1F& hist, int idx)
{
  //-----------------------------------------------
  // get proper axes label for fit histograms; get
  // the proper histogram idx from int the vector
  // axesIndex_, as the same label will be used
  // many times for each fit bin thi saves from 
  // repeatedly passing the same labels.
  //-----------------------------------------------
  if(idx<(int)axesIndex_.size()){
    int jdx = axesIndex_[idx];
    if( jdx<(int)xAxesFit_.size() && jdx<(int)yAxesFit_.size() ){
      setAxesStyle( hist, xAxesFit_[jdx].c_str(), yAxesFit_[jdx].c_str() );
    }
    else if( jdx<(int)xAxesFit_.size() ){
      setAxesStyle( hist, xAxesFit_[jdx].c_str(), "events"  );
    }
  }
  else{
    setAxesStyle( hist, hist.GetName(), "events" );	
  }
}

void 
FitHist::addBinLabelToFitHist(const TObjArray& hist, int& bin, TString& name, TString& zip)
{
  //---------------------------------------------
  // add pave text to fitted histograms keeping 
  // information of the corresponding bin edges
  //---------------------------------------------
  TPaveText* text  = new TPaveText(0.25, 0.85, 0.95, 0.95, "NDC");
  text->SetBorderSize(   0 );
  text->SetFillStyle( 4000 );
  text->SetTextAlign(   12 );
  text->SetTextSize ( 0.06 );
  text->SetTextColor(    1 );
  text->SetTextFont (   62 );

  char labelname[100];
  TString buffer(targetHistList_[0]); buffer+="_";
  TH1F&  target = findTargetHistogram(hist, zip, name, buffer);
  double lowerEdge = target.GetBinLowEdge(bin+1);
  double upperEdge = target.GetBinLowEdge(bin+1)+target.GetBinWidth(bin+1);
  sprintf(labelname, "[%3.0f GeV; %3.0f GeV]", lowerEdge, upperEdge );//FIXME: this does not need to be GeV only
  text->AddText( labelname );
  text->Draw();
}

void
FitHist::addParLabelToFitHist(const TH1F& hist)
{
  if( hist.GetFunction(fitFuncName_.c_str()) ){
    TPaveText* pars  = new TPaveText(0.40, 0.55, 0.95, 0.70, "NDC");
    pars->SetBorderSize(   0 );
    pars->SetFillStyle( 4000 );
    pars->SetTextAlign(   12 );
    pars->SetTextSize ( 0.06 );
    pars->SetTextColor(    1 );
    pars->SetTextFont (   62 );
   
    char parstring[100]; 
    if(fitFuncType_==0){
      sprintf(parstring, "#mu=%3.2f ; #sigma=%3.2f",  
	      hist.GetFunction(fitFuncName_.c_str())->GetParameter( 1 ), 
	      hist.GetFunction(fitFuncName_.c_str())->GetParameter( 2 ) );
    }
    pars->AddText( parstring );
    pars->Draw();
  }
}

void 
FitHist::fitAndDrawPs()
{
  //-----------------------------------------------
  // define canvas
  //-----------------------------------------------
  TCanvas *canv = new TCanvas("canv", "fit histograms", 600, 600);
  setCanvasStyle( *canv );
  canv->SetGridx( 1 ); canv->SetLogx ( 0 );
  canv->SetGridy( 1 ); canv->SetLogy ( 0 );

  //-----------------------------------------------
  // loop histograms via the list of histogram
  // names stored in histList_, open a new page 
  // for each histogram; open a new file for each 
  // sample.
  //-----------------------------------------------
  std::vector<TObjArray>::const_iterator hist = sampleList_.begin();
  for(int idx=0; hist!=sampleList_.end(); ++hist, ++idx){
    //-----------------------------------------------
    // open output file
    //-----------------------------------------------
    TString output( writeTo_.c_str() );
    output += "/inspectFit_";
    if(outputLabelList_.size()>=sampleList_.size())
      output += outputLabelList_[idx];
    else
      output == idx;
    output += ".ps";
    TPostScript psFile( output, 111 ); //112 for portrat
    //---------------------------------------------
    // loop histograms via the list of histogram 
    // names stored in histList_, open a new page 
    // for each histogram; fit any histogram con-
    // taining the prefix 'fit'
    //---------------------------------------------
    int label=0;
    for(int jdx=0; jdx<(*hist).GetEntriesFast(); ++jdx){
      TH1F& hfit = *((TH1F*)(*hist)[jdx]); //recieve histogram
      // prepare compare string
      TString cmp( hfit.GetName() );            // chop off root directory and 
      cmp.Remove(0, cmp.First('/')+1);          // file from histogram name  
      //prepare zip code
      TString zip( hfit.GetName() );            // chop off histogram name keep 
      zip.Remove(zip.First('/'), zip.Length()); // file and directory
      if ( cmp.BeginsWith(FitTarget::Fit) ){ //search for prefix 'fit'
	psFile.NewPage();
	if(verbose_){
	  std::cout << std::endl << "about to fit histogram: " 
	       << hfit.GetName() << " as " << cmp << std::endl; 
	}
	//-----------------------------------------------
	// determine proper bin by choping off everything
	// but the postfix which should correspond to the 
	// bin of NameScheme
	//-----------------------------------------------
	TString buffer( cmp );
 	int bin = buffer.Remove(0, buffer.Last('_')+1).Atoi();

	hfit.SetLineWidth( 5 );
	hfit.SetLineColor( 4 );
	hfit.SetLineStyle( 1 );
	hfit.SetMarkerStyle( 20 );
	hfit.SetMarkerSize( 2.0 );
	hfit.SetMarkerColor(  4 );
	hfit.SetMaximum(1.5*hfit.GetMaximum());
	setFitHistogramAxes( hfit, label++ );
	//hfit.Fit( "gaus" );

	//do stable fit
	StabilizedGauss func(fitFuncName_.c_str(), fitFuncType_, fitFuncLowerBound_, fitFuncUpperBound_);
	func.fit(hfit);

	hfit.Draw("esame");
	//---------------------------------------------
	// add TPaveLabel to keep info of binning
	// recieved from target hitogram
	//---------------------------------------------
	addParLabelToFitHist( hfit );	
	addBinLabelToFitHist(*hist, bin, cmp, zip);

	//---------------------------------------------
	// add legend to fitted histogram
	//---------------------------------------------
	TLegend* leg = new TLegend(0.25,0.70,0.95,0.87);
	setLegendStyle( *leg );
	leg->AddEntry( &hfit, legend(idx).c_str(), "PL" );
	leg->AddEntry( hfit.GetFunction(fitFuncName_.c_str()), fitFuncTitle_.c_str(),  "L" );
	leg->Draw( "same" );
	canv->RedrawAxis( );
	canv->Update( );
	if(jdx<((*hist).GetEntriesFast()-1)) delete leg;
      }
    }
    psFile.Close();
  }
  canv->Close();
  delete canv;
}

void 
FitHist::fitAndDrawEps()
{
  //-----------------------------------------------
  // define canvas
  //-----------------------------------------------
  TCanvas *canv = new TCanvas("canv", "fit histograms", 600, 600);
  setCanvasStyle( *canv );
  canv->SetGridx( 1 ); canv->SetLogx ( 0 );
  canv->SetGridy( 1 ); canv->SetLogy ( 0 );

  //-----------------------------------------------
  // loop histograms via the list of histogram
  // names stored in histList_, open a new page 
  // for each histogram; open a new file for each 
  // sample.
  //-----------------------------------------------
  std::vector<TObjArray>::const_iterator hist = sampleList_.begin();
  for(int idx=0; hist!=sampleList_.end(); ++hist, ++idx){
    //---------------------------------------------
    // loop histograms via the list of histogram 
    // names stored in histList_, open a new page 
    // for each histogram; fit any histogram con-
    // taining the prefix 'fit'
    //---------------------------------------------
    int label=0;
    for(int jdx=0; jdx<(*hist).GetEntriesFast(); ++jdx){
      TH1F& hfit = *((TH1F*)(*hist)[jdx]); //recieve histogram
      // prepare compare string
      TString cmp( hfit.GetName() );            // chop off root directory and 
      cmp.Remove(0, cmp.First('/')+1);          // file from histogram name  
      //prepare zip code
      TString zip( hfit.GetName() );            // chop off histogram name keep 
      zip.Remove(zip.First('/'), zip.Length()); // file and directory
      if ( cmp.BeginsWith(FitTarget::Fit) ){ //search for prefix 'fit'
	//-----------------------------------------------
	// open output file
	//-----------------------------------------------
	TString output( writeTo_.c_str() );
	output += "/";
	output += histList_[ jdx ];
	output += "_";
	if(outputLabelList_.size()>=sampleList_.size())
	  output += outputLabelList_[idx];
	else
	  output == idx;
	output += ".eps";
	TPostScript psFile( output, 113 );
	psFile.NewPage();
	
	//-----------------------------------------------
	// determine proper bin by choping off everything
	// bit the postfix which should correspond to the 
	// bin of NameScheme
	  //-----------------------------------------------
	TString buffer( cmp );
	int bin = buffer.Remove(0, buffer.Last('_')+1).Atoi();
	
	hfit.SetLineWidth( 5 );
	hfit.SetLineColor( 4 );
	hfit.SetLineStyle( 1 );
	hfit.SetMarkerStyle( 20 );
	hfit.SetMarkerSize( 2.0 );
	hfit.SetMarkerColor(  4 );
	setFitHistogramAxes( hfit, label++ );
	//hfit.Fit( "gaus" );
	
	//do stable fit
	StabilizedGauss func(fitFuncName_.c_str(), fitFuncType_, fitFuncLowerBound_, fitFuncUpperBound_);
	func.fit(hfit);
	
	hfit.Draw("esame");
	//---------------------------------------------
	// add TPaveLabel to keep info of binning
	// recieved from target hitogram
	//---------------------------------------------
	addParLabelToFitHist( hfit );	
	addBinLabelToFitHist( *hist, bin, cmp, zip );
	
	//---------------------------------------------
	// add legend to fitted histogram
	//---------------------------------------------
	TLegend* leg = new TLegend(0.25,0.70,0.95,0.87);
	setLegendStyle( *leg );
	leg->AddEntry( &hfit, legend(idx).c_str(), "PL" );
	leg->AddEntry( hfit.GetFunction(fitFuncName_.c_str()), fitFuncTitle_.c_str(),  "L" );
	leg->Draw( "same" );
	canv->RedrawAxis( );
	canv->Update( );
	psFile.Close();
	delete leg;
      }
    }
  }
  canv->Close();
  delete canv;
}

void 
FitHist::fillTargetHistograms()
{
  //---------------------------------------------
  // fill all target histograms 
  //---------------------------------------------
  for(unsigned int idx=0; idx<targetHistList_.size(); ++idx){
    fillTargetHistogram(targetHistList_[idx]);
  }
}

void 
FitHist::fillTargetHistogram(std::string& target)
{
  //---------------------------------------------
  // loop array of samples via sampleList_, which 
  // contains all histograms in TObjectArrays for 
  // each sample
  //---------------------------------------------
  std::vector<TObjArray>::const_iterator hist = sampleList_.begin();
  for(; hist!=sampleList_.end(); ++hist){
    //---------------------------------------------
    // loop histograms & search for prefix target;
    // check if it's contained in list of valid 
    // targets first
    //---------------------------------------------
    TString buffer( target );
    if( isInFitTargetList(target) ){
      for(int jdx=0; jdx<(*hist).GetEntriesFast(); ++jdx){
	TH1F& htarget = *((TH1F*)(*hist)[jdx]);
	// prepare compare string
	TString cmp( htarget.GetName() );         // chop off root directory and 
	cmp.Remove(0, cmp.First('/')+1);          // file from histogram name  
	//prepare zip code
	TString zip( htarget.GetName() );         // chop off histogram name keep 
	zip.Remove(zip.First('/'), zip.Length()); // file and directory
	if( cmp.BeginsWith( buffer ) ){ //found target hist
	  //---------------------------------------------
	  // now fill the bins of the target hist
	  //---------------------------------------------
	  for(int kdx=0; kdx<htarget.GetNbinsX(); ++kdx){
	    TH1F& hfit = findFitHistogram(*hist, zip, cmp, kdx);
	    fillTargetHistogramBin(htarget, hfit, (kdx+1));
	  }
	}
      }
    }
  }
}

void 
FitHist::writeFitOutput()
{
  //---------------------------------------------
  // write filled target histogram to output file
  // if specified
  //---------------------------------------------
  if( isOutputRequested() ){
    //-----------------------------------------------
    // open hist file for book keeping of hist names
    //-----------------------------------------------
    TString name( output_ );
    name.Remove(name.Last('.'), name.Length()); 
    name+=".hist"; //replace .root by .hist
    ofstream histFile(name, std::ios::out);

    //-----------------------------------------------
    // open root output file and create directory
    // if necessary
    //-----------------------------------------------
    TFile file( output_.c_str(), "update" );
    if( !file.GetDirectory(rootOutDir_.c_str()) ) 
      // create new directory if it does not yet exist
      file.mkdir( rootOutDir_.c_str(), rootOutDir_.c_str() );
    else 
      // clean-up directory if it was existing already 
      (file.GetDirectory(rootOutDir_.c_str()))->Delete("*;*");
    file.cd(rootOutDir_.c_str());
    
    // loop over requested target histogram labels if desired
    for(unsigned int jdx=0; jdx<targetHistList_.size(); ++jdx){
      TString buffer( targetHistList_[jdx] ); 
      //-----------------------------------------------
      // loop and write out histograms
      //-----------------------------------------------
      std::vector<TObjArray>::const_iterator hist = sampleList_.begin();
      for( ;hist!=sampleList_.end(); ++hist){
	for(int idx=0; idx<(int)histList_.size(); ++idx){  
	  TString cmp( ((TH1F*)(*hist)[idx])->GetName() );
	  if( cmp.BeginsWith( buffer ) ){ //found target hist
	    histFile << ((TH1F*)(*hist)[idx])->GetName() << "\n";
	    ((TH1F*)(*hist)[idx])->Write();
	  }
	} // end of histList_ loop
      }
    } // end of targetHistList_ loop
    file.Close();
  }
}
