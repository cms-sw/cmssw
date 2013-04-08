#include "Validation/RecoJets/interface/CompHist.h"
#include "Validation/RecoJets/interface/ConfigFile.h"
#include "Validation/RecoJets/interface/RootPostScript.h"


using namespace std;

void 
CompHist::readLabels(std::string s, std::vector<std::string>& vec)
{
  //-----------------------------------------------
  // fill vector of std::string's from a single
  // std::string s; the process starts as soon as 
  // leading " are found and starts a new substr
  // as soon as a "; is encountered
  //-----------------------------------------------
  std::stringstream stream( s );
  std::string buffer, label;
  while (!stream.eof()) {
    stream >> buffer;
    if(buffer.find("\"", 0)==0){
      //start new label if leading " are found
      label=buffer;
    }
    else{
      //concatenate buffer to label else
      label+=" ";
      label+=buffer;
    }

    //push_back label if it starts with " and ends with ";
    if(buffer.find("\"", buffer.size()-2)==buffer.size()-2 &&
       buffer.find(";",  buffer.size()-2)==buffer.size()-1){
      vec.push_back( label.substr(1, label.size()-3) );
    }
  }
}

void 
CompHist::configBlockIO(ConfigFile& cfg)
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
    histFile_  = cfg.read<std::string>( "histInput" );
    readVector ( cfg.read<std::string>( "rootInput" ), fileNameList_ );
    readVector ( cfg.read<std::string>( "inputDirs" ), dirNameList_  );
    filterOpt_ = cfg.read<std::string>( "filterOption" );
    readVector ( cfg.read<std::string>( "histFilter" ), histFilterList_ );
    readVector ( cfg.read<std::string>( "plotFilter" ), plotFilterList_ );
    writeTo_   = cfg.read<std::string>( "writePlotsTo" );
    writeAs_   = cfg.read<std::string>( "writePlotsAs" );
    rootOutDir_= cfg.read<std::string>( "outputDir" );
    output_    = cfg.read<std::string>( "rootOutput" );
    readVector ( cfg.read<std::string>( "outputLabels" ), outputLabelList_ );
  }
  catch(...){
    std::cerr << "ERROR during reading of config file" << std::endl;
    std::cerr << "      misspelled variables in cfg ?" << std::endl;
    std::cerr << "      [--called in configBlockIO--]" << std::endl;
    std::exit(1);
  }
}

void
CompHist::configBlockHist(ConfigFile& cfg)
{
  //-----------------------------------------------
  // read all configurables defined in CompHisto-
  // grams from config file. Throw human readable 
  // exception when misspellings occure
  //-----------------------------------------------
  try{
    //-----------------------------------------------
    // canvas steering
    //-----------------------------------------------
    readVector( cfg.read<std::string>( "xLog" ), logX_ );
    readVector( cfg.read<std::string>( "yLog" ), logY_ );
    readVector( cfg.read<std::string>( "xGrid" ), gridX_);
    readVector( cfg.read<std::string>( "yGrid" ), gridY_);

    //-----------------------------------------------
    // histogram steering
    //-----------------------------------------------
    readVector( cfg.read<std::string>( "histErrors" ), errors_);
    readVector( cfg.read<std::string>( "histScale" ), scale_ );
    readVector( cfg.read<std::string>( "histMinimum"), min_ );
    readVector( cfg.read<std::string>( "histMaximum"), max_ );
    readVector( cfg.read<std::string>( "histType" ), histStyle_ );
    readVector( cfg.read<std::string>( "histStyle" ), commonStyle_ );
    readVector( cfg.read<std::string>( "histColor" ), histColor_ );
    readVector( cfg.read<std::string>( "lineWidth" ), commonWidth_ );
    readVector( cfg.read<std::string>( "markerStyle" ), markerStyle_ );
    readVector( cfg.read<std::string>( "markerSize" ), markerSize_ );
    readLabels( cfg.read<std::string>( "xAxes" ), xAxes_ );
    readLabels( cfg.read<std::string>( "yAxes" ), yAxes_ );

    //-----------------------------------------------
    // legend steering
    //-----------------------------------------------
    readLabels( cfg.read<std::string>( "legEntries"), legendEntries_);
    legXLeft_ = cfg.read<double>( "legXLeft" );
    legXRight_= cfg.read<double>( "legXRight" );
    legYLower_= cfg.read<double>( "legYLower" );
    legYUpper_= cfg.read<double>( "legYUpper" );
  }
  catch(...){
    std::cerr << "ERROR during reading of config file"   << std::endl;
    std::cerr << "      misspelled variables in cfg ?"   << std::endl;
    std::cerr << "      [--called in configBlockHist--]" << std::endl;
    std::exit(1);
  }
}

void 
CompHist::readHistogramList()
{
  //-----------------------------------------------
  // fill the list histList_ with all requested
  // histogram names; the names are recieved from 
  // a .hist file; jump out if the reading takes 
  // too long 
  //-----------------------------------------------
  ifstream histFile( histFile_.c_str() );

  int count=0;
  while( !histFile.eof() ){
    std::string buffer;
    histFile >> buffer;
    if( buffer.size()>0 ){
      TString cmp(buffer);
      if( histFilter(cmp, histFilterList_) )
	histList_.push_back(buffer);
      else{
	if(verbose_){
	  std::cout << " histogram is filtered out according"
	       << " to settings in cfg file; filterOpt:  "
	       << filterOpt_ << std::endl;
	}
      }
    }
    if( count>999 ){
      std::cerr << "ERROR caught in slope for histogram" << std::endl;
      std::cerr << "      names. Misspelled file name ?" << std::endl;
      std::cerr << histFile_.c_str() << std::endl;
      break;
    }
    ++count;
  }
}

void
CompHist::loadHistograms(std::vector<std::string>& histList, std::vector<TObjArray>& sampleList)
{
  //fill vector of root files if not already done
  if( fileList_.empty() ){
    std::vector<std::string>::const_iterator name;
    for(name = fileNameList_.begin(); name!=fileNameList_.end(); ++name){
      fileList_.push_back( new TFile( name->c_str() ) );
    }
  }
  //fill vector of histogram arrays
  std::vector<TFile*>::const_iterator file = fileList_.begin();
  for(int jdx=0 ; file!=fileList_.end(); ++file, ++jdx){
    std::vector<std::string>::const_iterator rootDir;
    for(rootDir = dirNameList_.begin(); rootDir!=dirNameList_.end(); ++rootDir){
      TH1F *dummy;
      TObjArray hist;
      for( int idx=0; idx<(int)histList.size(); ++idx ){
	TString directory( *rootDir );
	directory+="/";
	directory+=histList[idx];
	
	if(verbose_){
	  std::cout << "load from: " << (*file)->GetName() << " " 
	       << "directory: " << directory << std::endl;
	}
	dummy = (TH1F*)(*file)->Get( directory );
	if( !dummy ){
	  std::cerr << "WARNING:" 
	       << " Didn't find indicated hist"
	       << " [" << directory << "]" 
	       << " in file" 
	       << " [" << (*file)->GetName() << "]" 
	       << std::endl;
	  return;
	}

	// prepare unambiguous histogram name
	TString outname( fileNameList_[jdx] );
	outname.Remove(outname.Last('.'), outname.Length());
	outname += ":";
	outname += *rootDir;
	outname +="/";
	outname +=dummy->GetName();
	dummy->SetName( outname );

	// add histogram to list
	hist.Add( dummy );
	if(verbose_){
	  std::cout << "added to list as: " << dummy->GetName() << std::endl;
	}
      } // end of histogram name list 
      sampleList.push_back( hist );
    } // end of root directory list
  } // end of root file list
}

void 
CompHist::loadHistograms()
{
  //-----------------------------------------------
  // fill histograms listed in histList_ into a
  // TObjectArray for each sample given in the 
  // list fileNameList_ and its directories; Each 
  // TObjectArray is then pushed into the vector 
  // sampleList_ which keeps all TObjectArrays of 
  // histograms for each sample 
  //-----------------------------------------------

  //fill histogram list
  if( !histFile_.empty() )
    readHistogramList();
  else{
    if( histFilterList_.size()>0 ){
      if( !filterOpt_.compare("contains") )
	cout << "WARNING: missing hist file in cfg file; will use list of histogram" << std::endl
	     << "         names given in histFilter instead, requires filterOption " << std::endl
	     << "         'optains'; will move filterOption to 'optain' to go on..." << std::endl;
      filterOpt_="contains";
      histList_ =histFilterList_;
    }
  }
  loadHistograms(histList_, sampleList_);
}

bool
CompHist::histFilter(TString& cmp, std::vector<std::string>& ref)
{
  bool contained=true;
  if(!filterOpt_.empty()){
    contained=false;
    if(!filterOpt_.compare("begins")){
      contained = histFilter(cmp, ref, kBeginsWith);
    }
    else if(!filterOpt_.compare("ends")){
      contained = histFilter(cmp, ref, kEndsWith);
    }
    else if(!filterOpt_.compare("contains")){
      contained = histFilter(cmp, ref, kContains);
    }
    else{
      std::cerr << "ERROR during histogram filtering   " << std::endl;
      std::cerr << "      filter option is not properly" << std::endl;
      std::cerr << "      specified : " << filterOpt_ << std::endl;
    }
  }
  return contained;
}

bool 
CompHist::histFilter(TString& cmp, std::vector<std::string>& ref, CompHist::HistFilter option)
{
  bool contained=true;
  if(ref.size()>0){
    contained=false;
    for(unsigned int idx=0; idx<ref.size(); ++idx){
      TString buffer(ref[idx]);
      if( !option==kContains ) buffer+="_";
      if( option==kBeginsWith && cmp.BeginsWith(buffer) ){ 
	contained=true;
	break;
      }
      if( option==kEndsWith && cmp.EndsWith(buffer) ){
	contained=true;
	break;
      }
      if( option==kContains && cmp.Contains(buffer) ){
	contained=true;
	break;
      }
    }
  }
  return contained;
}

void
CompHist::draw(TCanvas& canv, TLegend& leg, int& idx, int& jdx)
{  
  //-----------------------------------------------
  // loop all samples via the list sampleList_, which 
  // containas the histograms of each sample as 
  // TObjects in TObjectArrays
  //-----------------------------------------------
  TH1F hfirst; //draw first histogram on top of others
               //after all histograms have been drawn
  std::vector<TObjArray>::const_iterator hist = sampleList_.begin();
  for(int kdx=0; hist!=sampleList_.end(); ++hist, ++kdx){
    TH1F& hcmp = *((TH1F*)(*hist)[idx]); //recieve histogram
    setCanvLog( canv, jdx );
    setCanvGrid( canv, jdx );
    setHistStyles( hcmp, jdx, kdx );
    // for the first histogram just draw
    // for the following ones draw same
    if(kdx==0){
      hfirst = hcmp; // buffer first histogram to redraw it after all
      if(errors_[kdx]) 
	hcmp.Draw("e");
      else 
	hcmp.Draw(   );
    }
    else{
      if(errors_[kdx]) 
	hcmp.Draw("samee");
      else 
	hcmp.Draw("same" );
    }
    // add legend entry in appropriate format
    switch( histStyle_[kdx]){
    case HistStyle::Line:
      leg.AddEntry( &hcmp, legend(kdx).c_str(), "L"  );
      break;
      
    case HistStyle::Marker:
      leg.AddEntry( &hcmp, legend(kdx).c_str(), "PL" );
      break;
      
    case HistStyle::Filled:
      leg.AddEntry( &hcmp, legend(kdx).c_str(), "FL" );
      break;
    }
  }
  if(errors_[0]){
    hfirst.Draw("esame");
  }
  else{
    hfirst.Draw( "same"); 
  }
  leg.Draw( "same" );
  canv.RedrawAxis( );
  canv.Update( );
}

void 
CompHist::drawPs()
{
  //-----------------------------------------------
  // define canvas
  //-----------------------------------------------
  TCanvas *canv = new TCanvas("canv", "histograms", 600, 600);
  setCanvasStyle( *canv  );

  //-----------------------------------------------
  // open output file
  //-----------------------------------------------
  TString output( writeTo_.c_str() );
  output += "/";
  output += "inspect";
  output += ".";
  output += writeAs_;
  TPostScript psFile( output, 111); //112 for portrait

  //-----------------------------------------------
  // loop histograms via the list of histogram
  // names stored in histList_, open a new page 
  // for each histogram & plot each sample in 
  // the same canvas
  //-----------------------------------------------
  for(int idx=0, jdx=0; idx<(int)histList_.size(); ++idx){
    // prepare compare string for plot filtering
    TString cmp( histList_[idx] );
    if( !histFilter(cmp, plotFilterList_) ){
      if(verbose_){
	cout << " event is filtered out according to"
	     << " settings in cfg file; filterOpt:  "
	     << filterOpt_ 
	     << " hist: " << cmp << std::endl;
      }
      continue;
    }
    psFile.NewPage();
    //-----------------------------------------------
    // on each page the legend needs to be redeclared
    //-----------------------------------------------   
    TLegend* leg = new TLegend(legXLeft_,legYLower_,legXRight_,legYUpper_);
    setLegendStyle( *leg );  
    draw(*canv, *leg, idx, jdx);
    if(idx == (int)histList_.size()-1){
      psFile.Close();
    }
    ++jdx; delete leg;
  }
  canv->Close();
  delete canv;
}

void 
CompHist::drawEps()
{
  //-----------------------------------------------
  // define canvas
  //-----------------------------------------------
  TCanvas *canv = new TCanvas("canv", "histograms", 600, 600);
  setCanvasStyle( *canv  );

  //-----------------------------------------------
  // loop histograms via the list of histogram
  // names stored in histList_, open a new page 
  // for each histogram & plot each sample in 
  // the same canvas
  //-----------------------------------------------  
  for(int idx=0, jdx=0; idx<(int)histList_.size(); ++idx){
    //-----------------------------------------------
    // prepare compare string for plot filtering
    //-----------------------------------------------
    TString cmp( histList_[idx] );
    if( !histFilter(cmp, plotFilterList_) ){
      if(verbose_){
	cout << " event is filtered out according to"
	     << " settings in cfg file; filterOpt:  "
	     << filterOpt_ 
	     << " hist: " << cmp << std::endl;
      }
      continue;
    }
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
    //-----------------------------------------------
    // on each page the legend needs to be redeclared
    //-----------------------------------------------   
    TLegend* leg = new TLegend(legXLeft_,legYLower_,legXRight_,legYUpper_);
    setLegendStyle( *leg ); 
    draw(*canv, *leg, idx, jdx);
    psFile.Close();
    ++jdx; delete leg;
  }
  canv->Close();
  delete canv;
}

std::string 
CompHist::legend(int idx)
{
  char buffer[100];
  sprintf(buffer, "undefined sample %i", idx);
  if( legendEntries_.size()>=sampleList_.size() ){
    return legendEntries_[idx];
  }
  return buffer;
}

double
CompHist::findMaximum(int idx)
{
  double max=-1.;
  for(std::vector<TObjArray>::const_iterator hist = sampleList_.begin(); 
      hist!=sampleList_.end(); ++hist){
    TH1F& hcmp = *((TH1F*)(*hist)[idx]); //recieve histogram
    if( max<0 || hcmp.GetMaximum()>max ) max=hcmp.GetMaximum();
  }
  return max;
}

void 
CompHist::setLegendStyle(TLegend& leg)
{
  leg.SetFillStyle ( 0 );
  leg.SetFillColor ( 0 );
  leg.SetBorderSize( 0 ); 
}

void 
CompHist::setCanvasStyle(TCanvas& canv)
{
  canv.SetFillStyle   ( 4000 );
  canv.SetLeftMargin  ( 0.20 );
  canv.SetRightMargin ( 0.05 );
  canv.SetBottomMargin( 0.15 );
  canv.SetTopMargin   ( 0.05 );
}

void 
CompHist::setCanvLog(TCanvas& canv, int idx)
{
  if( idx<((int)logX_.size()-1) && idx<((int)logY_.size()-1) ){
    canv.SetLogx ( logX_[idx]  );
    canv.SetLogy ( logY_[idx]  );
  }
  else if( idx<((int)logY_.size()-1) ){
    canv.SetLogx ( 0 );
    canv.SetLogy ( logY_[idx]  );
  }
  else{
    canv.SetLogx ( 0 );
    canv.SetLogy ( 0 );
  }
}

void 
CompHist::setCanvGrid(TCanvas& canv, int idx)
{
  if( idx<((int)gridX_.size()-1) && idx<((int)gridY_.size()-1) ){
    canv.SetGridx ( gridX_[idx]  );
    canv.SetGridy ( gridY_[idx]  );
  }
  else if( idx<((int)gridY_.size()-1) ){
    canv.SetGridx ( 0 );
    canv.SetGridy ( gridY_[idx]  );
  }
  else{
    canv.SetGridx ( 0 );
    canv.SetGridy ( 0 );
  }
}

void 
CompHist::setAxesStyle( TH1& hist, const char* titleX, const char* titleY )
{
  hist.SetTitle( "" );

  hist.GetXaxis()->SetTitle( titleX );
  hist.GetXaxis()->SetTitleSize  ( 0.06 );
  hist.GetXaxis()->SetTitleColor (    1 );
  hist.GetXaxis()->SetTitleOffset(  1.0 );
  hist.GetXaxis()->SetTitleFont  (   62 );
  hist.GetXaxis()->SetLabelSize  ( 0.05 );
  hist.GetXaxis()->SetLabelFont  (   62 );
  hist.GetXaxis()->CenterTitle   (      );
  hist.GetXaxis()->SetNdivisions (  505 );
  
  hist.GetYaxis()->SetTitle( titleY );
  hist.GetYaxis()->SetTitleSize  ( 0.07 );
  hist.GetYaxis()->SetTitleColor (    1 );
  hist.GetYaxis()->SetTitleOffset(  1.3 );
  hist.GetYaxis()->SetTitleFont  (   62 );
  hist.GetYaxis()->SetLabelSize  ( 0.05 );
  hist.GetYaxis()->SetLabelFont  (   62 );
}

void 
CompHist::setHistStyles( TH1F& hist, int idx, int jdx )
{
  //-----------------------------------------------
  // check hist style; throw exception if style 
  // is not competible with specifications; set 
  // default line[0] if vector is too short
  //-----------------------------------------------
  int histType=0;
  if( jdx<((int)histStyle_.size()) ){
    if(HistStyle::Line<=histStyle_[jdx] && histStyle_[jdx]<=HistStyle::Filled){
      histType=histStyle_[jdx];
    }
    else{
      throw "Histogram Type cannot be specified ";
    }
  }
  
  //define histogram styles
  setHistLabels( hist, idx );
  setHistScale ( hist, idx );
  setHistMax   ( hist, idx );
  setHistMin   ( hist, idx );

  switch( histType ){
  case HistStyle::Line: 
    setLineWidth( hist, jdx );
    setLineColor( hist, jdx );
    setLineStyle( hist, jdx );
    break;

  case HistStyle::Marker:
    setLineWidth  ( hist, jdx );
    setLineColor  ( hist, jdx );
    setMarkerColor( hist, jdx );
    setMarkerStyle( hist, jdx ); 
    setMarkerSize ( hist, jdx );    
    break;

  case HistStyle::Filled:
    setLineWidth( hist, jdx );
    setLineColor( hist, jdx );
    setFillColor( hist, jdx );
    setFillStyle( hist, jdx );    
    break;
  }
}

void 
CompHist::setHistLabels(TH1F& hist, int idx)
{
  if( idx<((int)xAxes_.size()) && idx<((int)yAxes_.size()) ){
    setAxesStyle( hist, xAxes_[idx].c_str(), yAxes_[idx].c_str() );
  }
  else if( idx<((int)xAxes_.size()) ){
    setAxesStyle( hist, xAxes_[idx].c_str(), "events" );
  }
  else{
    if( strcmp(hist.GetTitle(),"") == 0 ){
      setAxesStyle( hist, hist.GetXaxis()->GetTitle(), "events" );  
    }
    else{
      setAxesStyle( hist, hist.GetTitle(), "events" );		  
    }    	
  }
}

void 
CompHist::setHistScale(TH1F& hist, int idx)
{
  if( idx<((int)scale_.size()) ){
    if( scale_[idx]>0 ) hist.Scale(scale_[idx]/hist.Integral());
  }
}

void 
CompHist::setHistMax(TH1F& hist, int idx)
{
  if( ((int)max_.size()>0) && (idx<(int)max_.size()) ){
    hist.SetMaximum(max_[idx]);
  }
  else{
    hist.SetMaximum(1.5*findMaximum(idx));
  }
}

void 
CompHist::setHistMin(TH1F& hist, int idx)
{
  if( ((int)min_.size()>0) && (idx<(int)min_.size()) ){
    hist.SetMinimum(min_[idx]);
  }
  else{
    hist.SetMinimum(0.);    
  }
}

void 
CompHist::setLineWidth(TH1F& hist, int jdx)
{
  hist.SetLineWidth( 5 );
  if( jdx<((int)commonWidth_.size()) ){
    hist.SetLineWidth(commonWidth_[jdx]);
  }
}

void 
CompHist::setLineStyle(TH1F& hist, int jdx)
{
  hist.SetLineStyle( 1 );
  if( jdx<((int)commonStyle_.size()) ){
    hist.SetLineStyle(commonStyle_[jdx]);
  }
}

void 
CompHist::setLineColor(TH1F& hist, int jdx)
{
  hist.SetLineColor( 1 );
  if( jdx<((int)histColor_.size()) ){
    hist.SetLineColor(histColor_[jdx]);
  }
}

void 
CompHist::setFillStyle(TH1F& hist, int jdx)
{
  hist.SetFillStyle( 3005 );
  if( jdx<((int)commonStyle_.size()) ){
    hist.SetFillStyle(commonStyle_[jdx]);
  }
}

void 
CompHist::setFillColor(TH1F& hist, int jdx)
{
  hist.SetFillColor( 1 );
  if( jdx<((int)histColor_.size()) ){
    hist.SetFillColor(histColor_[jdx]);
  }
}

void 
CompHist::setMarkerStyle(TH1F& hist, int jdx)
{
  hist.SetMarkerStyle( 23 );
  if( jdx<((int)markerStyle_.size()) ){
    hist.SetMarkerStyle(markerStyle_[jdx]);
  }
}

void 
CompHist::setMarkerColor(TH1F& hist, int jdx)
{
  hist.SetMarkerColor( 1 );
  if( jdx<((int)histColor_.size()) ){
    hist.SetMarkerColor(histColor_[jdx]);
  }
}

void 
CompHist::setMarkerSize(TH1F& hist, int jdx)
{
  hist.SetMarkerSize( 2.3 );
  if( jdx<((int)markerSize_.size()) ){
    hist.SetMarkerSize(markerSize_[jdx]);
  }
}

void 
CompHist::writeOutput(CompHist::RootOutput option)
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
    
    //-----------------------------------------------
    // loop and write out histograms
    //-----------------------------------------------
    if( option==kAll ){
      std::vector<TObjArray>::const_iterator hist = sampleList_.begin();
      for( ;hist!=sampleList_.end(); ++hist){
	for(unsigned int idx=0; idx<histList_.size(); ++idx){  
	  histFile << ((TH1F*)(*hist)[idx])->GetName() << "\n";
	  ((TH1F*)(*hist)[idx])->Write();
	}
      }
    }
    else{
      for(unsigned int idx=0; idx<histList_.size(); ++idx){ 
	// write first/last histograms in the sample list only
	histFile << (TH1F*)((sampleList_.back())[idx])->GetName() << "\n";
	switch( option ){
	case kFirstOnly:
	  ((TH1F*)((*sampleList_.begin())[idx]))->Write();
	  break;
	case kLastOnly:
	  ((TH1F*)((sampleList_.back())[idx]))->Write();
	  break;
	default:
	  std::cerr << "WARNING:" 
	       << " Wrong option indicated for writeOutput" << " [" << option 
	       << "]" << " while only " << kFirstOnly << " up to " << kAll 
	       << " are available" << std::endl;
	  break;	  
	}
      }
    }
    file.Close();
  }
}
