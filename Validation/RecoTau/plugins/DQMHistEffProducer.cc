#include "Validation/RecoTau/plugins/DQMHistEffProducer.h"

#include "Validation/RecoTau/plugins/dqmAuxFunctions.h"

// framework & common header files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


#include <TH1.h>
#include <TCanvas.h>

#include <iostream>

DQMHistEffProducer::cfgEntryPlot::cfgEntryPlot(const edm::ParameterSet& cfg)
{
  //std::cout << "<cfgEntryPlot::cfgEntryPlot>:" << std::endl;

  numerator_ = cfg.getParameter<std::string>("numerator");
  //std::cout << " numerator = " << numerator_ << std::endl;

  denominator_ = cfg.getParameter<std::string>("denominator");
  //std::cout << " denominator = " << denominator_ << std::endl;

  efficiency_ = cfg.getParameter<std::string>("efficiency");
  //std::cout << " efficiency = " << efficiency_ << std::endl;
}

DQMHistEffProducer::cfgEntryPlot::cfgEntryPlot(const std::string& numerator, const std::string& denominator, const std::string& efficiency)
  : numerator_(numerator), denominator_(denominator), efficiency_(efficiency)
{
  //std::cout << "<cfgEntryPlot::cfgEntryPlot>:" << std::endl;
  //std::cout << " numerator = " << numerator_ << std::endl;
  //std::cout << " denominator = " << denominator_ << std::endl;
  //std::cout << " efficiency = " << efficiency_ << std::endl;
}

//
//-----------------------------------------------------------------------------------------------------------------------
//

DQMHistEffProducer::DQMHistEffProducer(const edm::ParameterSet& cfg)
{
  //std::cout << "<DQMHistEffProducer::DQMHistEffProducer>:" << std::endl;
  
  edm::ParameterSet plots = cfg.getParameter<edm::ParameterSet>("plots");
  std::vector<std::string> plotNames = plots.getParameterNamesForType<edm::ParameterSet>();
  for ( std::vector<std::string>::const_iterator plotName = plotNames.begin(); plotName != plotNames.end(); ++plotName ) {
    edm::ParameterSet plotConfig = plots.getParameter<edm::ParameterSet>(*plotName);

    typedef std::vector<std::string> vstring;
    vstring plotParameter = plotConfig.getParameter<vstring>("parameter");
    if ( plotParameter.size() == 0 ) {
      cfgEntryPlot_.push_back(cfgEntryPlot(plotConfig));
    } else {
      std::string numerator = plotConfig.getParameter<std::string>("numerator");
      std::string denominator = plotConfig.getParameter<std::string>("denominator");
      std::string efficiency = plotConfig.getParameter<std::string>("efficiency");
      for ( vstring::const_iterator parameter = plotParameter.begin(); parameter != plotParameter.end(); ++parameter ) {
	int errorFlag = 0;
	std::string modNumerator = replace_string(numerator, parKeyword, *parameter, 1, 1, errorFlag);
	std::string modDenominator = replace_string(denominator, parKeyword, *parameter, 1, 1, errorFlag);
	std::string modEfficiency = replace_string(efficiency, parKeyword, *parameter, 1, 1, errorFlag);

	if ( !errorFlag ) {
	  cfgEntryPlot_.push_back(cfgEntryPlot(modNumerator, modDenominator, modEfficiency));
	} else {
	  edm::LogError("DQMHistEffProducer") << " Failed to decode histogram names for plotName = " << (*plotName) 
					      << " --> skipping !!";
	}
      }
    }
  }
}

DQMHistEffProducer::~DQMHistEffProducer() 
{
//--- nothing to be done yet
}

void DQMHistEffProducer::analyze(const edm::Event&, const edm::EventSetup&)
{
//--- nothing to be done yet
}

void DQMHistEffProducer::endRun(const edm::Run& r, const edm::EventSetup& c)
{
  //std::cout << "<DQMHistEffProducer::endJob>:" << std::endl;

//--- check that DQMStore service is available
  if ( !edm::Service<DQMStore>().isAvailable() ) {
    edm::LogError ("endJob") << " Failed to access dqmStore --> histograms will NOT be plotted !!";
    return;
  }

  DQMStore& dqmStore = (*edm::Service<DQMStore>());

  for ( std::vector<cfgEntryPlot>::const_iterator plot = cfgEntryPlot_.begin(); plot != cfgEntryPlot_.end(); ++plot ) {
    //std::cout << "plot->numerator_ = " << plot->numerator_ << std::endl;
    std::string numeratorHistogramName, numeratorHistogramDirectory;
    separateHistogramFromDirectoryName(plot->numerator_, numeratorHistogramName, numeratorHistogramDirectory);
    //std::cout << "numeratorHistogramName = " << numeratorHistogramName << std::endl;
    //std::cout << "numeratorHistogramDirectory = " << numeratorHistogramDirectory << std::endl;
    MonitorElement* meNumerator = dqmStore.get(std::string(numeratorHistogramDirectory).append(dqmSeparator).append(numeratorHistogramName));
    //std::cout << "meNumerator = " << meNumerator << std::endl;
    TH1* histoNumerator = ( meNumerator != NULL ) ? meNumerator->getTH1() : NULL;
    
    //std::cout << "plot->denominator_ = " << plot->denominator_ << std::endl;
    std::string denominatorHistogramName, denominatorHistogramDirectory;
    separateHistogramFromDirectoryName(plot->denominator_, denominatorHistogramName, denominatorHistogramDirectory);
    //std::cout << "denominatorHistogramName = " << denominatorHistogramName << std::endl;
    //std::cout << "denominatorHistogramDirectory = " << denominatorHistogramDirectory << std::endl;
    MonitorElement* meDenominator = dqmStore.get(std::string(denominatorHistogramDirectory).append(dqmSeparator).append(denominatorHistogramName));
    //std::cout << "meDenominator = " << meDenominator << std::endl;
    TH1* histoDenominator = ( meDenominator != NULL ) ? meDenominator->getTH1() : NULL;
    
    if ( histoNumerator != NULL && histoDenominator != NULL ) {
      if ( !histoNumerator->GetSumw2N() ) histoNumerator->Sumw2();
      //std::cout << " histoNumerator->GetName = " << histoNumerator->GetName() << std::endl;
      
      if ( !histoDenominator->GetSumw2N() ) histoDenominator->Sumw2();
      //std::cout << " histoDenominator->GetName = " << histoNumerator->GetName() << std::endl;
      
      std::string effHistogramName, effHistogramDirectory, dummy;
      separateHistogramFromDirectoryName(plot->efficiency_, effHistogramName, effHistogramDirectory);
      //if ( effHistogramDirectory == "" ) separateHistogramFromDirectoryName(numeratorHistogramName, dummy, effHistogramDirectory);
      if ( effHistogramDirectory != "" ) 
	{
	  if(dqmStore.dirExists(effHistogramDirectory))
	    dqmStore.setCurrentFolder(effHistogramDirectory);
	  else
	    std::cout<<"DQMHistEffProducer:: Directory: "<<effHistogramDirectory<<" does not exist!"<<std::endl;
	}
      
      MonitorElement* histoEfficiency = dqmStore.book1D(effHistogramName, effHistogramName, 
							histoNumerator->GetNbinsX(), histoNumerator->GetXaxis()->GetXmin(), histoNumerator->GetXaxis()->GetXmax());
      
      histoEfficiency->getTH1F()->Divide(histoNumerator, histoDenominator, 1., 1., "B");

      //to avoid the pointer to go out of scope:
      histoEfficiencyVector_.push_back(histoEfficiency);

      /*      std::vector<std::string> mes = dqmStore.getMEs();
      std::cout<<dqmStore.pwd()<<std::endl;
      for(unsigned int i =0; i<mes.size(); i++)
	std::cout<<mes[i]<<std::endl;
      */
    } else {
      edm::LogError("endRun") << " Failed to produce efficiency histogram = " << plot->efficiency_ << " !!";
      if ( histoNumerator   == NULL ) edm::LogError("endRun") << "  numerator = " << plot->numerator_ << " does not exist.";
      if ( histoDenominator == NULL ) edm::LogError("endRun") << "  denominator = " << plot->denominator_ << " does not exist.";
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(DQMHistEffProducer);
