#include "Validation/RecoTau/plugins/DQMHistAdder.h"

#include "Validation/RecoTau/plugins/dqmAuxFunctions.h"

// framework & common header files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <TH1.h>

#include <iostream>

const double defaultScaleFactor = 1.;

const int verbosity = 0;

//
//-----------------------------------------------------------------------------------------------------------------------
//

DQMHistAdder::cfgEntryAddJob::cfgEntryAddJob(const std::string& name, const edm::ParameterSet& cfg)
{
  //std::cout << "<DQMHistAdder::cfgEntryAddJob>" << std::endl;

  name_ = name;

  dqmDirectories_input_ = cfg.getParameter<vstring>("dqmDirectories_input");
  dqmDirectory_output_ = cfg.getParameter<std::string>("dqmDirectory_output");
  
  if ( verbosity ) print();
}

void DQMHistAdder::cfgEntryAddJob::print() const
{
  std::cout << "<cfgEntryAddJob::print>:" << std::endl;
  std::cout << " name = " << name_ << std::endl;
  std::cout << " dqmDirectories_input = " << format_vstring(dqmDirectories_input_) << std::endl;
  std::cout << " dqmDirectory_output = " << dqmDirectory_output_ << std::endl;
}

//
//-----------------------------------------------------------------------------------------------------------------------
//

DQMHistAdder::DQMHistAdder(const edm::ParameterSet& cfg)
{
  std::cout << "<DQMHistAdder::DQMHistAdder>:" << std::endl;

  cfgError_ = 0;

//--- configure processes  
  //std::cout << "--> configuring addJobs..." << std::endl;
  readCfgParameter<cfgEntryAddJob>(cfg, addJobs_);
  
  std::cout << "done." << std::endl;
}

DQMHistAdder::~DQMHistAdder() 
{
// nothing to be done yet...
}

void DQMHistAdder::analyze(const edm::Event&, const edm::EventSetup&)
{
// nothing to be done yet...
}

void DQMHistAdder::endJob()
{
  std::cout << "<DQMHistAdder::endJob>:" << std::endl;

//--- check that configuration parameters contain no errors
  if ( cfgError_ ) {
    edm::LogError ("endJob") << " Error in Configuration ParameterSet --> histograms will NOT be added !!";
    return;
  }

//--- check that DQMStore service is available
  if ( !edm::Service<DQMStore>().isAvailable() ) {
    edm::LogError ("endJob") << " Failed to access dqmStore --> histograms will NOT be added !!";
    return;
  }

//--- stop ROOT from keeping references to all histograms
  //TH1::AddDirectory(false);

//--- add histograms
  //std::cout << "--> adding histograms..." << std::endl;
  DQMStore& dqmStore = (*edm::Service<DQMStore>());
  for ( std::map<std::string, cfgEntryAddJob>::const_iterator addJob = addJobs_.begin();
	addJob != addJobs_.end(); ++addJob ) {
    const std::string& dqmDirectory_output = addJob->second.dqmDirectory_output_;
    for ( vstring::const_iterator dqmDirectory_input = addJob->second.dqmDirectories_input_.begin();
	  dqmDirectory_input != addJob->second.dqmDirectories_input_.end(); ++dqmDirectory_input ) {
      
      std::string inputDirectory = dqmDirectoryName(std::string(dqmRootDirectory)).append(*dqmDirectory_input);
      //std::cout << " inputDirectory = " << inputDirectory << std::endl;
      std::string outputDirectory = dqmDirectoryName(std::string(dqmRootDirectory)).append(dqmDirectory_output);
      //std::cout << " outputDirectory = " << outputDirectory << std::endl;
      
//--- when processing first inputDirectory, check that histograms in outputDirectory do not yet exist;
//    afterwards, add histograms in inputDirectory to those in outputDirectory
      int mode = ( dqmDirectory_input == addJob->second.dqmDirectories_input_.begin() ) ? 1 : 3;
      //std::cout << " mode = " << mode << std::endl;
      dqmCopyRecursively(dqmStore, inputDirectory, outputDirectory, 1., mode, false);
    }
  }

  std::cout << "done." << std::endl; 
  if ( verbosity ) dqmStore.showDirStructure();
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(DQMHistAdder);
