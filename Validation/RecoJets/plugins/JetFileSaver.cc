#include "Validation/RecoJets/plugins/JetFileSaver.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <vector>
#include <utility>
#include <ostream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <memory>
#include "DQMServices/Core/interface/DQMStore.h"

JetFileSaver::JetFileSaver(const edm::ParameterSet& iConfig)
{
  outputFile_              = iConfig.getUntrackedParameter<std::string>("OutputFile");
  if (outputFile_.size() > 0)
    edm::LogInfo("OutputInfo") << " Jet Task histograms will be saved to '" << outputFile_.c_str() << "'";
  else edm::LogInfo("OutputInfo") << " Jet Task histograms will NOT be saved";
  
}

void JetFileSaver::beginJob()
{
  // get ahold of back-end interface
  dbe_ = edm::Service<DQMStore>().operator->();

}

void JetFileSaver::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
}

void JetFileSaver::endJob() 
{
  // Store the DAQ Histograms
  if (outputFile_.size() > 0 && dbe_)
  dbe_->save(outputFile_);
}
