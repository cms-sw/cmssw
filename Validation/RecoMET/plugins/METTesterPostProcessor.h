#ifndef METTESTERPOSTPROCESSOR_H
#define METTESTERPOSTPROCESSOR_H

// author: Matthias Weber, Feb 2015

// system include files
#include <memory>
#include <stdio.h>
#include <math.h>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
//
// class decleration
//

class METTesterPostProcessor : public DQMEDHarvester {
   public:
      explicit METTesterPostProcessor(const edm::ParameterSet&);
      ~METTesterPostProcessor();

   private:
      virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) ;
      std::vector<std::string> met_dirs;

      void FillMETRes(std::string metdir,DQMStore::IGetter &);
      MonitorElement* mMETDifference_GenMETTrue_MET0to20;
      MonitorElement* mMETDifference_GenMETTrue_MET20to40;
      MonitorElement* mMETDifference_GenMETTrue_MET40to60;
      MonitorElement* mMETDifference_GenMETTrue_MET60to80;
      MonitorElement* mMETDifference_GenMETTrue_MET80to100;
      MonitorElement* mMETDifference_GenMETTrue_MET100to150;
      MonitorElement* mMETDifference_GenMETTrue_MET150to200;
      MonitorElement* mMETDifference_GenMETTrue_MET200to300;
      MonitorElement* mMETDifference_GenMETTrue_MET300to400;
      MonitorElement* mMETDifference_GenMETTrue_MET400to500;
      MonitorElement* mMETDifference_GenMETTrue_MET500;
      MonitorElement* mMETDifference_GenMETTrue_METResolution; 
};

#endif
