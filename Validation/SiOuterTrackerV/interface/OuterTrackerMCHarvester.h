#ifndef OuterTrackerMCHarvester_H
#define OuterTrackerMCHarvester_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class OuterTrackerMCHarvester : public DQMEDHarvester {
		public:
			explicit OuterTrackerMCHarvester(const edm::ParameterSet&);
			~OuterTrackerMCHarvester() override;
			void dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) override;

		private:
			// ----------member data ---------------------------
			 DQMStore *dbe;
	};

#endif
