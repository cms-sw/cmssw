#ifndef DT2DSegmentClients_H
#define DT2DSegmentClients_H

/** \class DTDataIntegrityTest
 * *
 *  DQM Client to check the data integrity
 *
 *  \author S. Bolognesi - INFN TO
 *   
 */
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/Framework/interface/ESHandle.h"
#include <FWCore/Framework/interface/EventSetup.h>
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

class DQMStore;
class MonitorElement;

class DT2DSegmentClients: public edm::EDAnalyzer{

public:

  /// Constructor
 DT2DSegmentClients(const edm::ParameterSet& ps);

 /// Destructor
 ~DT2DSegmentClients() override;
 void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void endJob() override;
void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg,
					 edm::EventSetup const& c) override;

protected:

  /// Analyze

private:

  DQMStore* dbe;

  MonitorElement *summaryHisto;
  MonitorElement *glbSummaryHisto;
 };

#endif
