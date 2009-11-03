#ifndef DT4DSegmentClients_H
#define DT4DSegmentClients_H

/** \class DTDataIntegrityTest
 * *
 *  DQM Client to check the data integrity
 *
 *  $Date: 2009/05/20 14:17:01 $
 *  $Revision: 1.12 $
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

class DT4DSegmentClients: public edm::EDAnalyzer{

public:

  /// Constructor
  DT4DSegmentClients(const edm::ParameterSet& ps);

 /// Destructor
 ~DT4DSegmentClients();

protected:

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

private:

  DQMStore* dbe;

  MonitorElement *summaryHisto;
  MonitorElement *glbSummaryHisto;
 };

#endif
