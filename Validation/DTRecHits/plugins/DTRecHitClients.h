#ifndef DTRecHitClients_H
#define DTRecHitClients_H

/** \class DTDataIntegrityTest
 * *
 *  DQM Client to check the data integrity
 *
 *  $Date: 2009/11/04 13:54:28 $
 *  $Revision: 1.3 $
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

class DTRecHitClients: public edm::EDAnalyzer{

public:

  /// Constructor
  DTRecHitClients(const edm::ParameterSet& ps);

 /// Destructor
  virtual ~DTRecHitClients();

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  void endJob();
void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg,
					 edm::EventSetup const& c);

protected:


private:

  DQMStore* dbe;
 };
 

#endif
