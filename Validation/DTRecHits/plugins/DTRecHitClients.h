#ifndef Validation_DTRecHits_DTRecHitClients_h
#define Validation_DTRecHits_DTRecHitClients_h

/** \class DTDataIntegrityTest
 * *
 *  DQM Client to check the data integrity
 *
 *  \author S. Bolognesi - INFN TO
 *
 */

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DTRecHitClients : public DQMEDHarvester {
public:
  /// Constructor
  DTRecHitClients(const edm::ParameterSet &ps);
  /// Destructor
  ~DTRecHitClients() override;

protected:
  /// End Job
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

private:
  // Switches for analysis at various steps
  bool doStep1_;
  bool doStep2_;
  bool doStep3_;
  bool local_;
  bool doall_;
};

#endif  // Validation_DTRecHits_DTRecHitClients_h
