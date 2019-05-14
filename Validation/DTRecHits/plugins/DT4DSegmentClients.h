#ifndef Validation_DTRecHits_DT4DSegmentClients_h
#define Validation_DTRecHits_DT4DSegmentClients_h

/** \class DTDataIntegrityTest
 * *
 *  DQM Client to check the data integrity
 *
 *  \author S. Bolognesi - INFN TO
 *
 */

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DT4DSegmentClients : public DQMEDHarvester {
public:
  /// Constructor
  DT4DSegmentClients(const edm::ParameterSet &ps);
  /// Destructor
  ~DT4DSegmentClients() override;

protected:
  /// End Job
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

private:
  bool doall_;
};

#endif  // Validation_DTRecHits_DT4DSegmentClients_h
