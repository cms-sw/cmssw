#ifndef Validation_DTRecHits_DT2DSegmentClients_h
#define Validation_DTRecHits_DT2DSegmentClients_h

/** \class DTDataIntegrityTest
 * *
 *  DQM Client to check the data integrity
 *
 *  \author S. Bolognesi - INFN TO
 *
 */

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DT2DSegmentClients : public DQMEDHarvester {
public:
  /// Constructor
  DT2DSegmentClients(const edm::ParameterSet &ps);

  /// Destructor
  ~DT2DSegmentClients() override;

protected:
  /// End Job
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

private:
  bool do2D_;
  bool doSLPhi_;
};

#endif  // Validation_DTRecHits_DT2DSegmentClients_h
