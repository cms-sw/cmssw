#ifndef CSCTriggerPrimitives_CSCDigiSuppressor_h
#define CSCTriggerPrimitives_CSCDigiSuppressor_h
#include <list>

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "SimMuon/CSCDigitizer/src/CSCStripElectronicsSim.h"
#include "SimMuon/CSCDigitizer/src/CSCDbStripConditions.h"


class CSCDigiSuppressor : public edm::EDProducer
{
public:
  explicit CSCDigiSuppressor(const edm::ParameterSet& ps);
  ~CSCDigiSuppressor() {delete theStripConditions;}

  /**Produces the EDM products,*/
  virtual void produce(edm::Event& e, const edm::EventSetup& c);

private:
  void fillDigis(const CSCDetId & id, const std::list<int> & keyStrips,
                 const CSCStripDigiCollection & oldStripDigis,
                 CSCStripDigiCollection & newStripDigis);

  std::list<int>
  cfebsToRead(const CSCDetId & id, const std::list<int> & keyStrips) const;

  std::list<int>
  stripsToRead(const std::list<int> & cfebs) const;

  void suppressWires(edm::Event & e);

  std::string theLCTLabel;
  std::string theDigiLabel;

  CSCStripElectronicsSim theStripElectronicsSim;
  CSCStripConditions * theStripConditions;
};

#endif

