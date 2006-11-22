#include "Validation/MuonCSCDigis/src/CSCDigiValidation.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

CSCDigiValidation::CSCDigiValidation(const edm::ParameterSet & ps)
: dbe_( edm::Service<DaqMonitorBEInterface>().operator->() ),
  outputFile_( ps.getParameter<std::string>("outputFile") ),
  theStripDigiValidation(ps, dbe_),
  theWireDigiValidation(ps, dbe_),
  theComparatorDigiValidation(ps, dbe_)
{

}


CSCDigiValidation::~CSCDigiValidation()
{
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}


void CSCDigiValidation::endJob() {
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}


void CSCDigiValidation::analyze(const edm::Event&e, const edm::EventSetup&es)
{
std::cout << "CSCDigiValidation " << std::endl;
  theStripDigiValidation.analyze(e,es);
  theWireDigiValidation.analyze(e,es);
  theComparatorDigiValidation.analyze(e,es);
}



