#ifndef AlcaBeamSpotHarvester_H
#define AlcaBeamSpotHarvester_H

/** \class AlcaBeamSpotHarvester
 *  No description available.
 *
 *  $Date: $
 *  $Revision: $
 *  \author L. Uplegger F. Yumiceva - Fermilab
 */
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

class AlcaBeamSpotHarvester : public edm::EDAnalyzer {
public:
  /// Constructor
  AlcaBeamSpotHarvester(const edm::ParameterSet&);

  /// Destructor
  virtual ~AlcaBeamSpotHarvester();
  
  // Operations
  virtual void beginJob();
  virtual void endJob();  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void endRun(const edm::Run&, const edm::EventSetup&);
  virtual void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
  virtual void endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);

protected:

private:

};
#endif

