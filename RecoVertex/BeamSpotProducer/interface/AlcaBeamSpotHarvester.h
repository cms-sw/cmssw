#ifndef AlcaBeamSpotHarvester_H
#define AlcaBeamSpotHarvester_H

/** \class AlcaBeamSpotHarvester
 *  No description available.
 *
 *  $Date: 2010/06/15 14:49:29 $
 *  $Revision: 1.1 $
 *  \author L. Uplegger F. Yumiceva - Fermilab
 */
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <string>


class AlcaBeamSpotHarvester : public edm::EDAnalyzer {
 public:
  /// Constructor
  AlcaBeamSpotHarvester(const edm::ParameterSet&);

  /// Destructor
  virtual ~AlcaBeamSpotHarvester();
  
  // Operations
  virtual void beginJob            (void);
  virtual void endJob              (void);  
  virtual void analyze             (const edm::Event&          , const edm::EventSetup&);
  virtual void beginRun            (const edm::Run&            , const edm::EventSetup&);
  virtual void endRun              (const edm::Run&            , const edm::EventSetup&);
  virtual void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
  virtual void endLuminosityBlock  (const edm::LuminosityBlock&, const edm::EventSetup&);

 protected:

 private:
  std::string beamSpotOutputBase_;
};
#endif

