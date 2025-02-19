#ifndef PerformanceAnalyzer_H
#define PerformanceAnalyzer_H

// user include files

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class PerformanceAnalyzer : public edm::EDAnalyzer 
{

   public:
   explicit PerformanceAnalyzer(const edm::ParameterSet&);
   ~PerformanceAnalyzer();
   virtual void analyze(const edm::Event&, const edm::EventSetup&);
   virtual void beginJob(){} 
   virtual void endJob() ;
 

   private:
   DQMStore*   fDBE ;
   std::string              fOutputFile ;
   MonitorElement*          fVtxSmeared ;
   MonitorElement*          fg4SimHits ;
   MonitorElement*          fMixing ;
   MonitorElement*          fSiPixelDigis ;
   MonitorElement*          fSiStripDigis ;
   MonitorElement*          fEcalUnsuppDigis ;
   MonitorElement*          fEcalZeroSuppDigis ;
   MonitorElement*          fPreShwZeroSuppDigis ;
   MonitorElement*          fHcalUnsuppDigis ;
   MonitorElement*          fMuonCSCDigis ;
   MonitorElement*          fMuonDTDigis ;
   MonitorElement*          fMuonRPCDigis ;

};

#endif

