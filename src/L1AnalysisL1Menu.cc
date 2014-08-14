#include "L1TriggerDPG/L1Ntuples/interface/L1AnalysisL1Menu.h"
#include <vector>

void L1Analysis::L1AnalysisL1Menu::SetPrescaleFactorIndex(L1GtUtils & l1GtUtils_, const edm::Event& iEvent)
{
  int iErrorCode=-1;
  const int pfSetIndexAlgorithmTrigger = 
    l1GtUtils_.prescaleFactorSetIndex(iEvent, L1GtUtils::AlgorithmTrigger, iErrorCode);

  if (iErrorCode == 0)
  {
    std::cout << "\nAlgorithm triggers: index for prescale factor set = "
              << pfSetIndexAlgorithmTrigger << "\nfor run " << iEvent.run()
              << ", luminosity block " << iEvent.luminosityBlock()
              << ", with L1 menu \n  " << l1GtUtils_.l1TriggerMenu()
              << std::endl;
    data_.AlgoTrig_PrescaleFactorIndexValid = true;
    data_.AlgoTrig_PrescaleFactorIndex      = pfSetIndexAlgorithmTrigger;
  }
  else
  {
    std::cout << "\nError encountered when retrieving the prescale factor set index"
              << "\n  for algorithm triggers, for run " << iEvent.run()
              << ", luminosity block " << iEvent.luminosityBlock()
              << " with L1 menu \n  " << l1GtUtils_.l1TriggerMenu()
              << "\n  Error code: " << iErrorCode << "\n" << std::endl;
    data_.AlgoTrig_PrescaleFactorIndexValid = false;
    data_.AlgoTrig_PrescaleFactorIndex      = 0;
  }

  

  iErrorCode=-1;
  const int pfSetIndexTechnicalTrigger = 
    l1GtUtils_.prescaleFactorSetIndex(iEvent, L1GtUtils::TechnicalTrigger, iErrorCode);

  if (iErrorCode == 0)
  {
    std::cout << "\nAlgorithm triggers: index for prescale factor set = "
              << pfSetIndexTechnicalTrigger << "\nfor run " << iEvent.run()
              << ", luminosity block " << iEvent.luminosityBlock()
              << ", with L1 menu \n  " << l1GtUtils_.l1TriggerMenu()
              << std::endl;
    data_.TechTrig_PrescaleFactorIndexValid = true;
    data_.TechTrig_PrescaleFactorIndex      = pfSetIndexTechnicalTrigger;
  }
  else
  {
    std::cout << "\nError encountered when retrieving the prescale factor set index"
              << "\n  for algorithm triggers, for run " << iEvent.run()
              << ", luminosity block " << iEvent.luminosityBlock()
              << " with L1 menu \n  " << l1GtUtils_.l1TriggerMenu()
              << "\n  Error code: " << iErrorCode << "\n" << std::endl;
    data_.TechTrig_PrescaleFactorIndexValid = false;
    data_.TechTrig_PrescaleFactorIndex      = 0;
  }

}
