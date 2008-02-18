#ifndef CaloSimAlgos_CaloHitAnalyzer_h
#define CaloSimAlgos_CaloHitAnalyzer_h

#include <map>
#include <string>
#include "SimCalorimetry/CaloSimAlgos/interface/CaloValidationStatistics.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

/**
  Sums up all the SimHit energy in each cell, and uses that
  to compare to the Rechits of the cell

  \Author Rick Wilkinson, Caltech
*/

class CaloVSimParameterMap;
class CaloVHitFilter;
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"


class CaloHitAnalyzer 
{
public:
  CaloHitAnalyzer(const std::string & name,
                  double hitEnergyThreshold,
                  const CaloVSimParameterMap * parameterMap,
                  const CaloVHitFilter * filter = 0);

  /// should be called each event
  void fillHits(MixCollection<PCaloHit> & hits);
  
  /// to be called for each RecHit
  void analyze(int detId, double recEnergy);

public:
  std::map<int, double> hitEnergySumMap_;
  double hitEnergyThreshold_;
  const CaloVSimParameterMap * simParameterMap_;
  const CaloVHitFilter * hitFilter_;
  CaloValidationStatistics summary_;
  int noiseHits_;
};

#endif

