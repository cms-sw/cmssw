#include "EgammaAnalysis/ElectronTools/interface/EcalIsolationCorrector.h"

EcalIsolationCorrector::RunRange EcalIsolationCorrector::checkRunRange(int runNumber) {
  
  EcalIsolationCorrector::RunRange runRange = EcalIsolationCorrector::RunRange::RunAB;

  if (runNumber <= 203755 && runNumber > 197770)
    runRange = EcalIsolationCorrector::RunRange::RunC;
  else if (runNumber > 203755)
    runRange = EcalIsolationCorrector::RunRange::RunD;

  return runRange;
}

float EcalIsolationCorrector::correctForNoise(float iso, bool isBarrel, EcalIsolationCorrector::RunRange runRange, bool isData) {

  float result = iso;

  if (!isData) {
    if (runRange == EcalIsolationCorrector::RunAB) {
      if (isBarrel)
	result = (iso+0.1174)/1.0012;
      else
	result = (iso+0.2736)/0.9948;
    } else if (runRange == EcalIsolationCorrector::RunC) {
      if (isBarrel)
	result = (iso+0.2271)/0.9684;
      else
	result = (iso+0.5962)/0.9568;
    } else if (runRange == EcalIsolationCorrector::RunD) {
      if (isBarrel) 
	result = (iso+0.2907)/1.0005;
      else
	result = (iso+0.9098)/0.9395;
    }
  } else {
    std::cout << "Warning: you should correct MC to data" << std::endl;
  } 

  return result;
}

#ifndef STANDALONE_ECALCORR
float EcalIsolationCorrector::correctForNoise(reco::GsfElectron e, int runNumber, bool isData) {
  
  float iso = e.dr03EcalRecHitSumEt();
  EcalIsolationCorrector::RunRange runRange = checkRunRange(runNumber);

  return correctForNoise(iso, e.isEB(), runRange, isData);
}

float EcalIsolationCorrector::correctForNoise(reco::GsfElectron e, bool isData, float intL_AB, float intL_C, float intL_D) {
  
  float iso = e.dr03EcalRecHitSumEt();
  float combination = (intL_AB * correctForNoise(iso, e.isEB(), EcalIsolationCorrector::RunAB, isData) +
		       intL_C  * correctForNoise(iso, e.isEB(), EcalIsolationCorrector::RunC,  isData) +
		       intL_D  * correctForNoise(iso, e.isEB(), EcalIsolationCorrector::RunD,  isData))/(intL_AB + intL_C + intL_D);

  return combination;
}
#else
float EcalIsolationCorrector::correctForNoise(float iso, bool isBarrel, int runNumber, bool isData) {
  
  EcalIsolationCorrector::RunRange runRange = checkRunRange(runNumber);

  return correctForNoise(iso, isBarrel, runRange, isData);
}

float EcalIsolationCorrector::correctForNoise(float iso , bool isBarrel, bool isData, float intL_AB, float intL_C, float intL_D) {
  
  float combination = (intL_AB * correctForNoise(iso, isBarrel, EcalIsolationCorrector::RunAB, isData) +
		       intL_C  * correctForNoise(iso, isBarrel, EcalIsolationCorrector::RunC,  isData) +
		       intL_D  * correctForNoise(iso, isBarrel, EcalIsolationCorrector::RunD,  isData))/(intL_AB + intL_C + intL_D);

  return combination;
}
#endif

float EcalIsolationCorrector::correctForHLTDefinition(float iso, bool isBarrel, EcalIsolationCorrector::RunRange runRange) {

  float result = iso;

  if (runRange == EcalIsolationCorrector::RunAB) {
    if (isBarrel)
      result = iso*0.8499-0.6510;
    else
      result = iso*0.8504-0.5658;
  } else if (runRange == EcalIsolationCorrector::RunC) {
    if (isBarrel)
      result = iso*0.9346-0.9987;
    else
      result = iso*0.8529-0.6816;
  } else if (runRange == EcalIsolationCorrector::RunD) {
    if (isBarrel) 
      result = iso*0.8318-0.9999;
    else
      result = iso*0.8853-0.8783;
  }
  
  return result;
}

#ifndef STANDALONE_ECALCORR
float EcalIsolationCorrector::correctForHLTDefinition(reco::GsfElectron e, int runNumber, bool isData) {
  
  float iso = e.dr03EcalRecHitSumEt();
  EcalIsolationCorrector::RunRange runRange = checkRunRange(runNumber);

  if (!isData)
    iso = correctForNoise(iso, e.isEB(), runRange, false);

  return correctForHLTDefinition(iso, e.isEB(), runRange);
}

float EcalIsolationCorrector::correctForHLTDefinition(reco::GsfElectron e, bool isData, float intL_AB, float intL_C, float intL_D) {
  
  float iso = e.dr03EcalRecHitSumEt();
  if (!isData)
    iso = correctForNoise(e, isData, intL_AB, intL_C, intL_D);

  float combination = (intL_AB * correctForHLTDefinition(iso, e.isEB(), EcalIsolationCorrector::RunAB) +
		       intL_C  * correctForHLTDefinition(iso, e.isEB(), EcalIsolationCorrector::RunC ) +
		       intL_D  * correctForHLTDefinition(iso, e.isEB(), EcalIsolationCorrector::RunD ))/(intL_AB + intL_C + intL_D);

  return combination;
}
#else
float EcalIsolationCorrector::correctForHLTDefinition(float iso, bool isBarrel, int runNumber, bool isData) {
  
  EcalIsolationCorrector::RunRange runRange = checkRunRange(runNumber);

  if (!isData)
    iso = correctForNoise(iso, isBarrel, runRange, false);

  return correctForHLTDefinition(iso, isBarrel, runRange);
}

float EcalIsolationCorrector::correctForHLTDefinition(float iso, bool isBarrel, bool isData, float intL_AB, float intL_C, float intL_D) {
  
  if (!isData)
    iso = correctForNoise(iso, isBarrel, false, intL_AB, intL_C, intL_D);
  
  float combination = (intL_AB * correctForHLTDefinition(iso, isBarrel, EcalIsolationCorrector::RunAB) +
		       intL_C  * correctForHLTDefinition(iso, isBarrel, EcalIsolationCorrector::RunC ) +
		       intL_D  * correctForHLTDefinition(iso, isBarrel, EcalIsolationCorrector::RunD ))/(intL_AB + intL_C + intL_D);

  return combination;
}
#endif


