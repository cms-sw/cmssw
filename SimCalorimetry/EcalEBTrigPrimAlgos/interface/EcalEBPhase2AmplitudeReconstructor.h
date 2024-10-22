#ifndef SimCalorimetry_EcalEBTrigPrimAlgos_EcalEBPhase2AmplitudeReconstructor_h
#define SimCalorimetry_EcalEBTrigPrimAlgos_EcalEBPhase2AmplitudeReconstructor_h

#include <vector>
#include <cstdint>

class EcalEBPhase2TPGAmplWeightIdMap;
class EcalTPGWeightGroup;

/** \class EcalPhase2AmplitudeReconstructor 
\author L. Lutton, N. Marinelli - Univ. of Notre Dame
 Description: forPhase II 
 It uses the new Phase2 digis based on the new EB electronics
 and measures the amplitude on xTals basis
*/

class EcalEBPhase2AmplitudeReconstructor {
private:
  static const int maxSamplesUsed_ = 12;
  bool debug_;
  int inputsAlreadyIn_;
  int buffer_[maxSamplesUsed_];
  int weights_[maxSamplesUsed_];
  int shift_;
  int setInput(int input);
  void process();
  int processedOutput_;

public:
  EcalEBPhase2AmplitudeReconstructor(bool debug);
  virtual ~EcalEBPhase2AmplitudeReconstructor();
  virtual void process(std::vector<int> &addout, std::vector<int> &output);
  void setParameters(uint32_t raw,
                     const EcalEBPhase2TPGAmplWeightIdMap *ecaltpgWeightMap,
                     const EcalTPGWeightGroup *ecaltpgWeightGroup);
};

#endif
