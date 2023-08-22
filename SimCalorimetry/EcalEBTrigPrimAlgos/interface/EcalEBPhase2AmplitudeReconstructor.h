#ifndef ECAL_EB_PHASE2_AMPLITUDE_RECONSTRUCTOR_H
#define ECAL_EB_PHASE2_AMPLITUDE_RECONSTRUCTOR_H

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
  bool debug_;
  int inputsAlreadyIn_;
  int buffer_[12];
  int weights_[12];
  int shift_;
  int setInput(int input);
  void process();
  
  int processedOutput_;

 public:
  EcalEBPhase2AmplitudeReconstructor(bool debug);
  virtual ~EcalEBPhase2AmplitudeReconstructor();
  virtual void process(std::vector<int> & addout, std::vector<int> & output);
  void setParameters(uint32_t raw,const EcalEBPhase2TPGAmplWeightIdMap * ecaltpgWeightMap, const EcalTPGWeightGroup *ecaltpgWeightGroup );
  
};

#endif

