#include "TObject.h"
#include "TF1.h"
#include "TMath.h"
#include <iostream>

#include "TkPulseShape.h"

// ROOT script to generate fast and accurate pulse shape functions
// use (in ROOT interpreter) :
//.L parametrization.C+
// parametrizePulse deconv(1)
// parametrizePulse peak(2)
// deconv.generateCode(-30,35,0.1); > deconv.C
// peak.generateCode(-60,400,0.5); > peak.C

class parametrizePulse {
 public:
  // constructor: takes the mode as input
  // mode 1 = deconvolution mode
  // mode 2 = peak mode 
  parametrizePulse(int fitMode = 1);
  // destructor
  virtual ~parametrizePulse();
  // returns a pointer to the theoretical pulse function
  TF1* getTheoreticalPulse() const;
  // gives the correction factor for a given offset
  float getCorrectionFactor(float offset) const;
  // returns an array evaluated in a given range with a given step
  void getArray(float low, float high, float step);
  // generates code to evaluate the function
  void generateCode(float low, float high, float step);
 private:
  // member functions
  // members
  int fitMode_;
};

parametrizePulse::parametrizePulse(int fitMode) 
{
  fitMode_ = fitMode; // 1 = deconvolution, 2 = peak
}

parametrizePulse::~parametrizePulse() 
{

}

float parametrizePulse::getCorrectionFactor(float offset) const {
    TF1* pulse = getTheoreticalPulse();
    if(fitMode_ == 1) { return 1./pulse->Eval(offset); }
    else { return 1./pulse->Eval(100+offset); }
}

TF1* parametrizePulse::getTheoreticalPulse() const { 
    TF1* output = NULL;
    if(fitMode_ == 1) {
      TF1* deconv_fitter = TkPulseShape::GetDeconvFitter();
      deconv_fitter->SetParameters(0,-2.816035506,0.066320437,50,20);
      output = deconv_fitter;
    }
    else {
      TF1* peak_fitter = TkPulseShape::GetPeakFitter();
      peak_fitter->SetParameters(0,-45.90116379,.056621237,50,20);
      output = peak_fitter;
    }
    return output;
}

void parametrizePulse::getArray(float low, float high, float step)
{
  TF1* pulse = getTheoreticalPulse();
  double end = high+step/2.;
  double maximum = pulse->GetMaximum(); 
  for(float val=low;val<end;val+=step) {
    std::cout << "(" << val << ", " << pulse->Eval(val)/maximum << ")" << std::endl;
  }
}

void parametrizePulse::generateCode(float low, float high, float step)
{
  unsigned int size =0;
  TF1* pulse = getTheoreticalPulse();
  double end = high+step/2.;
  double maximum = pulse->GetMaximum(); 
  double maximumX = pulse->GetMaximumX(); 
  for(float val=low;val<end;val+=step) ++size;
  std::cout << "float evaluate(float x) {" << std::endl;
  std::cout << "  // Automatically generated using parametrizePulse::generateCode(low="
            << low << ", high=" << high << ", step="<< step << ")" << std::endl;
  std::cout << "  float valuesArray[" << size << "] = { " ;
  size=0;
  for(float val=low;val<end;val+=step) {
    if(size) std::cout << ", " ;
    std::cout << pulse->Eval(val+maximumX)/maximum;
    if(!((++size)%5)) std::cout << std::endl << "                           ";
  }
  std::cout << " };" << std::endl;
  std::cout << "  if(x<("<<low<<")) return 0;" << std::endl;
  std::cout << "  if(x>("<<high<<")) return 0;" << std::endl;
  std::cout << "  return valuesArray[unsigned int(((x-("<<low<<"))/("<<step<<"))+0.5)];" << std::endl;
  std::cout << "}" << std::endl;
}

