#include <SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalFenixPeakFinder.h>
#include <iostream>
  
EcalFenixPeakFinder::EcalFenixPeakFinder()
  :inputsAlreadyIn_(0){ 
}

  
EcalFenixPeakFinder::~EcalFenixPeakFinder(){
}

int EcalFenixPeakFinder::setInput(int input)
{
  if(inputsAlreadyIn_<3)
    {
      std::cout << " EcalFenixPeakFinder::setInput inputsAlreadyIn_<3 input " << input << std::endl;
      buffer_[inputsAlreadyIn_]=input;
      inputsAlreadyIn_++;
    }
  else
    {
      for(int i=0; i<2; i++) {
	buffer_[i]=buffer_[i+1];
	std::cout << " EcalFenixPeakFinder::setInput inputsAlreadyIn buffer " << buffer_[i] << std::endl;
      }
      buffer_[2]=input;
    }
  return 1;
}
  
int EcalFenixPeakFinder::process()
{

  if(inputsAlreadyIn_<3) return 0;
  if(buffer_[1]>buffer_[0] && buffer_[1]>buffer_[2]) return 1;
  else return 0;
    
}

std::vector<int> EcalFenixPeakFinder::process(std::vector<int> &filtout, std::vector<int> & output)
{
  
  // FIXME: 3
  inputsAlreadyIn_=0;
  for (unsigned int i =0;i<3;i++) buffer_[i]=0;

  //  std::vector<int> output;

  std::cout << "  EcalFenixPeakFinder::process(   " << filtout.size() << std::endl;    
  // attention, we have to shift by one, because the peak is found one too late
  for (unsigned int i =0;i<filtout.size();i++){
    
    setInput(filtout[i]);
    for (unsigned int i =0;i<3;i++){
      std::cout << " buffer_ " << buffer_[i];
    }
    std::cout << "  " << std::endl;

    if (i>0) {
      //      int outone = process();
      //      output.push_back(outone);
      output[i-1]= process();
    }
  }
  //  output.resize(filtout.size());

  return output;
}


