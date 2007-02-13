#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixAmplitudeFilter.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/DBInterface.h>

EcalFenixAmplitudeFilter::EcalFenixAmplitudeFilter(DBInterface * db)
  :db_(db), inputsAlreadyIn_(0), shift_(6) {
  }

EcalFenixAmplitudeFilter::~EcalFenixAmplitudeFilter(){
}


int EcalFenixAmplitudeFilter::setInput(int input)
{
  if(input>0X3FFFF)
    {
      std::cout<<"ERROR IN INPUT OF AMPLITUDE FILTER"<<std::endl;
      return -1;
    }
  if(inputsAlreadyIn_<5)
    {
      buffer_[inputsAlreadyIn_]=input;
      inputsAlreadyIn_++;
    }
  else
    {
      for(int i=0; i<4; i++) buffer_[i]=buffer_[i+1];
      buffer_[4]=input;
    }
  return 1;
}


std::vector<int> EcalFenixAmplitudeFilter::process(std::vector<int> addout)
{
  std::vector<int> output;
  // test
  inputsAlreadyIn_=0;
  for (unsigned int i =0;i<5;i++) buffer_[i]=0;
  
  // test end
  
  for (unsigned int i =0;i<addout.size();i++){
    
    setInput(addout[i]);
    int outone= process();
    output.push_back(outone);
  }
  // shift the result by 1!
  for (unsigned int i=0 ; i<(output.size());i++){
    if (i!=output.size()-1) output[i]=output[i+1];
    else output[i]=0;
  }  
    return output;
}

int EcalFenixAmplitudeFilter::process()
{
  //UB FIXME: 5
  if(inputsAlreadyIn_<5) return 0;
  int output=0;
  for(int i=0; i<5; i++)
    {
      output+=(weights_[i]*buffer_[i])>>shift_;
    }
  if(output<0) output=0;
  if(output>0X3FFFF)  output=0X3FFFF;
  return output;
}

void EcalFenixAmplitudeFilter::setParameters(int SM, int towerInSM, int stripInTower)
{
  std::vector<unsigned int> params = db_->getStripParameters(SM, towerInSM, stripInTower) ;
  for (int i=0 ; i<5 ; i++) weights_[i] = params[i+1] ;
}




