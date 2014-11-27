#include "SimCalorimetry/HGCSimProducers/interface/HGCFEElectronics.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

using namespace std;

//
template<class D>
HGCFEElectronics<D>::HGCFEElectronics(const edm::ParameterSet &ps,float bxTime) : bxTime_(bxTime)
{
  fwVersion_      = ps.getParameter< uint32_t >("fwVersion");
  lsbInMIP_       = ps.getParameter<double>("lsbInMIP");
  if( ps.exists("mipInfC") && ps.exists("gainChangeInfC") && fwVersion_==WITHTOT)
    {
      mipInfC_        =  ps.getParameter< double >("mipInfC");
      gainChangeInfC_ =  ps.getParameter< double >("gainChangeInfC");
      cout << "Gain change will be active for hits >" << gainChangeInfC_/mipInfC_ << " MIP" << endl;
    }
  else
    {
      cout << "Given no gain change is set/or fw is not configured, will not activate ToT in digi" << endl;
      mipInfC_ = 0.;
      gainChangeInfC_ = -1;
    }
  shaperN_        = ps.getParameter< double >("shaperN");
  shaperTau_      = ps.getParameter< double >("shaperTau");
}


//
template<class D>
void HGCFEElectronics<D>::runSimpleShaper(D &dataFrame)
{
  std::vector<uint16_t> oldADC(dataFrame.size());
  for(int it=0; it<dataFrame.size(); it++)
    {
      uint16_t gain=dataFrame[it].gain();
      oldADC[it]=dataFrame[it].adc();
      uint16_t newADC(oldADC[it]);
      
      if(shaperN_*shaperTau_>0){
	for(int jt=0; jt<it; jt++)
	  {
	    float relTime(bxTime_*(it-jt)+shaperN_*shaperTau_);	
	    newADC += uint16_t(oldADC[jt]*pow(relTime/(shaperN_*shaperTau_),shaperN_)*exp(-(relTime-shaperN_*shaperTau_)/shaperTau_));	      
	  }
      }
      
      HGCSample newSample;
      newSample.set(gain,newADC);
      dataFrame.setSample(it,newSample);
    }
}

//
template<class D>
void HGCFEElectronics<D>::runShaperWithToT(D &dataFrame)
{
  std::vector<uint16_t> oldADC(dataFrame.size());
  std::vector<bool> busyFlags(dataFrame.size(),false),totFlags(dataFrame.size(),false);

  //bool doDebug(false);
  //for(int it=0; it<dataFrame.size(); it++) if (dataFrame[it].adc()>100) doDebug=true;

  for(int it=0; it<dataFrame.size(); it++)
    {
      uint16_t gain=dataFrame[it].gain();
      oldADC[it]=dataFrame[it].adc();
      uint16_t newADC(oldADC[it]);      
      
      //check if this bx has been already put in busy state 
      //by ToT occuring in previous bunch(es)
      //if so then nothing will be readout
      if(busyFlags[it])
	{
	  gain=0;
	  newADC=0;
	}
      //readout is enabled
      else
	{
	  //update with the leakage from previous bunches
	  for(int jt=0; jt<it; jt++)
	    {
	      if(totFlags[jt] || busyFlags[jt]) continue;
	      float relTime(bxTime_*(it-jt));
	      newADC += uint16_t(oldADC[jt]*pow(1+relTime/(shaperTau_*shaperN_),shaperN_)*exp(-relTime/shaperTau_));
	    }
	  
	  //check if TOT is enabled
	  float charge(newADC*lsbInMIP_*mipInfC_);
	  if(charge>gainChangeInfC_)
	    {
	      //change to TOT
	      gain=1;
	      totFlags[it]=true;

	      //compute how long it will take to integrate this charge
	      float integTime(0);
	      if(charge<500) integTime=166.406*pow(charge*1e-3,0.741);
	      else           integTime=34.736*pow(charge*1e-3,0.845)+92.64;
	      
	      //foward declare the next bunches as busy
	      uint16_t busyBxs(floor(integTime/bxTime_)+1);
	      //if(doDebug) cout << it <<  "bunch will raise busy for " << busyBxs 
	      //<< "(integTime=" << integTime << ")" << endl;
	      
	      for(int jt=it; jt<it+busyBxs || jt<dataFrame.size() ; jt++) 
		busyFlags[jt]=true;
	    }
	}
      
      HGCSample newSample;
      newSample.set(gain,newADC);
      dataFrame.setSample(it,newSample);
    }

  //if(doDebug)
  //  for(int it=0; it<dataFrame.size(); it++)
  //    std::cout << oldADC[it] << " busy=" << busyFlags[it] << " ToT=" << totFlags[it] << " finalword gain=" << dataFrame[it].gain() << " adc=" << dataFrame[it].adc() << std::endl;
}
