using namespace std; // necessary!
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixLinearizer.h>
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "RecoCaloTools/Navigation/interface/EcalBarrelNavigator.h"
//#include "Calorimetry/EcalEndcap/interface/EcalEndcapBase.h"
#include <iostream>

namespace tpg {

  EcalFenixLinearizer::EcalFenixLinearizer(EcalBarrelTopology *top): ebTopol_(top)
{
    //UB FIXME : should come from a database!!!
    FILE * fin=fopen("../data/FenixStripPedMult.txt","r");
    //    if (!fin) throw an exception
    unsigned int tow, str, strXtal, gain, ped, bid1, bid2;
    for(int i=0; i<75; i++)
      {
	//	fscanf(fin,"%x %d %d %d %x %x %x\n",&tow, &str, &strXtal, &gain, &ped, &bid1, &bid2);
	fscanf(fin,"%x %u %u %u %x %x %x\n",&tow, &str, &strXtal, &gain, &ped, &bid1, &bid2);

	baseLine_[str-1][strXtal-1][gain-1]=ped;
	//	cout<<"read in FenixStripPedMult.txt: "<<tow<<" "<<str<<" "<<strXtal<<" "<<gain<<" "<<ped<<endl;
      }

    fclose(fin);
  }

  EcalFenixLinearizer::~EcalFenixLinearizer(){
  }


  EBDataFrame EcalFenixLinearizer::process(EBDataFrame df,int stripnr)
  {
    // find correct crystal index
    // why not coded in CellID?

    int crystalNumberInStrip=findCrystalNumber(df.id());

    EBDataFrame frame(df.id());
    frame.setSize(SIZEMAX);
    for (int i=0;i<df.size();i++) {
      setInput(df[i],stripnr, crystalNumberInStrip);
      frame.setSample(i,EcalMGPASample(process(),gainID_)); 
    }
    return frame;
  }	 

  int EcalFenixLinearizer::process()
  {
    int output=(uncorrectedSample_-base_); //Substract base
    if(output<0) return 0;
    output=(output*mult_)>>shift_;        //Apply multiplicative factor
    if(output>0X3FFFF)output=0X3FFFF;         //Saturation if too high
    return output;
  }

int EcalFenixLinearizer::setInput(EcalMGPASample RawSam, int stripNum, int XtalNumberInStrip)
{
  strip_=stripNum;
  if(RawSam.raw()>0X3FFF)
    {
      std::cout<<"ERROR IN INPUT SAMPLE OF FENIX LINEARIZER"<<std::endl;
      return -1;
    }
  uncorrectedSample_=RawSam.adc(); //uncorrectedSample_ is coded in the 12 LSB
  gainID_=RawSam.gainId();       //uncorrectedSample_ is coded in the 2 next bits!
  //UB FIXME: necessary or not???  gainID_--;       //To have gainID_ in the range [0; 2] and not [1;3]!! //UB FIXME??
  if (gainID_>=3)  cout<< "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! gain ID= "<<gainID_<<endl;

  base_=getBase(XtalNumberInStrip);
  mult_=getMult();
  shift_=getShift();
  return 1;
}

int EcalFenixLinearizer::getBase(int XtalNumberInStrip) const
{
  
  return baseLine_[strip_][XtalNumberInStrip][gainID_];
  
}
int EcalFenixLinearizer::getMult() const
{
 //Mult values from ??? config file 2004
 if(gainID_==2) return 0X60;
 else  return 0XFF;
}

int EcalFenixLinearizer::getShift() const
{
  //shift values from ??? config File 2004
  int shift=99;
  if(gainID_==0) shift=5;
  if(gainID_==1) shift=4;
  if(gainID_==2) shift=0;

  return shift+2; //to be consistant with the Test-Beam Data
}

//We know a tower numbering is:
       // S1 S2 S3 S4 S5
       //
       // 4  5  14 15 24
       // 3  6  13 16 23
       // 2  7  12 17 22
       // 1  8  11 18 21
       // 0  9  10 19 20


  int EcalFenixLinearizer::findCrystalNumber(const EBDetId &id) {
  // finds crystal number inside the same strip (0-4)

    int crystnr = numberOfCrystalsInStrip;
    EBDetId tmp=id;
    EcalBarrelNavigator nav(tmp,ebTopol_);
    unsigned int  towernumber = tmp.tower().rawId();
    while ( towernumber==tmp.tower().rawId()) {
      --crystnr;
      tmp=nav.south();
      if (tmp.null()) {
	cout <<" south border "<<endl; // should never happen
	break; //no cell south of this one
      }
    }
    return  crystnr;
  }
} //End of namespace tpg

