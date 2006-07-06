using namespace std;
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalBarrelFenixStrip.h>
#include <DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h>
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include <DataFormats/EcalDigi/interface/EBDataFrame.h>
#include <TTree.h>

namespace tpg {

  EcalBarrelFenixStrip::EcalBarrelFenixStrip(EcalBarrelTopology *top, const TTree *tree) { 
    for (int i=0;i<nCrystalsPerStrip_;i++) linearizer_[i] = new  EcalFenixLinearizer(top); 
    adder_ = new  EcalFenixEtStrip();
    amplitude_filter_ = new EcalFenixAmplitudeFilter();
    peak_finder_ = new  EcalFenixPeakFinder();
    formatter_= new EcalFenixStripFormat();
  }


  EcalBarrelFenixStrip::~EcalBarrelFenixStrip() {
    for (int i=0;i<nCrystalsPerStrip_;i++) delete linearizer_[i]; 
    delete adder_; 
    delete amplitude_filter_; 
    delete peak_finder_;
    delete formatter_;
  }

  // stripnr should be coded in CellID of EcalTrigPrim in the future
  std::vector<int>  EcalBarrelFenixStrip::process(std::vector<EBDataFrame> df, int stripnr)

  { 

    // call linearizer
    vector<EBDataFrame> lin_out;
    //this is a test:
    //    cout<<"EcalBarrelFenixStrip input is a vector of size: "<<df.size()<<endl;
    for (unsigned int cryst=0;cryst<df.size();cryst++) {
      EBDataFrame cc=df[cryst]; 
      //this is a test:
//       cout<<endl;
//       cout <<"cryst= "<<cryst<<" EBDataFrame is: "<<endl; 
//       for ( int i = 0; i<cc.size();i++){
// 	cout <<" "<<df[cryst][i];
// 	if (df[cryst][i].adc() > 210) cout<<" is great!!";
//       }
//       cout<<endl;

      //      const CellID & myid=cc.getMyCell();   //cellid of crystal
      //      const CellID coarser(myid.coarserGranularity()); //cellid of tower 

      // stripnr should be coded in CellID in the future
      lin_out.push_back(this->getLinearizer(cryst)->process(cc,stripnr));
    }
    //this is a test:
//     cout<< "output of linearizers is a vector of size: "<<lin_out.size()<<endl; 
//     for (unsigned int ix=0;ix<lin_out.size();ix++){
//       cout<< "cryst: "<<ix<<"  value : "<<endl;
//       for (int i =0; i<lin_out[ix].size();i++){
// 	cout <<" "<<lin_out[ix][i];
//       }
//     }
		
//    cout<<endl;
    // call adder
    vector<int> add_out;
    add_out = this->getAdder()->process(lin_out);
//     //this is a test:
//     cout<< "output of adder is a vector of size: "<<add_out.size()<<endl; 
//     cout<< "value : "<<endl;
//     for (unsigned int i =0; i<add_out.size();i++){
//       cout <<" "<<add_out[i];
//     }
      
//     cout<<endl;
    
 
    // call amplitudefilter
    vector<int> filt_out;
    filt_out= this->getFilter()->process(add_out);
    //this is a test:
//     cout<< "output of amplitude filter is a vector of size: "<<filt_out.size()<<endl; 
//     cout<< "value : "<<endl;
//     for (unsigned int i =0; i<filt_out.size();i++){
//       cout <<" "<<filt_out[i];
//     }    
//     cout<<endl;

    // call peakfinder
    vector<int> peak_out;
    peak_out =this->getPeakFinder()->process(filt_out);

    //this is a test:
//     cout<< "output of peak finder is a vector of size: "<<peak_out.size()<<endl; 
//     cout<< "value : "<<endl;
//     for (unsigned int i =0; i<peak_out.size();i++){
//       cout <<" "<<peak_out[i];
//     }    
//     cout<<endl;

    // call formatter
    vector<int> format_out(peak_out.size());
    format_out =this->getFormatter()->process(peak_out,filt_out);
     
    //this is a test:
//     cout<< "output of formatter is a vector of size: "<<format_out.size()<<endl; 
//     cout<< "value : "<<endl;
//     for (unsigned int i =0; i<format_out.size();i++){
//       cout <<" "<<format_out[i];
//     }    
//    cout<<endl;

     return format_out;

  }

} /* End of namespace tpg */

