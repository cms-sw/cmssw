#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalBarrelFenixTcp.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcpFormat.h>
#include <iostream>

  EcalBarrelFenixTcp::EcalBarrelFenixTcp() { 
    for (int i=0;i<nStripsPerTower_;i++) bypasslin_[i] = new EcalFenixBypassLin();
    adder_= new EcalFenixEtTot();
    maxOf2_=new EcalFenixMaxof2();
    formatter_= new EcalFenixTcpFormat();
    fgvb_= new EcalFenixFgvbEB();

  }
  EcalBarrelFenixTcp::~EcalBarrelFenixTcp() {
    for (int i=0;i<nStripsPerTower_;i++) delete bypasslin_[i];
    delete adder_; 
    delete maxOf2_;
    delete formatter_;
    delete fgvb_;
  }
 
 void EcalBarrelFenixTcp:: process(std::vector<std::vector<int> > & tpframetow, std::vector<EcalTriggerPrimitiveSample> &tcp_out) 
  { 

    //call bypasslin
    std::vector< std::vector<int> >  bypasslin_out;
    for (unsigned int istrip=0;istrip<tpframetow.size();istrip ++){
      std::vector<int> stripin= tpframetow[istrip];
      for (unsigned int is=0;is<stripin.size();is++){
	//	std::cout<<stripin[is]<<" ";
      }
      //      std::cout<<endl;
      bypasslin_out.push_back(this->getBypasslin(istrip)->process(stripin));
    }
//     //this is a test
//     std::cout<<"bypasslinout = "<<endl;
//     for (unsigned int istrip=0;istrip<bypasslin_out.size();istrip ++){
//       vector<int> stripin= bypasslin_out[istrip];
//       for (unsigned int is=0;is<stripin.size();is++){
// 	std::cout<<stripin[is]<<" ";
//       }
//       std::cout<<endl;
//     }
    
    
    //call adder
    std::vector<int> adder_out;
    adder_out = this->getAdder()->process(bypasslin_out);
    //this is a test:
//     std::cout<< "output of adder is a vector of size: "<<adder_out.size()<<endl; 
//     std::cout<< "value : "<<endl;
//     for (unsigned int i =0; i<adder_out.size();i++){
//       std::cout <<" "<<adder_out[i];
//     }    
//     std::cout<<endl;

    //call maxof2
    std::vector<int> maxof2_out;
    maxof2_out = this->getMaxOf2()->process(bypasslin_out);
     //this is a test:
//     std::cout<< "output of maxof2 is a vector of size: "<<adder_out.size()<<endl; 
//     std::cout<< "value : "<<endl;
//     for (unsigned int i =0; i<maxof2_out.size();i++){
//       std::cout <<" "<<maxof2_out[i];
//     }    
//     std::cout<<endl;

    //call fgvb
    std::vector<int> fgvb_out;
    fgvb_out = this->getFGVB()->process(adder_out,maxof2_out);
     //this is a test:
//     std::cout<< "output of fgvb is a vector of size: "<<fgvb_out.size()<<endl; 
//     std::cout<< "value : "<<endl;
//     for (unsigned int i =0; i<fgvb_out.size();i++){
//       std::cout <<" "<<fgvb_out[i];
//     }    
//     std::cout<<endl;

    // call formatter
    this->getFormatter()->process(adder_out,fgvb_out,tcp_out);
     //this is a test:
    //    std::cout<< "output of formatter is a vector of size: "<<formatter_out.size()<<endl; 
    //    std::cout<< "value : "<<endl;
//     for (unsigned int i =0; i<formatter_out.size();i++){
//            std::cout <<" "<<i<<" "<<formatter_out[i];
//     }    
//     std::cout<<endl;


    return;
      

  }

