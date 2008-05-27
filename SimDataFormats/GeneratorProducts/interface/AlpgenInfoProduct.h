#ifndef AlpgenInfoProduct_h
#define AlpgenInfoProduct_h

/** \class AlpgenInfoProduct
 *
 *  \author 
 */

#include <vector>

namespace edm {

  class AlpgenInfoProduct {
  public:
    AlpgenInfoProduct() : nEv_(0), nTot_(0), subproc_(0), q_(0.) { }
    explicit AlpgenInfoProduct( int nEv );
    virtual ~AlpgenInfoProduct() { }
    
    // Get Lines of Alpgen .unw file and fill 
    // corresponding variables
    void EventInfo(const char* buffer);
    void InPartonInfo(const char* buffer);
    void OutPartonInfo(const char* buffer);

    const int nEv()  const {return nEv_  ;}
    const int nTot() const {return nTot_ ;}
    const int subproc() const {return subproc_ ;}
    const double Q() const {return q_ ;}

    const std::vector<int> lundIn() const {return lundIn_ ;}
    const std::vector<int> colorIn() const {return colorIn_ ;}
    const std::vector<int> colorBarIn() const {return colorBarIn_ ;}
    const std::vector<double> pzIn() const {return pzIn_ ;}

    const std::vector<int> lundOut() const {return lundOut_ ;}
    const std::vector<int> colorOut() const {return colorOut_ ;}
    const std::vector<int> colorBarOut() const {return colorBarOut_ ;}
    const std::vector<double> pxOut() const {return pxOut_ ;}
    const std::vector<double> pyOut() const {return pyOut_ ;}
    const std::vector<double> pzOut() const {return pzOut_ ;}
    const std::vector<double> massOut() const {return massOut_ ;}

    const int lundIn(unsigned int i)      const {return (i< lundIn_.size()  ? lundIn_[i]  : -999);}
    const int colorIn(unsigned int i)     const {return (i< colorIn_.size() ? colorIn_[i] : -999);}
    const int colorBarIn(unsigned int i)  const {return (i< colorBarIn_.size() ? colorBarIn_[i] : -999);}
    const double pzIn(unsigned int i)     const {return (i< pzIn_.size() ? pzIn_[i] : -999.);}

    const int lundOut(unsigned int i)      const {return (i< lundOut_.size()  ? lundOut_[i]  : -999);}
    const int colorOut(unsigned int i)     const {return (i< colorOut_.size() ? colorOut_[i] : -999);}
    const int colorBarOut(unsigned int i)  const {return (i< colorBarOut_.size() ? colorBarOut_[i] : -999);}
    const double pxOut(unsigned int i)     const {return (i< pxOut_.size() ? pxOut_[i] : -999.);}
    const double pyOut(unsigned int i)     const {return (i< pyOut_.size() ? pyOut_[i] : -999.);}
    const double pzOut(unsigned int i)     const {return (i< pzOut_.size() ? pzOut_[i] : -999.);}
    const double massOut(unsigned int i)   const {return (i< massOut_.size() ? massOut_[i] : -999.);}

    AlpgenInfoProduct(AlpgenInfoProduct const& x);
    
  private:

    // File format

    // nEv_        subproc_        nTot_           idk2_     qsq_ 
    // lundIn_[0]  colorIn_[0]  colorBarIn_[0]  pzIn_[0]  
    // lundIn_[1]  colorIn_[1]  colorBarIn_[1]  pzIn_[1]
    // lundOut_[0] colorOut_[0] colorBarOut_[0] pxOut_[0] pyOut_[0] pzOut_[0] massOut_[0]
    // ...


    // general information (first line)
    int nEv_;                // Event number
    int nTot_;               // total number of outgoing particles
    int subproc_;            // Subprocess
    double q_;               // Q for the event (might be a constant)

    // Properties of the 2 incoming partons 
    std::vector<int>    lundIn_;      // Lund id     incoming partons       
    std::vector<int>    colorIn_;     // Color       incoming partons        
    std::vector<int>    colorBarIn_;  // Anticolor   incoming partons        
    std::vector<double> pzIn_;        // pZ          incoming partons        

    // Properties of the outgoing partons
    std::vector<int> lundOut_;        // Lund id     outgoing partons    
    std::vector<int> colorOut_;       // Color       outgoing partons
    std::vector<int> colorBarOut_;    // Anticolor   outgoing partons  
    std::vector<double> pxOut_;       // pX          outgoing partons  
    std::vector<double> pyOut_;       // pY          outgoing partons  
    std::vector<double> pzOut_;       // pZ          outgoing partons  
    std::vector<double> massOut_;     // mass        outgoing partons  

  };

   
}
#endif
