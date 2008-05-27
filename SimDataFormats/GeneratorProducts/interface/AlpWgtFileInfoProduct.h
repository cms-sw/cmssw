#ifndef AlpWgtFileInfoProduct_h
#define AlpWgtFileInfoProduct_h

/** \class AlpWgtFileInfoProduct
 *
 *  \author Maurizio Pierini
 */

#include <vector>

namespace edm {

   

  class AlpWgtFileInfoProduct {
  public:
    AlpWgtFileInfoProduct() {}
    //    explicit AlpWgtFileInfoProduct( double cross_section );
    virtual ~AlpWgtFileInfoProduct() { }
    
    const int    nevents()    const {return seed1_.size();}
    const int    seed1(unsigned int i) const {return (i< seed1_.size() ? seed1_[i] : -999); }
    const int    seed2(unsigned int i) const {return (i< seed2_.size() ? seed2_[i] : -999); }
    const double wgt1(unsigned int i) const {return (i< wgt1_.size()  ? wgt1_[i]  : -999.); }
    const double wgt2(unsigned int i) const {return (i< wgt2_.size()  ? wgt2_[i]  : -999.); }

    const std::vector<int> seed1() const {return seed1_;}
    const std::vector<int> seed2() const {return seed2_;}
    const std::vector<double> wgt1()  const {return wgt1_;}
    const std::vector<double> wgt2()  const {return wgt2_;}

    void AddEvent(const char* buffer);

    AlpWgtFileInfoProduct(AlpWgtFileInfoProduct const& x);

  private:
    std::vector<int> seed1_;
    std::vector<int> seed2_;
    std::vector<double> wgt1_;
    std::vector<double> wgt2_;
  };

   
}
#endif
