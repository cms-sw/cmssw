#ifndef SimDataFormats_GeneratorProducts_LHEEventProductLite_h
#define SimDataFormats_GeneratorProducts_LHEEventProductLite_h

#include <memory>
#include <vector>
#include <string>

#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"
#include "SimDataFormats/GeneratorProducts/interface/PdfInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/WeightsInfo.h"

class LHEEventProductLite {
    public:
    typedef gen::PdfInfo PDF;
    typedef float WGT;

    void copyWeightsVector (std::vector<float> & outV,
                            const std::vector<gen::WeightsInfo> & inV)
      {
        for (unsigned int i = 0 ; i < inV.size () ; ++i)
        	outV.push_back (inV.at (i).wgt) ;
        return ;
      }

    LHEEventProductLite() {}
        LHEEventProductLite(const lhef::HEPEUP hepeup) : 
      hepeup_(hepeup), originalXWGTUP_(0) {}
    LHEEventProductLite(const lhef::HEPEUP &hepeup,
            const double originalXWGTUP) : 
      hepeup_(hepeup), originalXWGTUP_(originalXWGTUP) {}

    LHEEventProductLite(const LHEEventProduct * lep)
      {
        hepeup_         = lep->hepeup () ;
        originalXWGTUP_ = lep->originalXWGTUP () ;
        npLO_           = lep->npLO () ;
        npNLO_          = lep->npNLO () ;

        if (lep->pdf ())
          {
            this->setPDF (*lep->pdf ()) ;
          }
//        pdf_.reset(new PDF()) ; 
        this->setScales (lep->scales ()) ; 
        copyWeightsVector (weights_, lep->weights ()) ;

//        std::cout << "npLO " << npLO_  << " " << lep->npLO ()  << std::endl ;
//        std::cout << "npNLO " << npNLO_ << " " << lep->npNLO () << std::endl ;

      }

    ~LHEEventProductLite() {}

    void setPDF (const PDF &pdf) { pdf_.reset(new PDF(pdf)); }
    void addWeight(const WGT& wgt) {       
      weights_.push_back(wgt);
    }

    double originalXWGTUP() const { return originalXWGTUP_; }
    const std::vector<WGT>& weights() const { return weights_; }

    const std::vector<float> &scales() const { return scales_; }
    void setScales(const std::vector<float> &scales) { scales_ = scales; }
    
    int npLO() const { return npLO_; }
    int npNLO() const { return npNLO_; }
        
    void setNpLO(int n) { npLO_ = n; }
    void setNpNLO(int n) { npNLO_ = n; }    
    
    const lhef::HEPEUP &hepeup() const { return hepeup_; }
    const PDF *pdf() const { return pdf_.get(); }

    class const_iterator {
        public:
        typedef std::forward_iterator_tag  iterator_category;
        typedef std::string                value_type;
        typedef std::ptrdiff_t             difference_type;
        typedef std::string                *pointer;
        typedef std::string                &reference;

        const_iterator() : line_(npos_) {}
        ~const_iterator() {}

        inline bool operator == (const const_iterator &other) const
        { return line_ == other.line_ ; }
        inline bool operator != (const const_iterator &other) const
        { return !operator == (other); }

        inline const_iterator &operator ++ ()
        { next(); return *this; }
        inline const_iterator operator ++ (int dummy)
        { const_iterator orig = *this; next(); return orig; }

        const std::string &operator * () const { return tmp_ ; }
        const std::string *operator -> () const { return &tmp_ ; }

        private:
        friend class LHEEventProductLite ;

        void next();

        const LHEEventProductLite  *event_ ;
        unsigned int               line_ ;
        std::string                tmp_ ;
        static const unsigned int  npos_ = 99999;
    };

    const_iterator begin() const;
    inline const_iterator end() const { return const_iterator(); }

    private:
    lhef::HEPEUP                    hepeup_;
    std::auto_ptr<PDF>              pdf_;
    std::vector<WGT>                weights_;
    float                           originalXWGTUP_;
    std::vector<float>              scales_; //scale value used to exclude EWK-produced partons from matching
    int                             npLO_;   //number of partons for LO process (used to steer matching/merging)
    int                             npNLO_;  //number of partons for NLO process (used to steer matching/merging)
};

#endif // GeneratorEvent_LHEInterface_LHEEventProductLite_h
