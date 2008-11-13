#ifndef SimDataFormats_GeneratorProducts_GenInfoProduct_h
#define SimDataFormats_GeneratorProducts_GenInfoProduct_h

/** \class GenInfoProduct
 *
 *  \author Filip Moortgat
 */


namespace edm {

  class GenInfoProduct {
  public:
    GenInfoProduct() : cs_( 0 ), cs2_(0), fe_(0) { }
    explicit GenInfoProduct( double cross_section );
    virtual ~GenInfoProduct() { }
    
    const double cross_section() const {return cs_ ; }
    const double external_cross_section() const {return cs2_ ; }
    const double filter_efficiency() const {return fe_ ; }
 
    void set_cross_section(double cross_section) { cs_ = cross_section ; }
    void set_external_cross_section(double cross_sect) { cs2_ = cross_sect ; }
    void set_filter_efficiency(double eff) { fe_ = eff ; }

    GenInfoProduct(GenInfoProduct const& x);

    virtual bool mergeProduct(const GenInfoProduct &other);
    virtual bool isProductEqual(const GenInfoProduct &other)
    { return cs_ == other.cs_ && cs2_ == other.cs2_ && fe_ == other.fe_; }

  private:
    double cs_;
    double cs2_;
    double fe_;

  };

}
#endif // SimDataFormats_GeneratorProducts_GenInfoProduct_h
