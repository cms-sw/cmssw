#ifndef SimDataFormats_GeneratorProducts_LHEXMLStringProduct_h
#define SimDataFormats_GeneratorProducts_LHEXMLStringProduct_h

/** \class LHEXMLStringProduct
 *
 */

#include <string>
#include <vector>

class LHEXMLStringProduct {
public:
  
  // constructors, destructors
  LHEXMLStringProduct();
  LHEXMLStringProduct(const std::string& content);
  LHEXMLStringProduct(const std::vector<std::string>& content);
  virtual ~LHEXMLStringProduct();
  
  // getters
  const std::vector<std::string>& getStrings() const{
    return content_; 
  }

  // merge method. It will be used when merging different jobs populating the same lumi section
  bool mergeProduct(LHEXMLStringProduct const &other);
  
  
private:
  std::vector<std::string> content_;

};


#endif // SimDataFormats_GeneratorProducts_LHEXMLStringProduct_h
