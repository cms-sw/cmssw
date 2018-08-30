#ifndef SimDataFormats_GeneratorProducts_LHEXMLStringProduct_h
#define SimDataFormats_GeneratorProducts_LHEXMLStringProduct_h

/** \class LHEXMLStringProduct
 *
 */

#include <string>
#include <vector>
#include <cstdint>

class LHEXMLStringProduct {
public:
    
  // constructors, destructors
  LHEXMLStringProduct();
  LHEXMLStringProduct(const std::string& content);
  virtual ~LHEXMLStringProduct();
  
  // getters
  const std::vector<std::string>& getStrings() const{
    return content_; 
  }

  const std::vector<std::vector<uint8_t> >& getCompressed() const{
    return compressedContent_; 
  }
  
  void fillCompressedContent(std::istream &input, unsigned int initialSize = 4*1024*1024);
  void writeCompressedContent(std::ostream &output, unsigned int i) const;
  
  // merge method. It will be used when merging different jobs populating the same lumi section
  bool mergeProduct(LHEXMLStringProduct const &other);
  void swap(LHEXMLStringProduct& other);
  
private:
  std::vector<std::string> content_;
  std::vector<std::vector<uint8_t> > compressedContent_;

};


#endif // SimDataFormats_GeneratorProducts_LHEXMLStringProduct_h
