#include <iostream>
#include <algorithm> 
#include <lzma.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/GeneratorProducts/interface/LHEXMLStringProduct.h"

using namespace edm;
using namespace std;


LHEXMLStringProduct::LHEXMLStringProduct()
{
}

LHEXMLStringProduct::LHEXMLStringProduct(const string& onelheoutput) :
  content_()
{
  content_.push_back(onelheoutput);
}


LHEXMLStringProduct::~LHEXMLStringProduct()
{
}

void LHEXMLStringProduct::fillCompressedContent(std::istream &input, unsigned int initialSize) {
  //create blob with desired size
  compressedContent_.emplace_back(initialSize);
  std::vector<uint8_t> &output = compressedContent_.back();
    
  //read buffer
  constexpr unsigned int bufsize = 4096;
  char inbuf[bufsize];
  
  const unsigned int threshsize = 32*1024*1024;
  
  //initialize lzma
  uint32_t preset = 9;
  lzma_stream strm = LZMA_STREAM_INIT;
  lzma_ret ret = lzma_easy_encoder(&strm, preset, LZMA_CHECK_CRC64);
  
  lzma_action action = LZMA_RUN;
  
  strm.next_in = reinterpret_cast<uint8_t*>(&inbuf[0]);
  strm.avail_in = 0;
  strm.next_out = output.data();
  strm.avail_out = output.size();
  
  unsigned int compressedSize = 0;
    
  while (ret==LZMA_OK) {
    //read input to buffer if necessary
    if (strm.avail_in == 0 && !input.eof()) {
      input.read(inbuf, bufsize);
      strm.next_in = reinterpret_cast<uint8_t*>(&inbuf[0]);
      strm.avail_in = input.gcount();
      if (input.eof()) {
        //signal to lzma that there is no more input
        action = LZMA_FINISH;
      }
    }

    //actual compression
    ret = lzma_code(&strm, action);
    
    //update compressed size
    compressedSize = output.size() - strm.avail_out;
    
    //if output blob is full and compression is still going, allocate more memory
    if (strm.avail_out == 0 && ret==LZMA_OK) {
      unsigned int oldsize = output.size();
      if (oldsize<threshsize) {
        output.resize(2*oldsize);
      }
      else {
        output.resize(oldsize + threshsize);
      }
      strm.next_out = &output[oldsize];
      strm.avail_out = output.size() - oldsize;
    }
    
  }
  
  lzma_end(&strm);
  
  if (ret!=LZMA_STREAM_END) {
    throw cms::Exception("CompressionError")
      << "There was a failure in LZMA compression in LHEXMLStringProduct.";
  }
  
  //trim output blob
  output.resize(compressedSize);
  
}

void LHEXMLStringProduct::writeCompressedContent(std::ostream &output, unsigned int i) const {
  
  //initialize lzma
  lzma_stream strm = LZMA_STREAM_INIT;
  lzma_ret ret = lzma_stream_decoder(&strm, UINT64_MAX, LZMA_CONCATENATED);
  //all output available from the start, so start "close out" immediately
  lzma_action action = LZMA_FINISH;
  
  //write buffer
  constexpr unsigned int bufsize = 4096;
  char outbuf[bufsize];
  
  const std::vector<uint8_t> &input = compressedContent_[i];
  
  strm.next_in = input.data();
  strm.avail_in = input.size();
  strm.next_out = reinterpret_cast<uint8_t*>(&outbuf[0]);
  strm.avail_out = bufsize;
  
  while (ret==LZMA_OK) {
    
    ret = lzma_code(&strm, action);
    
    //write to stream
    output.write(outbuf,bufsize-strm.avail_out);
    
    //output buffer full, recycle
    if (strm.avail_out==0 && ret==LZMA_OK) {
      strm.next_out = reinterpret_cast<uint8_t*>(&outbuf[0]);
      strm.avail_out = bufsize;
    }
  }
    
  lzma_end(&strm);
  
  if (ret!=LZMA_STREAM_END) {
    throw cms::Exception("DecompressionError")
      << "There was a failure in LZMA decompression in LHEXMLStringProduct.";
  }
  
}



bool LHEXMLStringProduct::mergeProduct(LHEXMLStringProduct const &other)
{
  content_.insert(content_.end(), other.getStrings().begin(), other.getStrings().end());
  compressedContent_.insert(compressedContent_.end(), other.getCompressed().begin(), other.getCompressed().end());
  return true;
}

void LHEXMLStringProduct::swap(LHEXMLStringProduct& other) {
  content_.swap(other.content_);
  compressedContent_.swap(other.compressedContent_);
}
