#ifndef UTILITIES_XERCES_STRING_UTILS_H
#define UTILITIES_XERCES_STRING_UTILS_H

#include <xercesc/util/XercesDefs.hpp>
#include <xercesc/util/XMLString.hpp>
#include <memory>
#include <sstream>

namespace cms {
  namespace xerces {

#ifdef XERCES_CPP_NAMESPACE_USE
    XERCES_CPP_NAMESPACE_USE
#endif

    inline void dispose(XMLCh* ptr) { XMLString::release(&ptr); }
    inline void dispose(char* ptr) { XMLString::release(&ptr); }

    template <class CharType>
    class ZStr  // Zero-terminated string.
    {
    public:
      ZStr(CharType const* str) : m_array(const_cast<CharType*>(str), &dispose) {}

      CharType const* ptr() const { return m_array.get(); }

    private:
      std::unique_ptr<CharType, void (*)(CharType*)> m_array;
    };

    inline ZStr<XMLCh> uStr(char const* str) { return ZStr<XMLCh>(XMLString::transcode(str)); }

    inline ZStr<char> cStr(XMLCh const* str) { return ZStr<char>(XMLString::transcode(str)); }

    inline std::string toString(XMLCh const* toTranscode) { return std::string(cStr(toTranscode).ptr()); }

    inline unsigned int toUInt(XMLCh const* toTranscode) {
      std::istringstream iss(toString(toTranscode));
      unsigned int returnValue;
      iss >> returnValue;
      return returnValue;
    }

    inline bool toBool(XMLCh const* toTranscode) {
      std::string value = toString(toTranscode);
      if ((value == "true") || (value == "1"))
        return true;
      return false;
    }

    inline double toDouble(XMLCh const* toTranscode) {
      std::istringstream iss(toString(toTranscode));
      double returnValue;
      iss >> returnValue;
      return returnValue;
    }
  }  // namespace xerces
}  // namespace cms

#endif
