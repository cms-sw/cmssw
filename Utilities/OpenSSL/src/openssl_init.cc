#include "Utilities/OpenSSL/interface/openssl_init.h"
#include <mutex>

namespace cms {
  void openssl_init() {
    static std::once_flag flag;
    std::call_once(flag, []() {
#if OPENSSL_API_COMPAT < 0x10100000L
      OpenSSL_add_all_digests();
#else
      OPENSSL_init_crypto();
#endif
    });
  }
}  // namespace cms
