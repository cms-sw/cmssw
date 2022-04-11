#ifndef UTILITIES_OPENSSL_OPENSSL_INIT_H
#define UTILITIES_OPENSSL_OPENSSL_INIT_H
#include <openssl/evp.h>

#if OPENSSL_API_COMPAT < 0x10100000L
#define EVP_MD_CTX_new  EVP_MD_CTX_create
#define EVP_MD_CTX_free EVP_MD_CTX_destroy
#endif

namespace cms {
  void openssl_init();
} //cms

#endif //UTILITIES_OPENSSL_OPENSSL_INIT_H
