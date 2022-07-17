#include "Utilities/OpenSSL/interface/openssl_init.h"
#include <iostream>
#include <iomanip>
#include <sstream>

using namespace std;
int main() {
  unsigned char hash[EVP_MAX_MD_SIZE];
  unsigned int md_len = 0;
  cms::openssl_init();
  EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
  const EVP_MD *md = EVP_get_digestbyname("MD5");

  EVP_DigestInit_ex(mdctx, md, NULL);
  EVP_DigestUpdate(mdctx, "foo-bar-bla", 11);
  EVP_DigestFinal_ex(mdctx, hash, &md_len);
  EVP_MD_CTX_free(mdctx);

  char tmp[EVP_MAX_MD_SIZE * 2 + 1];
  // re-write bytes in hex
  for (unsigned int i = 0; i < md_len; i++) {
    ::sprintf(&tmp[i * 2], "%02x", hash[i]);
  }
  tmp[md_len * 2] = 0;
  string sha = tmp;
  string sha_result = "ad42a5e51344e880010799a7b7c612fc";
  if (sha != sha_result) {
    cout << "Failed: SHA1 Mismatch:" << sha << " vs " << sha_result << endl;
    return 1;
  }
  cout << "Passed: SHA1 match:" << sha << " vs " << sha_result << endl;
  return 0;
}
