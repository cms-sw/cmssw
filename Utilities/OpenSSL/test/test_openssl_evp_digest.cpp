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
  const EVP_MD *md = EVP_get_digestbyname("SHA1");

  EVP_DigestInit_ex(mdctx, md, NULL);
  EVP_DigestUpdate(mdctx, "foo", 3);
  EVP_DigestUpdate(mdctx, "bar", 3);
  EVP_DigestFinal_ex(mdctx, hash, &md_len);
  EVP_MD_CTX_free(mdctx);

  stringstream ss;
  for (unsigned int i = 0; i < md_len; i++) {
    ss << hex << setw(2) << setfill('0') << (int)hash[i];
  }
  hash[md_len] = 0;
  string sha_result = "8843d7f92416211de9ebb963ff4ce28125932878";
  string sha = ss.str();
  if (sha != sha_result) {
    cout << "Failed: SHA1 Mismatch:" << sha << " vs " << sha_result << endl;
    return 1;
  }
  cout << "Passed: SHA1 match:" << sha << " vs " << sha_result << endl;
  return 0;
}
