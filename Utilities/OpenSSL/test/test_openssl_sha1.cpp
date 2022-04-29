#include <openssl/sha.h>
#include <iostream>
#include <iomanip>
#include <sstream>

using namespace std;
int main() {
  SHA_CTX ctx;
  SHA1_Init(&ctx);
  SHA1_Update(&ctx, "foo", 3);
  SHA1_Update(&ctx, "bar", 3);
  unsigned char hash[SHA_DIGEST_LENGTH];
  SHA1_Final(hash, &ctx);

  stringstream ss;
  for (unsigned int i = 0; i < SHA_DIGEST_LENGTH; i++) {
    ss << hex << setw(2) << setfill('0') << (int)hash[i];
  }
  string sha_result = "8843d7f92416211de9ebb963ff4ce28125932878";
  string sha = ss.str();
  if (sha != sha_result) {
    cout << "Failed: SHA1 Mismatch:" << sha << " vs " << sha_result << endl;
    return 1;
  }
  cout << "Passed: SHA1 match:" << sha << " vs " << sha_result << endl;
  return 0;
}
