#include <openssl/md5.h>
#include <iostream>
#include <iomanip>
#include <sstream>

using namespace std;
int main() {
  unsigned char hash[MD5_DIGEST_LENGTH];
  string data("foo-bar-bla");
  MD5((unsigned char*)data.c_str(), 11, hash);
  stringstream ss;
  for (unsigned int i = 0; i < MD5_DIGEST_LENGTH; i++) {
    ss << hex << setw(2) << setfill('0') << (int)hash[i];
  }
  string sha_result = "ad42a5e51344e880010799a7b7c612fc";
  string sha = ss.str();
  if (sha != sha_result) {
    cout << "Failed: SHA1 Mismatch:" << sha << " vs " << sha_result << endl;
    return 1;
  }
  cout << "Passed: SHA1 match:" << sha << " vs " << sha_result << endl;
  return 0;
}
