#ifndef RecoTracker_PixelTrackFitting_test_test_common_h
#define RecoTracker_PixelTrackFitting_test_test_common_h

#include <algorithm>
#include <cassert>
#include <random>

template <class C>
__host__ __device__ void printIt(C* m) {
#ifdef TEST_DEBUG
  printf("\nMatrix %dx%d\n", (int)m->rows(), (int)m->cols());
  for (u_int r = 0; r < m->rows(); ++r) {
    for (u_int c = 0; c < m->cols(); ++c) {
      printf("Matrix(%d,%d) = %f\n", r, c, (*m)(r, c));
    }
  }
#endif
}

template <class C1, class C2>
bool isEqualFuzzy(C1 a, C2 b, double epsilon = 1e-6) {
  for (unsigned int i = 0; i < a.rows(); ++i) {
    for (unsigned int j = 0; j < a.cols(); ++j) {
      assert(std::abs(a(i, j) - b(i, j)) < std::min(std::abs(a(i, j)), std::abs(b(i, j))) * epsilon);
    }
  }
  return true;
}

bool isEqualFuzzy(double a, double b, double epsilon = 1e-6) {
  return std::abs(a - b) < std::min(std::abs(a), std::abs(b)) * epsilon;
}

template <typename T>
void fillMatrix(T& t) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 2.0);
  for (int row = 0; row < t.rows(); ++row) {
    for (int col = 0; col < t.cols(); ++col) {
      t(row, col) = dis(gen);
    }
  }
  return;
}

#endif
