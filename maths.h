#ifndef maths_h
#define maths_h

#include <cmath>
#include "vecs.h"

inline float minf(const float& a, const float& b) { return (a < b) ? a : b; }
inline float maxf(const float& a, const float& b) { return (a > b) ? a : b; }

// ------------------------------------------------------------------
// function to divide two numbers in floating point,
// take the ceiling, and return another integer
// very useful for CUDA 
// ------------------------------------------------------------------
inline int ceil_int_div(int numer, int denom){
    return (int)ceilf((float)numer / (float) denom);
}

template <class T>
inline T lerp(T& val1, T& val2, float t){
  return (val1 * (1.0f - t) +
          val2 * (t));
}

template <class T>
inline T interpolateBary(   T& v0,    T& v1,         T& v2, 
                         float alpha, float beta, float gamma){
  return (v0 * alpha + 
          v1 * beta + 
          v2 * gamma);
}

template <class T>
inline T interpolateBary(T& v0, T& v1, T& v2, vec3f& interpolant){
  return (v0 * interpolant.x() + 
          v1 * interpolant.y() + 
          v2 * interpolant.z());
}

inline float dist2d(const float& x1, const float& y1, 
const float& x2, const float& y2){
  return sqrtf( (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) );
}


#endif
