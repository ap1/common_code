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

// formula source: http://en.wikipedia.org/wiki/Volume_of_an_n-ball
inline float sphereVolume(float r, int dimension){

  float pi = M_PI;

  switch(dimension){
    case  0: return ( 1.0f);                                                  break;
    case  1: return ( 2.0f)                           * r;                    break;
    case  2: return ( 1.0f)          * pi             * r*r;                  break;
    case  3: return ( 4.0f/  3.0f)   * pi             * r*r*r;                break;
    case  4: return ( 1.0f/  2.0f)   * pi*pi          * r*r*r*r;              break;
    case  5: return ( 8.0f/ 15.0f)   * pi*pi          * r*r*r*r*r;            break;
    case  6: return ( 1.0f/  6.0f)   * pi*pi*pi       * r*r*r*r*r*r;          break;
    case  7: return (16.0f/105.0f)   * pi*pi*pi       * r*r*r*r*r*r*r;        break;
    case  8: return ( 1.0f/ 24.0f)   * pi*pi*pi*pi    * r*r*r*r*r*r*r*r;      break;
    case  9: return (32.0f/945.0f)   * pi*pi*pi*pi    * r*r*r*r*r*r*r*r*r;    break;
    case 10: return ( 1.0f/120.0f)   * pi*pi*pi*pi*pi * r*r*r*r*r*r*r*r*r*r;  break;
    case 11: return (64.0f/10395.0f) * pi*pi*pi*pi*pi * r*r*r*r*r*r*r*r*r*r*r;break;
    default: assertPrint(0, "Sphere volume not defined for %d dimensions\n",
               dimension);
  };

  return 0.0f;
}

#endif
