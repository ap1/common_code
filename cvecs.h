#ifndef cvecs_h
#define cvecs_h

#include <cstdio>
#include <cmath>
#include <string>

#ifndef M_PI
#define M_PI 3.1415926f
#endif

#include "maths.h"

// ---------------------------------------------------
//  cvecs: c-style vecs (but in C++ environment only)
// SAFE to use with CUDA
// ---------------------------------------------------

// ---------------------------------------------------
// Note: These are flat data structures. Do not put any 
// malloc / new inside its members
// ---------------------------------------------------


struct cvec2f{ float  x, y;       };
struct cvec3f{ float  x, y, z;    };
struct cvec4f{ float  x, y, z, w; };

struct cvec2i{ int    x, y;       };
struct cvec3i{ int    x, y, z;    };
struct cvec4i{ int    x, y, z, w; };

inline cvec2f   gencvec2f(const float& x, const float& y)                                { cvec2f v; v.x=x; v.y=y; return v;}
inline cvec3f   gencvec3f(const float& x, const float& y, const float& z)                { cvec3f v; v.x=x; v.y=y; v.z=z; return v;}
inline cvec4f   gencvec4f(const float& x, const float& y, const float& z, const float& w){ cvec4f v; v.x=x; v.y=y; v.z=z; v.w=w; return v;}

inline cvec2i   gencvec2i(const int& x, const int& y)                                    { cvec2i v; v.x=x; v.y=y; return v;}
inline cvec3i   gencvec3i(const int& x, const int& y, const int& z)                      { cvec3i v; v.x=x; v.y=y; v.z=z; return v;}
inline cvec4i   gencvec4i(const int& x, const int& y, const int& z, const int& w)        { cvec4i v; v.x=x; v.y=y; v.z=z; v.w=w; return v;}

inline cvec2f   negative(const cvec2f& v){ cvec2f n; n.x=-v.x; n.y=-v.y; return n; }
inline cvec3f   negative(const cvec3f& v){ cvec3f n; n.x=-v.x; n.y=-v.y; n.z=-v.z; return n; }
inline cvec4f   negative(const cvec4f& v){ cvec4f n; n.x=-v.x; n.y=-v.y; n.z=-v.z; n.w=-v.w; return n; }

inline float    magsqr(const cvec2f& v){ return (v.x*v.x + v.y*v.y); }
inline float    magsqr(const cvec3f& v){ return (v.x*v.x + v.y*v.y + v.z*v.z); }
inline float    magsqr(const cvec4f& v){ return (v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w); }

// operators: +, -, *, dot, cross
inline cvec2f   addvec(const cvec2f& v1, const cvec2f& v2){ cvec2f s; s.x=v1.x+v2.x; s.y=v1.y+v2.y; return s; }
inline cvec3f   addvec(const cvec3f& v1, const cvec3f& v2){ cvec3f s; s.x=v1.x+v2.x; s.y=v1.y+v2.y; s.z=v1.z+v2.z; return s; }
inline cvec4f   addvec(const cvec4f& v1, const cvec4f& v2){ cvec4f s; s.x=v1.x+v2.x; s.y=v1.y+v2.y; s.z=v1.z+v2.z; s.w=v1.w+v2.w;return s; }

inline cvec2f   subvec(const cvec2f& v1, const cvec2f& v2){ cvec2f s; s.x=v1.x-v2.x; s.y=v1.y-v2.y; return s; }
inline cvec3f   subvec(const cvec3f& v1, const cvec3f& v2){ cvec3f s; s.x=v1.x-v2.x; s.y=v1.y-v2.y; s.z=v1.z-v2.z; return s; }
inline cvec4f   subvec(const cvec4f& v1, const cvec4f& v2){ cvec4f s; s.x=v1.x-v2.x; s.y=v1.y-v2.y; s.z=v1.z-v2.z; s.w=v1.w-v2.w;return s; }

inline cvec2f   mulvec(const cvec2f& v1, const float& k){ cvec2f s; s.x=v1.x*k; s.y=v1.y*k; return s; }
inline cvec3f   mulvec(const cvec3f& v1, const float& k){ cvec3f s; s.x=v1.x*k; s.y=v1.y*k; s.z=v1.z*k; return s; }
inline cvec4f   mulvec(const cvec4f& v1, const float& k){ cvec4f s; s.x=v1.x*k; s.y=v1.y*k; s.z=v1.z*k; s.w=v1.w*k;return s; }

inline void     mulvecInplace(cvec2f& v1, const float& k){ v1.x=v1.x*k; v1.y=v1.y*k; }
inline void     mulvecInplace(cvec3f& v1, const float& k){ v1.x=v1.x*k; v1.y=v1.y*k; v1.z=v1.z*k; }
inline void     mulvecInplace(cvec4f& v1, const float& k){ v1.x=v1.x*k; v1.y=v1.y*k; v1.z=v1.z*k; v1.w=v1.w*k; }

inline cvec2f   operator+(const cvec2f& v1, const cvec2f& v2){ return addvec(v1, v2); }
inline cvec3f   operator+(const cvec3f& v1, const cvec3f& v2){ return addvec(v1, v2); }
inline cvec4f   operator+(const cvec4f& v1, const cvec4f& v2){ return addvec(v1, v2); }

inline cvec2f   operator-(const cvec2f& v1, const cvec2f& v2){ return subvec(v1, v2); }
inline cvec3f   operator-(const cvec3f& v1, const cvec3f& v2){ return subvec(v1, v2); }
inline cvec4f   operator-(const cvec4f& v1, const cvec4f& v2){ return subvec(v1, v2); }

inline cvec2f   operator*(const cvec2f& v1, const float& k){ return mulvec(v1, k); }
inline cvec3f   operator*(const cvec3f& v1, const float& k){ return mulvec(v1, k); }
inline cvec4f   operator*(const cvec4f& v1, const float& k){ return mulvec(v1, k); }

inline cvec2f   operator*(const float& k, const cvec2f& v1){ return mulvec(v1, k); }
inline cvec3f   operator*(const float& k, const cvec3f& v1){ return mulvec(v1, k); }
inline cvec4f   operator*(const float& k, const cvec4f& v1){ return mulvec(v1, k); }

inline cvec2f&  operator+=(cvec2f& v1, const cvec2f& v2){ v1.x+=v2.x; v1.y+=v2.y; return v1;}
inline cvec3f&  operator+=(cvec3f& v1, const cvec3f& v2){ v1.x+=v2.x; v1.y+=v2.y; v1.z+=v2.z; return v1;}
inline cvec4f&  operator+=(cvec4f& v1, const cvec4f& v2){ v1.x+=v2.x; v1.y+=v2.y; v1.z+=v2.z; v1.w+=v2.w; return v1;}

inline cvec2f&  operator-=(cvec2f& v1, const cvec2f& v2){ v1.x-=v2.x; v1.y-=v2.y; return v1;}
inline cvec3f&  operator-=(cvec3f& v1, const cvec3f& v2){ v1.x-=v2.x; v1.y-=v2.y; v1.z-=v2.z; return v1;}
inline cvec4f&  operator-=(cvec4f& v1, const cvec4f& v2){ v1.x-=v2.x; v1.y-=v2.y; v1.z-=v2.z; v1.w-=v2.w; return v1;}

inline cvec2f&  operator*=(cvec2f& v1, const float& k){ v1.x*=k; v1.y*=k; return v1;}
inline cvec3f&  operator*=(cvec3f& v1, const float& k){ v1.x*=k; v1.y*=k; v1.z*=k; return v1;}
inline cvec4f&  operator*=(cvec4f& v1, const float& k){ v1.x*=k; v1.y*=k; v1.z*=k; v1.w*=k; return v1;}

inline float    dotvec(const cvec2f& v1, const cvec2f& v2){ return (v1.x*v2.x + v1.y*v2.y); }
inline float    dotvec(const cvec3f& v1, const cvec3f& v2){ return (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z); }
inline float    dotvec(const cvec4f& v1, const cvec4f& v2){ return (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z + v1.w*v2.w); }

inline bool     equalvec(const cvec2f& v1, const cvec2f& v2){ return ((v1.x==v2.x)&&(v1.y==v2.y)); }
inline bool     equalvec(const cvec3f& v1, const cvec3f& v2){ return ((v1.x==v2.x)&&(v1.y==v2.y)&&(v1.z==v2.z)); }
inline bool     equalvec(const cvec4f& v1, const cvec4f& v2){ return ((v1.x==v2.x)&&(v1.y==v2.y)&&(v1.z==v2.z)&&(v1.w==v2.w)); }

inline bool     operator==(const cvec2f& v1, const cvec2f& v2){ return  equalvec(v1,v2); }
inline bool     operator==(const cvec3f& v1, const cvec3f& v2){ return  equalvec(v1,v2); }
inline bool     operator==(const cvec4f& v1, const cvec4f& v2){ return  equalvec(v1,v2); }

inline bool     operator!=(const cvec2f& v1, const cvec2f& v2){ return !equalvec(v1,v2); }
inline bool     operator!=(const cvec3f& v1, const cvec3f& v2){ return !equalvec(v1,v2); }
inline bool     operator!=(const cvec4f& v1, const cvec4f& v2){ return !equalvec(v1,v2); }

inline void     clampInplace(cvec2f& v){ clampInplace(v.x); clampInplace(v.y); }
inline void     clampInplace(cvec3f& v){ clampInplace(v.x); clampInplace(v.y); clampInplace(v.z); }
inline void     clampInplace(cvec4f& v){ clampInplace(v.x); clampInplace(v.y); clampInplace(v.z); clampInplace(v.w); }

template <class T>  inline void disp2(const T& v){ printf("% 0.2f % 0.2f",v.x, v.y); }
template <class T>  inline void disp3(const T& v){ printf("% 0.2f % 0.2f % 0.2f",v.x, v.y, v.z); }
template <class T>  inline void disp4(const T& v){ printf("% 0.2f % 0.2f % 0.2f % 0.2f",v.x, v.y, v.z, v.w); }

template <class T>  cvec2f getcvec2f(const T& v){ cvec2f v2; v2.x = v.x; v2.y = v.y; return v2; }
template <class T>  cvec3f getcvec3f(const T& v){ cvec3f v3; v3.x = v.x; v3.y = v.y; v3.z = v.z; return v3; }

template <class T>  inline float magvec(const T& v){ return sqrtf(magsqr(v)); }

template <class T>  inline void normalizeInplace(T& v){ 
  float m = magvec(v);
  if(m != 0.0f) mulvecInplace(v, (1.0f/m));
}

template <class T>  inline T lerp(const T& a, const T& b, float t) { 
  return (addvec(mulvec(a, 1.0f-t),mulvec(b, t)));
}

inline cvec3f cross( const cvec3f& v1, const cvec3f& v2 )
{
  cvec3f crossprod;
  crossprod.x = v1.y * v2.z - v1.z * v2.y;
  crossprod.y = v1.z * v2.x - v1.x * v2.z;
  crossprod.z = v1.x * v2.y - v1.y * v2.x;
  return crossprod;
}

//inline cvec3f reflect(cvec3f & n, cvec3f & d)
//{
//  cvec3f output = d - 2 *dot(n,d) * n;
//  return output;
//}
//

inline void matmultvec4f(const float * matrix, const cvec4f& v, cvec4f& ssv)
{
	ssv.x = matrix[0] * v.x + matrix[4] * v.y +  matrix[8] * v.z + matrix[12] * v.w;
	ssv.y = matrix[1] * v.x + matrix[5] * v.y +  matrix[9] * v.z + matrix[13] * v.w;
	ssv.z = matrix[2] * v.x + matrix[6] * v.y + matrix[10] * v.z + matrix[14] * v.w;
	ssv.w = matrix[3] * v.x + matrix[7] * v.y + matrix[11] * v.z + matrix[15] * v.w;
}


#endif
