#ifndef matrices_h
#define matrices_h

#include "cvecs.h"

inline void dispMat(float mat[], int size, int cols){
    printf("showing matrix:\n");
    for(int i=0; i<size; i++){
        printf("%0.3f ", mat[i]);
        if(i%cols == (cols-1)) printf("\n");
    }
}


inline float Determinant4f(const float m[16])
{
    return
        m[12]*m[9]*m[6]*m[3]-
        m[8]*m[13]*m[6]*m[3]-
        m[12]*m[5]*m[10]*m[3]+
        m[4]*m[13]*m[10]*m[3]+
        m[8]*m[5]*m[14]*m[3]-
        m[4]*m[9]*m[14]*m[3]-
        m[12]*m[9]*m[2]*m[7]+
        m[8]*m[13]*m[2]*m[7]+
        m[12]*m[1]*m[10]*m[7]-
        m[0]*m[13]*m[10]*m[7]-
        m[8]*m[1]*m[14]*m[7]+
        m[0]*m[9]*m[14]*m[7]+
        m[12]*m[5]*m[2]*m[11]-
        m[4]*m[13]*m[2]*m[11]-
        m[12]*m[1]*m[6]*m[11]+
        m[0]*m[13]*m[6]*m[11]+
        m[4]*m[1]*m[14]*m[11]-
        m[0]*m[5]*m[14]*m[11]-
        m[8]*m[5]*m[2]*m[15]+
        m[4]*m[9]*m[2]*m[15]+
        m[8]*m[1]*m[6]*m[15]-
        m[0]*m[9]*m[6]*m[15]-
        m[4]*m[1]*m[10]*m[15]+
        m[0]*m[5]*m[10]*m[15];
}

inline int GenerateInverseMatrix4f(float i[16], const float m[16])
{
    float x=Determinant4f(m);
    if (x==0) return 0;

    i[0]= (-m[13]*m[10]*m[7] +m[9]*m[14]*m[7] +m[13]*m[6]*m[11]
    -m[5]*m[14]*m[11] -m[9]*m[6]*m[15] +m[5]*m[10]*m[15])/x;
    i[4]= ( m[12]*m[10]*m[7] -m[8]*m[14]*m[7] -m[12]*m[6]*m[11]
    +m[4]*m[14]*m[11] +m[8]*m[6]*m[15] -m[4]*m[10]*m[15])/x;
    i[8]= (-m[12]*m[9]* m[7] +m[8]*m[13]*m[7] +m[12]*m[5]*m[11]
    -m[4]*m[13]*m[11] -m[8]*m[5]*m[15] +m[4]*m[9]* m[15])/x;
    i[12]=( m[12]*m[9]* m[6] -m[8]*m[13]*m[6] -m[12]*m[5]*m[10]
    +m[4]*m[13]*m[10] +m[8]*m[5]*m[14] -m[4]*m[9]* m[14])/x;
    i[1]= ( m[13]*m[10]*m[3] -m[9]*m[14]*m[3] -m[13]*m[2]*m[11]
    +m[1]*m[14]*m[11] +m[9]*m[2]*m[15] -m[1]*m[10]*m[15])/x;
    i[5]= (-m[12]*m[10]*m[3] +m[8]*m[14]*m[3] +m[12]*m[2]*m[11]
    -m[0]*m[14]*m[11] -m[8]*m[2]*m[15] +m[0]*m[10]*m[15])/x;
    i[9]= ( m[12]*m[9]* m[3] -m[8]*m[13]*m[3] -m[12]*m[1]*m[11]
    +m[0]*m[13]*m[11] +m[8]*m[1]*m[15] -m[0]*m[9]* m[15])/x;
    i[13]=(-m[12]*m[9]* m[2] +m[8]*m[13]*m[2] +m[12]*m[1]*m[10]
    -m[0]*m[13]*m[10] -m[8]*m[1]*m[14] +m[0]*m[9]* m[14])/x;
    i[2]= (-m[13]*m[6]* m[3] +m[5]*m[14]*m[3] +m[13]*m[2]*m[7]
    -m[1]*m[14]*m[7] -m[5]*m[2]*m[15] +m[1]*m[6]* m[15])/x;
    i[6]= ( m[12]*m[6]* m[3] -m[4]*m[14]*m[3] -m[12]*m[2]*m[7]
    +m[0]*m[14]*m[7] +m[4]*m[2]*m[15] -m[0]*m[6]* m[15])/x;
    i[10]=(-m[12]*m[5]* m[3] +m[4]*m[13]*m[3] +m[12]*m[1]*m[7]
    -m[0]*m[13]*m[7] -m[4]*m[1]*m[15] +m[0]*m[5]* m[15])/x;
    i[14]=( m[12]*m[5]* m[2] -m[4]*m[13]*m[2] -m[12]*m[1]*m[6]
    +m[0]*m[13]*m[6] +m[4]*m[1]*m[14] -m[0]*m[5]* m[14])/x;
    i[3]= ( m[9]* m[6]* m[3] -m[5]*m[10]*m[3] -m[9]* m[2]*m[7]
    +m[1]*m[10]*m[7] +m[5]*m[2]*m[11] -m[1]*m[6]* m[11])/x;
    i[7]= (-m[8]* m[6]* m[3] +m[4]*m[10]*m[3] +m[8]* m[2]*m[7]
    -m[0]*m[10]*m[7] -m[4]*m[2]*m[11] +m[0]*m[6]* m[11])/x;
    i[11]=( m[8]* m[5]* m[3] -m[4]*m[9]* m[3] -m[8]* m[1]*m[7]
    +m[0]*m[9]* m[7] +m[4]*m[1]*m[11] -m[0]*m[5]* m[11])/x;
    i[15]=(-m[8]* m[5]* m[2] +m[4]*m[9]* m[2] +m[8]* m[1]*m[6]
    -m[0]*m[9]* m[6] -m[4]*m[1]*m[10] +m[0]*m[5]* m[10])/x;

    return 1;
} 


inline void vtransform(const float m[16], const vec4f& vin, vec4f& vout){
    vout.x() = vin.peekx() * m[ 0] + vin.peeky() * m[ 4] + vin.peekz() * m[ 8] + vin.peekw() * m[12];
    vout.y() = vin.peekx() * m[ 1] + vin.peeky() * m[ 5] + vin.peekz() * m[ 9] + vin.peekw() * m[13];
    vout.z() = vin.peekx() * m[ 2] + vin.peeky() * m[ 6] + vin.peekz() * m[10] + vin.peekw() * m[14];
    vout.w() = vin.peekx() * m[ 3] + vin.peeky() * m[ 7] + vin.peekz() * m[11] + vin.peekw() * m[15];
}

inline void vtransform(const float m[16], const vec3f& vin, vec4f& vout){
    vec4f vh = vec4f(vin.peekx(), vin.peeky(), vin.peekz(), 1.0f);
    vtransform(m, vh, vout);
}

inline void ntransform(const float m[16], const vec3f& vin, vec3f& vout){
    vout.x() = vin.peekx() * m[ 0] + vin.peeky() * m[ 1] + vin.peekz() * m[ 2];
    vout.y() = vin.peekx() * m[ 4] + vin.peeky() * m[ 5] + vin.peekz() * m[ 6];
    vout.z() = vin.peekx() * m[ 8] + vin.peeky() * m[ 9] + vin.peekz() * m[10];
}


inline void vtransform(const float m[16], const cvec4f& vin, cvec4f& vout){
    vout.x = vin.x * m[ 0] + vin.y * m[ 4] + vin.z * m[ 8] + vin.w * m[12];
    vout.y = vin.x * m[ 1] + vin.y * m[ 5] + vin.z * m[ 9] + vin.w * m[13];
    vout.z = vin.x * m[ 2] + vin.y * m[ 6] + vin.z * m[10] + vin.w * m[14];
    vout.w = vin.x * m[ 3] + vin.y * m[ 7] + vin.z * m[11] + vin.w * m[15];
}

inline void vtransform(const float m[16], const cvec3f& vin, cvec4f& vout){
    cvec4f vh = gencvec4f(vin.x, vin.y, vin.z, 1.0f);
    vtransform(m, vh, vout);
}

inline void ntransform(const float m[16], const cvec3f& vin, cvec3f& vout){
    vout.x = vin.x * m[ 0] + vin.y * m[ 1] + vin.z * m[ 2];
    vout.y = vin.x * m[ 4] + vin.y * m[ 5] + vin.z * m[ 6];
    vout.z = vin.x * m[ 8] + vin.y * m[ 9] + vin.z * m[10];
}

#endif
