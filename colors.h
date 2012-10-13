#ifndef colors_h
#define colors_h

#include "misc.h"

#define CLR_RED                  (0x000000ff)
#define CLR_GREEN                (0x0000ff00)
#define CLR_BLUE                 (0x00ff0000)
#define CLR_BLACK                (0x00000000)
#define CLR_WHITE                (0x00ffffff)
#define CLR_GREY                 (0x00aaaaaa)
#define CLR_DARKGREY             (0x00555555)

inline float saturatef(float v, float vmax=1.0f, float vmin=0.0f){
    return ((v < vmin) ? vmin : (v > vmax) ? vmax : v);
}

inline uint rgbaFloatToInt(float r, float g, float b, float a=0.0f){
    r = saturatef(r); g = saturatef(g); b = saturatef(b); a = saturatef(a);
    return ((uint)(a*255)<<24) | ((uint)(b*255)<<16) | ((uint)(g*255)<<8) | (uint)(r*255);
}


#endif
