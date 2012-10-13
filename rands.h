#ifndef rands_h
#define rands_h

#include <cstdlib>

inline double rand01(){
    return ((double)rand()/(double)RAND_MAX);
}

inline void shuffle(float *samp, int count){
    for(int i=0;i<(count-1);i++){
        int other = i + (rand()%(count-i));
        swap(samp[i],samp[other]);
    }
}

#endif
