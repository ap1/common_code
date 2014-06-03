#pragma once

#include <windows.h>
#include <windowsx.h>



class FPSMeter
{
public:
  long long ticks_per_sec;
  long long cur_tick;
  long long last_tick;
  long long counter_tick;
  float frame_diff_sec, counter_diff_sec;
  int counter;
  bool started;
  float fps;

  FPSMeter()
  {
    started = false;
    QueryPerformanceFrequency((LARGE_INTEGER *)&ticks_per_sec);
  }

  inline void RegisterFrame()
  {
    QueryPerformanceCounter((LARGE_INTEGER *)&cur_tick);

    if(!started)
    {
      started = true;
      counter = 0;
      counter_tick = cur_tick;
      last_tick = cur_tick;
    }
    else
    {
      frame_diff_sec = (float)((double)(cur_tick - last_tick) / (double) ticks_per_sec);
      counter_diff_sec = (float)((double)(cur_tick - counter_tick) / (double) ticks_per_sec);

      if(counter_diff_sec >= 3.0f && counter > 0)
      {
        fps = (float)counter / counter_diff_sec;
        printf("%0.1f fps\n", fps);
        counter_tick = cur_tick;
        counter = 0;
      } else {
        counter++;
      }
      last_tick = cur_tick;
    }
  }
};


class Stopwatch
{
public:
  long long ticks_per_sec;
  long long cur_tick;
  long long start_tick;

  bool started;

  Stopwatch()
  {
    started = false;
    QueryPerformanceFrequency((LARGE_INTEGER *)&ticks_per_sec);
  }

  inline float GetTime()
  {
    if(started)
    {
      QueryPerformanceCounter((LARGE_INTEGER *)&cur_tick);
      return (float)((double)(cur_tick - start_tick) / (double) ticks_per_sec);
    }
    else
    {
      printf("stopwatch not running!\n");
      return 0.0f;
    }
  }

  inline void Reset()
  {
    started = true;
    QueryPerformanceCounter((LARGE_INTEGER *)&start_tick);
  }

  inline void Stop()
  {
    started = false;
  }
};