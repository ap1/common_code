#pragma once


#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <string>
#include <functional>
#include <algorithm>
#include <vector>
#include "cvecs.h"

class Win32Form
{
private:
public:
  HDC hdc;
  HWND hWnd;
  
  Win32Form();
  virtual ~Win32Form();

  virtual void            Create            ();
  virtual void            Show              ();
  virtual void            Update            ();
  virtual void            Refresh           (bool bErase = true);
  virtual void            MainLoop          ();

  virtual std::string     GetTitle          ();
  virtual void            SetTitle          ();
  virtual cvec2i          GetPosition       ();
  virtual void            SetPosition       (const cvec2i& p);
  virtual cvec2i          GetSize           ();
  virtual void            SetSize           (const cvec2i& sz);
  virtual cvec2i          GetClientSize     ();
  virtual void            SetClientSize     (const cvec2i& sz);

  virtual void            Paint             ();
  virtual void            UserPaint         ();
  virtual void            DrawLine          (cvec2f& beg, cvec2f& end);
  virtual void            DrawRect          (cvec2f& beg, cvec2f& end);

  virtual void            MessageHandler    (UINT message, WPARAM wParam, LPARAM lParam);
  virtual LPCSTR          GetFormClassName  ();
};

#endif
