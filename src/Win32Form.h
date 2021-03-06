#pragma once


#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <string>
#include <functional>
#include <algorithm>
#include <vector>
#include "cvecs.h"

enum class Win32FormState
{
  Registered,
  Destroyed,
  Count
};

class Win32Form
{
private:
public:
  HDC hdc;
  HWND hWnd;
  Win32FormState state_;
  
  Win32Form();
  virtual ~Win32Form();

  virtual void            Create            ();
  virtual void            Show              ();
  virtual void            Update            ();
  virtual void            Refresh           (bool bErase = true);
  virtual void            Destroy           ();
  virtual void            MainLoop          ();
  virtual void            MainLoopInstance  ();


  virtual std::string     GetTitle          ();
  virtual void            SetTitle          (const std::string& title);
  virtual cvec2i          GetPosition       ();
  virtual void            SetPosition       (const cvec2i& p);
  virtual cvec2i          GetSize           ();
  virtual void            SetSize           (const cvec2i& sz);
  virtual cvec2i          GetClientSize     ();
  virtual void            SetClientSize     (const cvec2i& sz);

  virtual void            Paint             ();
  virtual void            UserPaint         ();
  virtual void            DrawLine          (const cvec2f& beg, const cvec2f& end);
  virtual void            DrawRect          (const cvec2f& beg, const cvec2f& end);

  virtual LRESULT         MessageHandler    (UINT message, WPARAM wParam, LPARAM lParam);
  virtual LPCSTR          GetFormClassName  ();
};

#endif
