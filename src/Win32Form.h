#pragma once


#ifdef _WIN32

#include "cvecs.h"

#include "Win32App.h"

class Win32Form : public Win32GUI
{
private:
public:
  HDC hdc;
  
  Win32Form(Win32GUI* parent);
  virtual ~Win32Form();

  virtual void            Create            ();
  virtual void            Show              ();
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
  virtual void            UserPaint         () = 0;
  virtual void            DrawLine          ();
  virtual void            DrawRect          ();
};

#endif
