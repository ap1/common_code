

#include "../../src/everything.h"
#include "../../src/Win32Form.h"
#include "../../src/Win32App.h"


class MainForm : public Win32Form
{
public:
  MainForm() : Win32Form()
  {
    SetTitle("Window 1");
    Refresh();
  }
  ~MainForm() { }

  virtual void UserPaint()
  {
    //printf("derived\n");
    DrawRect(gencvec2f(20.0f,20.0f), gencvec2f(200.0f,200.0f));
    DrawLine(gencvec2f(20.0f,20.0f), gencvec2f(200.0f,200.0f));
  }

  virtual LPCSTR GetFormClassName ()
  {
    return "MainForm";
  }
};


int main()
{
  printf("hello\n");
  Win32App::Init();
  MainForm f;
  Win32App::MainLoop();
  return 0;
}