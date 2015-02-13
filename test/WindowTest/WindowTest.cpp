

#include "../../src/everything.h"
#include "../../src/Win32Form.h"
#include "../../src/Win32App.h"


class MainForm : public Win32Form
{
public:
  MainForm() {}
  ~MainForm() {}
};


int main()
{
  printf("hello\n");
  Win32App::Init();
  MainForm f, f1, f2;

  Win32App::MainLoop();
  return 0;
}