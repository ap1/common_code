#include "Win32Form.h"

// C RunTime Header Files
#include <cstdlib>
#include <tchar.h>
#include <memory>

#include "Win32App.h"

using namespace std;

Win32Form::Win32Form()
{
  Win32App::RegisterForm(this);
  state_ = Win32FormState::Registered;
  this->Create();
}

Win32Form::~Win32Form()
{
  if(state_ != Win32FormState::Destroyed)
    this->Destroy();
}

void Win32Form::Create()
{
  hWnd = CreateWindow(GetFormClassName(), "Untitled Form", WS_OVERLAPPEDWINDOW, 
    CW_USEDEFAULT, CW_USEDEFAULT, 256, 256, NULL, NULL, Win32App::hInstance, NULL);

  if (!hWnd)
  {
    printf("hWnd is NULL!\n");
  }

  this->Show();
  this->Update();
}

void Win32Form::Show()                { int ret = ShowWindow(hWnd, SW_SHOW); } // printf("Show win returned %d\n", ret); }
void Win32Form::Update()              { UpdateWindow(hWnd); }
void Win32Form::Refresh(bool bErase)  { InvalidateRect(hWnd, NULL, bErase); }
void Win32Form::MainLoop()            { }

void Win32Form::Destroy()
{
  Win32App::UnregisterForm(this);
  state_ = Win32FormState::Destroyed;
}

void Win32Form::MainLoopInstance()    { }

std::string Win32Form::GetTitle()
{
  int len = 512;//GetWindowTextLength(hWnd);
  std::shared_ptr<char> title(new char[len+1]);
  GetWindowText(hWnd, title.get(), len);
  std::string ret = title.get();
  return ret;
}

void Win32Form::SetTitle(const std::string& title)
{
  SetWindowText(hWnd, title.c_str());
}

cvec2i Win32Form::GetPosition()
{
  //GetWindowRect(hWnd, )
  return gencvec2i(0,0);
}

void Win32Form::SetPosition(const cvec2i& p)
{
}

cvec2i Win32Form::GetSize()
{
  return gencvec2i(0,0);
}

void Win32Form::SetSize(const cvec2i& sz)
{
}

cvec2i Win32Form::GetClientSize()
{
  return gencvec2i(0,0);
}

void Win32Form::SetClientSize(const cvec2i& sz)
{
}

void Win32Form::Paint()
{
}

void Win32Form::UserPaint()
{
  //printf("base\n");
  //DrawRect(gencvec2f(40.0f,40.0f), gencvec2f(100.0f,100.0f));
  //DrawLine(gencvec2f(40.0f,40.0f), gencvec2f(100.0f,100.0f));
}

void Win32Form::DrawRect(const cvec2f& beg, const cvec2f& end)
{
  MoveToEx(hdc, (int)beg.x, (int)beg.y, NULL);
  LineTo  (hdc, (int)end.x, (int)beg.y);
  LineTo  (hdc, (int)end.x, (int)end.y);
  LineTo  (hdc, (int)beg.x, (int)end.y);
  LineTo  (hdc, (int)beg.x, (int)beg.y);

}

void Win32Form::DrawLine(const cvec2f& beg, const cvec2f& end)
{
  MoveToEx(hdc, (int)beg.x, (int)beg.y, NULL);
  LineTo  (hdc, (int)end.x, (int)end.y);
}


LRESULT Win32Form::MessageHandler (UINT message, WPARAM wParam, LPARAM lParam)
{

  int wmId, wmEvent;
  PAINTSTRUCT ps;

  switch (message)
  {
  // case WM_COMMAND:
  //   wmId = LOWORD(wParam);
  //   wmEvent = HIWORD(wParam);
  //   // // Parse the menu selections:
  //   // switch (wmId)
  //   // {
  //   //   default:
  //   //     return DefWindowProc(hWnd, message, wParam, lParam);
  //   // }
  //   break;
  case WM_PAINT:
    hdc = BeginPaint(hWnd, &ps);
    this->Paint();
    this->UserPaint();
    EndPaint(hWnd, &ps);
    return DefWindowProc(hWnd, message, wParam, lParam);
    break;
  case WM_DESTROY:
    this->Destroy();
    PostQuitMessage(0);
    return DefWindowProc(hWnd, message, wParam, lParam);
    break;
  default:
    return DefWindowProc(hWnd, message, wParam, lParam);
  }
}


LPCSTR Win32Form::GetFormClassName ()
{
  return "DefaultForm";
}
