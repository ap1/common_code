#include "Win32Form.h"

// C RunTime Header Files
#include <cstdlib>
#include <tchar.h>

#include "Win32App.h"

using namespace std;

Win32Form::Win32Form()
{
  Create();
}

Win32Form::~Win32Form()
{
  Win32App::UnregisterForm(this);
}

void Win32Form::Create()
{
  hWnd = CreateWindow(GetFormClassName(), "Untitled Form", WS_OVERLAPPEDWINDOW, 
    CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, NULL, NULL, Win32App::hInstance, NULL);

  Win32App::RegisterForm(this);

  this->Show();
  this->Update();
}

void Win32Form::Show()                { ShowWindow(hWnd, SW_SHOW); }
void Win32Form::Update()              { UpdateWindow(hWnd); }
void Win32Form::Refresh(bool bErase)  { InvalidateRect(hWnd, NULL, bErase); }
void Win32Form::MainLoop()            { }

std::string Win32Form::GetTitle()
{
  return "";
}

void Win32Form::SetTitle()
{
}

cvec2i Win32Form::GetPosition()
{
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
}

void Win32Form::DrawRect(cvec2f& beg, cvec2f& end)
{
  MoveToEx(hdc, (int)beg.x, (int)beg.y, NULL);
  LineTo  (hdc, (int)end.x, (int)beg.y);
  LineTo  (hdc, (int)end.x, (int)end.y);
  LineTo  (hdc, (int)beg.x, (int)end.y);
  LineTo  (hdc, (int)beg.x, (int)beg.y);
}

void Win32Form::DrawLine(cvec2f& beg, cvec2f& end)
{
  MoveToEx(hdc, (int)beg.x, (int)beg.y, NULL);
  LineTo  (hdc, (int)end.x, (int)end.y);
}


void Win32Form::MessageHandler (UINT message, WPARAM wParam, LPARAM lParam)
{

}


LPCSTR Win32Form::GetFormClassName ()
{
  return "DefaultForm";
}
