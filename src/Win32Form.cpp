#include "Win32Form.h"

// C RunTime Header Files
#include <cstdlib>
#include <tchar.h>

using namespace std;

Win32Form::Win32Form()
{
}

Win32Form::~Win32Form()
{
}

void Win32Form::Create()
{
  hWnd = CreateWindow(windowClass, L"Untitled Form", WS_OVERLAPPEDWINDOW, 
    CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, NULL, NULL, hInstance, NULL);

  this->Show();
  this->Update();
}

void Win32Form::Show()                { ShowWindow(hWnd, SW_SHOW); }
void Win32Form::Update()              { UpdateWindow(hWnd); }
void Win32Form::Refresh(bool bErase)  { InvalidateRect(hWnd, NULL, bErase); }
void Win32Form::MainLoop()            { }

std::string Win32Form::GetTitle()
{
}

void Win32Form::SetTitle()
{
}

cvec2i Win32Form::GetPosition()
{
}

void Win32Form::SetPosition(const cvec2i& p)
{
}

cvec2i Win32Form::GetSize()
{
}

void Win32Form::SetSize(const cvec2i& sz)
{
}

cvec2i Win32Form::GetClientSize()
{
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
