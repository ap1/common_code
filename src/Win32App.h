#pragma once

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <string>
#include <functional>
#include <algorithm>
#include <vector>

class Win32GUI
{
public:
  HWND hWnd;
  
  std::vector<Win32GUI*> children_;
  Win32GUI* parent_;

  Win32GUI(Win32GUI* parent) : parent_(parent)
  {
    if(parent_ != NULL)
    {
      parent_->children_.push_back(this);
    }
  }

  virtual ~Win32GUI()
  {
    if(parent_ != NULL) 
    {
      auto& plist = parent_->children_;
      plist.erase(std::remove(plist.begin(), plist.end(), this), plist.end());
    }
  }
};

class Win32App : public Win32GUI
{
private:
public:
  HINSTANCE hInstance;
  std::string sCommandLine;

  Win32App() : Win32GUI(NULL)
  {
  }

  ~Win32App()
  {
  }

  static inline LRESULT CALLBACK MessageHandler
  (Win32App* app, HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
  {
    app->MessageHandler(hWnd, message, wParam, lParam);
  }

  inline int MessageHandler(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
  {
    int wmId, wmEvent;

    switch (message)
    {
    case WM_COMMAND:
      wmId = LOWORD(wParam);
      wmEvent = HIWORD(wParam);
      // // Parse the menu selections:
      // switch (wmId)
      // {
      //   default:
      //     return DefWindowProc(hWnd, message, wParam, lParam);
      // }
      break;
    case WM_PAINT:
      // this->Paint();
      // hdc = BeginPaint(hWnd, &ps);

      // EndPaint(hWnd, &ps);
      // break;
    case WM_DESTROY:
      PostQuitMessage(0);
      break;
    default:
      return DefWindowProc(hWnd, message, wParam, lParam);
    }    
  }

  inline void Init()
  {
    hInstance = GetModuleHandle(NULL);
    sCommandLine = GetCommandLine();

    InitInstance(hInstance, SW_SHOW);
  }

  inline ATOM RegisterClass()
  {
    WNDCLASSEX wcex;

    wcex.cbSize = sizeof(WNDCLASSEX);

    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = this->MessageHandler;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = 0;
    wcex.hInstance = hInstance;
    // //wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_RECORDAPP));
    // //wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
    // //wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    // //wcex.lpszMenuName = MAKEINTRESOURCE(IDC_RECORDAPP);
    // wcex.lpszClassName = ???;
    // //wcex.hIconSm = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

    return RegisterClassEx(&wcex);
  }
};

#endif