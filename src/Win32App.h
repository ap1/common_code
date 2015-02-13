#pragma once

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <string>
#include <functional>
#include <algorithm>
#include <vector>

#include "Win32Form.h"

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

enum class Win32AppState
{
  Uninitialized,
  Initialized,
  Running,
  Done,
  Count
};

class Win32App
{
private:
public:
  static HINSTANCE hInstance;
  static std::string sCommandLine;
  static std::vector<Win32Form*> forms_;
  static Win32AppState state_;

  static inline void RegisterForm(Win32Form* form)
  {
    forms_.push_back(form);

    WNDCLASSEX wcex;

    wcex.cbSize         = sizeof(WNDCLASSEX);
    wcex.style          = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc    = Win32App::MessageHandler;
    wcex.cbClsExtra     = 0;
    wcex.cbWndExtra     = 0;
    wcex.hInstance      = hInstance;
    wcex.hIcon          = 0;
    wcex.hCursor        = 0;
    wcex.hbrBackground  = 0;
    wcex.lpszMenuName   = 0;
    wcex.lpszClassName  = form->GetFormClassName();
    wcex.hIconSm        = 0;

    int ret = RegisterClassEx(&wcex);
    printf("Regisering %s returned %d\n", form->GetFormClassName(), ret);
  }

  static inline void UnregisterForm(Win32Form* form)
  {
    forms_.erase(
      std::remove(forms_.begin(), forms_.end(), form), 
      forms_.end());
  }

  static inline Win32Form* FindForm(HWND hWnd)
  {
    for(auto pform : forms_)
    {
      if(pform->hWnd == hWnd) return pform;
    }
    return NULL;
  }

  static inline LRESULT CALLBACK MessageHandler(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
  {
    auto pform =  FindForm(hWnd);

    if(pform != NULL) return pform->MessageHandler(message, wParam, lParam);
    else              return DefWindowProc  (hWnd, message, wParam, lParam);

    // int wmId, wmEvent;

    // switch (message)
    // {
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
    // case WM_PAINT:
    //   // this->Paint();
    //   // hdc = BeginPaint(hWnd, &ps);

    //   // EndPaint(hWnd, &ps);
    //   // break;
    // case WM_DESTROY:
    //   PostQuitMessage(0);
    //   break;
    // default:
    //   return DefWindowProc(hWnd, message, wParam, lParam);
    // }    

  }

  static inline void Init()
  {
    hInstance = GetModuleHandle(NULL);
    sCommandLine = GetCommandLine();

    //InitInstance(hInstance, SW_SHOW);
  }

  static void MainLoopInstance()
  {
    int ret;
    MSG msg;

    if ((ret = PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) != 0)
    {
      if (ret == -1)
      {
      }
      else //if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
      {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
      }
    }

    for(auto f : forms_)
    {
      f->MainLoopInstance();
    }
  }

  static void MainLoop()
  {
    while(state_ != Win32AppState::Done)
    {
      MainLoopInstance();
    }
  }

  //static inline ATOM RegisterClass()
  //{
  //  WNDCLASSEX wcex;

  //  wcex.cbSize = sizeof(WNDCLASSEX);

  //  wcex.style = CS_HREDRAW | CS_VREDRAW;
  //  wcex.lpfnWndProc = Win32App::MessageHandler;
  //  wcex.cbClsExtra = 0;
  //  wcex.cbWndExtra = 0;
  //  wcex.hInstance = hInstance;
  //  // //wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_RECORDAPP));
  //  // //wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
  //  // //wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
  //  // //wcex.lpszMenuName = MAKEINTRESOURCE(IDC_RECORDAPP);
  //  // wcex.lpszClassName = ???;
  //  // //wcex.hIconSm = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

  //  return RegisterClassEx(&wcex);
  //}
};

#endif