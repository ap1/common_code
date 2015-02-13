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

class Win32App
{
private:
public:
  static HINSTANCE hInstance;
  static std::string sCommandLine;
  static std::vector<Win32Form*> forms_;

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

    RegisterClassEx(&wcex);
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


  // static inline LRESULT CALLBACK MessageHandler
  // (Win32App* app, HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
  // {
  //   app->MessageHandler(hWnd, message, wParam, lParam);
  // }

  static inline LRESULT CALLBACK MessageHandler(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
  {
    auto pform =  FindForm(hWnd);

    if(pform != NULL) 
      pform->MessageHandler(message, wParam, lParam);

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

    return 0;
  }

  static inline void Init()
  {
    hInstance = GetModuleHandle(NULL);
    sCommandLine = GetCommandLine();

    //InitInstance(hInstance, SW_SHOW);
  }

  static inline ATOM RegisterClass()
  {
    WNDCLASSEX wcex;

    wcex.cbSize = sizeof(WNDCLASSEX);

    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = Win32App::MessageHandler;
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