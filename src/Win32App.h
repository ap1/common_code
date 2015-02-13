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
  static                  HINSTANCE hInstance;
  static                  std::string sCommandLine;
  static                  std::vector<Win32Form*> forms_;
  static                  Win32AppState state_;
  static void             RegisterForm(Win32Form* form);
  static void             UnregisterForm(Win32Form* form);
  static Win32Form*       FindForm(HWND hWnd);
  static LRESULT CALLBACK MessageHandler(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
  static void             Init();
  static void             MainLoopInstance();
  static void             MainLoop();
};

#endif